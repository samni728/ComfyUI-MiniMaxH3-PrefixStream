"""Smart Seam Frame Selector, CIE LAB Color/Luminance Transfer, Optical-Flow Micro-Blend, and Zero-Click Audio Stitching.

Designed for MiniMax H3 PrefixStream multi-segment continuation:
1. Searches the shared overlap window for the minimum-discontinuity frame cut (pixel + luma + Sobel edge + velocity continuity)
   without altering timeline duration when max_phase_shift == 0.
2. Eliminates subtle exposure, contrast, and warm/cool tint shifts across segments via clamped CIE L*a*b* mean/std transfer
   with optional anchor-frame pull to prevent cumulative multi-segment color drift.
3. Supports hard cut, cosine S-curve alpha blend, or pure-PyTorch coarse-to-fine Lucas-Kanade optical-flow warp blend.
4. Harmonizes audio sample rates/channels and applies DC-balanced complementary cosine crossfades with zero clicks.
"""

from typing import Any, Dict, Optional, Tuple
import json
import math
import torch
import torch.nn.functional as F

from .seam_protector import (
    _rgb_luminance,
    _standardize_audio_dict,
    _standardize_image_tensor,
    fit_audio_length,
)


# ---------------------------------------------------------------------------
# Perceptual CIE L*a*b* Conversion (Pure PyTorch, D65 White Point)
# ---------------------------------------------------------------------------

def _srgb_to_linear(rgb: torch.Tensor) -> torch.Tensor:
    rgb = rgb.clamp(0.0, 1.0)
    return torch.where(rgb <= 0.04045, rgb / 12.92, ((rgb + 0.055) / 1.055).pow(2.4))


def _linear_to_srgb(lin: torch.Tensor) -> torch.Tensor:
    lin = lin.clamp(0.0, 1.0)
    return torch.where(lin <= 0.0031308, lin * 12.92, 1.055 * (lin.clamp_min(1e-12).pow(1.0 / 2.4)) - 0.055).clamp(0.0, 1.0)


def rgb_to_lab(images: torch.Tensor) -> torch.Tensor:
    """Convert sRGB tensor [..., 3] in [0, 1] to CIE L*a*b* (L in [0, 100], a/b in ~[-128, 127])."""
    orig_dtype = images.dtype
    rgb = _srgb_to_linear(images[..., :3].float())

    r, g, b = rgb[..., 0], rgb[..., 1], rgb[..., 2]
    # sRGB to XYZ (D65)
    x = r * 0.4124564 + g * 0.3575761 + b * 0.1804375
    y = r * 0.2126729 + g * 0.7151522 + b * 0.0721750
    z = r * 0.0193339 + g * 0.1191920 + b * 0.9503041

    # Normalize by D65 reference white
    xr = x / 0.95047
    yr = y / 1.00000
    zr = z / 1.08883

    eps = 216.0 / 24389.0  # (6/29)^3
    kappa = 24389.0 / 27.0

    def f(t: torch.Tensor) -> torch.Tensor:
        return torch.where(t > eps, t.clamp_min(1e-12).pow(1.0 / 3.0), (kappa * t + 16.0) / 116.0)

    fx, fy, fz = f(xr), f(yr), f(zr)
    L = 116.0 * fy - 16.0
    a = 500.0 * (fx - fy)
    b_ch = 200.0 * (fy - fz)
    return torch.stack([L, a, b_ch], dim=-1).to(orig_dtype if orig_dtype in (torch.float32, torch.float64) else torch.float32)


def lab_to_rgb(lab: torch.Tensor, target_dtype: torch.dtype = torch.float32) -> torch.Tensor:
    """Convert CIE L*a*b* tensor [..., 3] back to clamped sRGB [0, 1]."""
    lab_f = lab.float()
    L, a, b_ch = lab_f[..., 0], lab_f[..., 1], lab_f[..., 2]

    fy = (L + 16.0) / 116.0
    fx = fy + a / 500.0
    fz = fy - b_ch / 200.0

    delta = 6.0 / 29.0

    def f_inv(t: torch.Tensor) -> torch.Tensor:
        return torch.where(t > delta, t.pow(3.0), 3.0 * (delta ** 2) * (t - 4.0 / 29.0)).clamp_min(0.0)

    x = 0.95047 * f_inv(fx)
    y = 1.00000 * f_inv(fy)
    z = 1.08883 * f_inv(fz)

    # XYZ (D65) to linear sRGB
    r = x * 3.2404542 - y * 1.5371385 - z * 0.4985314
    g = -x * 0.9692660 + y * 1.8760108 + z * 0.0415560
    b = x * 0.0556434 - y * 0.2040259 + z * 1.0572252

    srgb = _linear_to_srgb(torch.stack([r, g, b], dim=-1))
    return srgb.to(target_dtype)


# ---------------------------------------------------------------------------
# Spatial Gradient & Downsampled Feature Helpers
# ---------------------------------------------------------------------------

def _downsample_frames(images: torch.Tensor, max_side: int = 96) -> torch.Tensor:
    """Fast area downsample of [F, H, W, 3] frames for rapid seam search & flow."""
    rgb = images[..., :3].float()
    f, h, w, _ = rgb.shape
    if max(h, w) <= max_side:
        return rgb
    scale = float(max_side) / float(max(h, w))
    nh = max(8, int(round(h * scale)))
    nw = max(8, int(round(w * scale)))
    chw = rgb.permute(0, 3, 1, 2)
    down = F.interpolate(chw, size=(nh, nw), mode="area")
    return down.permute(0, 2, 3, 1)


def _spatial_edge_magnitude(rgb_frames: torch.Tensor) -> torch.Tensor:
    """Compute Sobel-like spatial gradient magnitude [F, H-2, W-2] on luminance."""
    luma = _rgb_luminance(rgb_frames)
    if luma.shape[-2] < 3 or luma.shape[-1] < 3:
        return torch.zeros_like(luma)
    gx = luma[:, 1:-1, 2:] - luma[:, 1:-1, :-2]
    gy = luma[:, 2:, 1:-1] - luma[:, :-2, 1:-1]
    return (gx.square() + gy.square() + 1e-12).sqrt()


# ---------------------------------------------------------------------------
# 1. Smart Seam Frame Selector
# ---------------------------------------------------------------------------

def find_optimal_seam_cut(
    prev_images: torch.Tensor,
    curr_images: torch.Tensor,
    overlap_frames: int,
    search_radius: int = 12,
    max_phase_shift: int = 0,
) -> Dict[str, Any]:
    """Search within the overlap window for the optimal seam cut index.

    Let W = min(overlap_frames, Fp, Fc).
    In nominal hard-trim stitching, prev keeps all Fp frames and curr drops W frames (cut_index k = W).
    For candidate k in [k_min, k_max] and phase_shift in [-max_phase_shift, +max_phase_shift]:
      - prev_keep_frames = Fp - W + k   (prev ends at frame index prev_keep_frames - 1)
      - curr_trim_frames = k + phase_shift (curr starts at frame index curr_trim_frames)
    When phase_shift == 0, total stitched frames = (Fp - W + k) + (Fc - k) = Fp + Fc - W,
    strictly preserving exact AV timeline duration while placing the seam at the quietest,
    lowest-discontinuity frame pair inside the overlap window.
    """
    prev_std = _standardize_image_tensor(prev_images)
    curr_std = _standardize_image_tensor(curr_images)

    if prev_std is None or curr_std is None or prev_std.shape[0] == 0 or curr_std.shape[0] == 0:
        return {
            "overlap_window": 0,
            "best_cut_k": 0,
            "phase_shift": 0,
            "prev_keep_frames": int(prev_std.shape[0]) if prev_std is not None else 0,
            "curr_trim_frames": max(0, int(overlap_frames)),
            "seam_cost": 0.0,
            "nominal_seam_cost": 0.0,
            "improvement_pct": 0.0,
            "candidates_evaluated": 0,
        }

    fp = int(prev_std.shape[0])
    fc = int(curr_std.shape[0])
    w = min(max(0, int(overlap_frames)), fp, fc - 1 if fc > 1 else fc)

    if w <= 1:
        return {
            "overlap_window": w,
            "best_cut_k": w,
            "phase_shift": 0,
            "prev_keep_frames": fp,
            "curr_trim_frames": w,
            "seam_cost": 0.0,
            "nominal_seam_cost": 0.0,
            "improvement_pct": 0.0,
            "candidates_evaluated": 1,
        }

    radius = max(0, int(search_radius))
    max_shift = max(0, int(max_phase_shift))

    # Extract only the tail of prev and head of curr needed for analysis
    prev_win_start = max(0, fp - w - 2)
    curr_win_end = min(fc, w + max_shift + 3)

    prev_small = _downsample_frames(prev_std[prev_win_start:], max_side=96)
    curr_small = _downsample_frames(curr_std[:curr_win_end].to(prev_std.device), max_side=96)

    prev_luma = _rgb_luminance(prev_small)
    curr_luma = _rgb_luminance(curr_small)
    prev_edge = _spatial_edge_magnitude(prev_small)
    curr_edge = _spatial_edge_magnitude(curr_small)

    # Candidate k values inside [1, w]. Center search across [max(1, w - radius), w]
    # or if radius >= w, search all interior overlap frames [1, w].
    if radius == 0:
        k_candidates = [w]
    else:
        # Prefer interior overlap frames [max(1, w - radius), w] avoiding frame 0 VAE warmup
        min_safe_k = 2 if w >= 4 else 1
        k_lo = max(min_safe_k, w - radius)
        k_hi = w
        k_candidates = list(range(k_lo, k_hi + 1))
        if w not in k_candidates:
            k_candidates.append(w)

    def _eval_candidate(k: int, shift: int) -> Optional[float]:
        prev_keep = fp - w + k
        curr_start = k + shift
        if prev_keep < 1 or prev_keep > fp:
            return None
        if curr_start < 0 or curr_start >= fc:
            return None

        p_last_idx = (prev_keep - 1) - prev_win_start
        c_first_idx = curr_start
        if p_last_idx < 0 or p_last_idx >= prev_small.shape[0]:
            return None
        if c_first_idx < 0 or c_first_idx >= curr_small.shape[0]:
            return None

        p_last = prev_small[p_last_idx]
        c_first = curr_small[c_first_idx]

        # 1. Direct synchronous overlap alignment error (when corresponding same-time frames exist)
        align_terms = []
        # prev[prev_keep] corresponds in time to curr[curr_start] when prev_keep < fp
        if p_last_idx + 1 < prev_small.shape[0]:
            p_same = prev_small[p_last_idx + 1]
            align_terms.append(float((p_same - c_first).abs().mean().item()))
            align_terms.append(0.5 * float((prev_edge[p_last_idx + 1] - curr_edge[c_first_idx]).abs().mean().item()))
        # curr[curr_start - 1] corresponds in time to prev[prev_keep - 1] when curr_start >= 1
        if c_first_idx - 1 >= 0:
            c_prev_same = curr_small[c_first_idx - 1]
            align_terms.append(float((p_last - c_prev_same).abs().mean().item()))
            align_terms.append(0.5 * float((prev_edge[p_last_idx] - curr_edge[c_first_idx - 1]).abs().mean().item()))

        # 2. Across-seam transition error compensated by local motion velocity
        v_seam = c_first - p_last
        velocities = []
        if p_last_idx - 1 >= 0:
            velocities.append(prev_small[p_last_idx] - prev_small[p_last_idx - 1])
        if c_first_idx + 1 < curr_small.shape[0]:
            velocities.append(curr_small[c_first_idx + 1] - curr_small[c_first_idx])

        if velocities:
            v_expected = sum(velocities) / float(len(velocities))
            motion_err = float((v_seam - v_expected).abs().mean().item())
        else:
            motion_err = float(v_seam.abs().mean().item())

        # 3. Global luma & RGB mean jump across the seam
        luma_jump = abs(float(prev_luma[p_last_idx].mean().item()) - float(curr_luma[c_first_idx].mean().item()))
        rgb_mean_jump = float((p_last.mean(dim=(0, 1)) - c_first.mean(dim=(0, 1))).abs().mean().item())
        edge_jump = float((prev_edge[p_last_idx] - curr_edge[c_first_idx]).abs().mean().item())

        align_err = sum(align_terms) / float(len(align_terms)) if align_terms else (motion_err + 0.5 * edge_jump)

        # Gentle regularization against non-zero phase shift so strict sync wins ties
        shift_penalty = 0.004 * abs(shift)
        # Tiny preference against exact boundary k == w where unmasked denoiser step begins
        unmasked_edge_penalty = 0.0015 if (k == w and w >= 4) else 0.0

        return (
            0.40 * align_err
            + 0.30 * motion_err
            + 0.15 * luma_jump
            + 0.10 * rgb_mean_jump
            + 0.05 * edge_jump
            + shift_penalty
            + unmasked_edge_penalty
        )

    nominal_cost = _eval_candidate(w, 0)
    if nominal_cost is None:
        nominal_cost = 0.0

    best_cost = float("inf")
    best_k = w
    best_shift = 0
    evaluated = 0

    for k in k_candidates:
        for shift in range(-max_shift, max_shift + 1):
            cost = _eval_candidate(k, shift)
            if cost is None:
                continue
            evaluated += 1
            if cost < best_cost:
                best_cost = cost
                best_k = k
                best_shift = shift

    if not math.isfinite(best_cost):
        best_cost = nominal_cost
        best_k = w
        best_shift = 0

    prev_keep = fp - w + best_k
    curr_trim = best_k + best_shift
    improvement_pct = 0.0
    if nominal_cost > 1e-8 and best_cost <= nominal_cost:
        improvement_pct = round((nominal_cost - best_cost) / nominal_cost * 100.0, 2)

    return {
        "overlap_window": w,
        "best_cut_k": int(best_k),
        "phase_shift": int(best_shift),
        "prev_keep_frames": int(prev_keep),
        "curr_trim_frames": int(curr_trim),
        "prev_tail_trimmed": int(fp - prev_keep),
        "seam_cost": round(float(best_cost), 6),
        "nominal_seam_cost": round(float(nominal_cost), 6),
        "improvement_pct": improvement_pct,
        "candidates_evaluated": evaluated,
    }


def smart_select_seam_and_trim(
    prev_images: Optional[torch.Tensor],
    curr_images: Optional[torch.Tensor],
    overlap_frames: int = 39,
    search_radius: int = 12,
    max_phase_shift: int = 0,
    prev_audio: Optional[Dict[str, Any]] = None,
    curr_audio: Optional[Dict[str, Any]] = None,
    fps: float = 24.0,
) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor], Optional[Dict[str, Any]], Optional[Dict[str, Any]], Dict[str, Any]]:
    """Apply Smart Seam Selection to both IMAGE and AUDIO streams synchronously.

    Preserves discarded overlap tails/heads as metadata (`minimax_overlap_tail` / `minimax_audio_context`)
    so downstream crossfade stitchers can blend across the chosen cut without losing timeline sync.
    """
    prev_std = _standardize_image_tensor(prev_images)
    curr_std = _standardize_image_tensor(curr_images)
    prev_aud = _standardize_audio_dict(prev_audio)
    curr_aud = _standardize_audio_dict(curr_audio)

    if prev_std is None or prev_std.shape[0] == 0 or curr_std is None or curr_std.shape[0] == 0:
        report = {
            "mode": "initial_or_passthrough",
            "overlap_window": 0,
            "best_cut_k": 0,
            "phase_shift": 0,
            "prev_keep_frames": int(prev_std.shape[0]) if prev_std is not None else 0,
            "curr_trim_frames": 0,
            "seam_cost": 0.0,
            "nominal_seam_cost": 0.0,
            "improvement_pct": 0.0,
        }
        return prev_std, curr_std, prev_aud, curr_aud, report

    plan = find_optimal_seam_cut(
        prev_images=prev_std,
        curr_images=curr_std,
        overlap_frames=overlap_frames,
        search_radius=search_radius,
        max_phase_shift=max_phase_shift,
    )
    plan["mode"] = "smart_seam_selected"

    prev_keep = plan["prev_keep_frames"]
    curr_trim = plan["curr_trim_frames"]

    aligned_prev_img = prev_std[:prev_keep]
    trimmed_curr_img = curr_std[curr_trim:]

    aligned_prev_aud = prev_aud
    if prev_aud is not None and "waveform" in prev_aud:
        sr_p = int(prev_aud.get("sample_rate", 32000))
        keep_samples = int(round(prev_keep / float(fps) * sr_p))
        aligned_prev_aud = dict(prev_aud, waveform=fit_audio_length(prev_aud["waveform"], keep_samples), sample_rate=sr_p)

    trimmed_curr_aud = curr_aud
    if curr_aud is not None and "waveform" in curr_aud:
        sr_c = int(curr_aud.get("sample_rate", 32000))
        head_samples = int(round(curr_trim / float(fps) * sr_c))
        kept_frames = int(trimmed_curr_img.shape[0])
        want_samples = int(round(kept_frames / float(fps) * sr_c))
        cw = curr_aud["waveform"]
        body = fit_audio_length(cw[..., head_samples:head_samples + want_samples], want_samples)
        trimmed_curr_aud = dict(curr_aud, waveform=body, sample_rate=sr_c)
        if 0 < head_samples <= cw.shape[-1]:
            trimmed_curr_aud["minimax_audio_context"] = cw[..., :head_samples]

    return aligned_prev_img, trimmed_curr_img, aligned_prev_aud, trimmed_curr_aud, plan


# ---------------------------------------------------------------------------
# 2. Boundary CIE LAB Color & Luminance Transfer + Optical-Flow / Cosine Blend
# ---------------------------------------------------------------------------

def _lab_robust_stats(lab_frames: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """Compute per-channel (mean, std) in CIE LAB [3], excluding extreme specular/black pixels when possible."""
    flat = lab_frames.reshape(-1, 3).float()
    if flat.shape[0] == 0:
        return torch.zeros(3, device=lab_frames.device), torch.ones(3, device=lab_frames.device)
    L = flat[:, 0]
    valid = (L > 3.0) & (L < 97.0)
    if int(valid.sum().item()) >= max(16, int(flat.shape[0] * 0.05)):
        sub = flat[valid]
    else:
        sub = flat
    mean = sub.mean(dim=0)
    std = sub.std(dim=0, unbiased=False).clamp_min(1e-4)
    return mean, std


def apply_lab_color_match(
    curr_images: torch.Tensor,
    ref_prev_context: torch.Tensor,
    curr_overlap_context: Optional[torch.Tensor] = None,
    anchor_image: Optional[torch.Tensor] = None,
    mode: str = "lab_mean_std",
    strength: float = 0.85,
    anchor_pull_strength: float = 0.25,
    decay_frames: int = 0,
    max_scale_delta: float = 0.25,
    max_luma_shift: float = 25.0,
    max_chroma_shift: float = 24.0,
) -> Tuple[torch.Tensor, Dict[str, Any]]:
    """Transfer luminance and chrominance statistics in CIE LAB space to eliminate seam tint/exposure jumps."""
    curr_std = _standardize_image_tensor(curr_images)
    prev_ctx = _standardize_image_tensor(ref_prev_context)
    if curr_std is None or curr_std.shape[0] == 0 or prev_ctx is None or prev_ctx.shape[0] == 0 or mode == "none" or strength <= 1e-5:
        return curr_std, {"applied": False, "mode": mode}

    device = curr_std.device
    orig_dtype = curr_std.dtype
    has_alpha = curr_std.shape[-1] == 4
    alpha_ch = curr_std[..., 3:4] if has_alpha else None

    src_ctx = _standardize_image_tensor(curr_overlap_context)
    if src_ctx is None or src_ctx.shape[0] == 0:
        src_ctx = curr_std[:min(8, curr_std.shape[0])]

    # Compute stats on downsampled overlap windows for speed and spatial robustness
    ref_lab = rgb_to_lab(_downsample_frames(prev_ctx.to(device), max_side=128))
    src_lab = rgb_to_lab(_downsample_frames(src_ctx.to(device), max_side=128))

    mu_ref, std_ref = _lab_robust_stats(ref_lab)
    mu_src, std_src = _lab_robust_stats(src_lab)

    # Blend target stats gently toward Shot 1 anchor image if provided (prevents multi-segment drift!)
    anchor_used = False
    if anchor_image is not None and anchor_pull_strength > 1e-4:
        anc_std = _standardize_image_tensor(anchor_image)
        if anc_std is not None and anc_std.shape[0] > 0:
            anc_lab = rgb_to_lab(_downsample_frames(anc_std[:4].to(device), max_side=128))
            mu_anc, std_anc = _lab_robust_stats(anc_lab)
            w_anc = min(1.0, max(0.0, float(anchor_pull_strength)))
            mu_ref = (1.0 - w_anc) * mu_ref + w_anc * mu_anc
            std_ref = (1.0 - w_anc) * std_ref + w_anc * std_anc
            anchor_used = True

    raw_scale = (std_ref / std_src).clamp(1.0 - max_scale_delta, 1.0 + max_scale_delta)
    # Apply transfer
    curr_lab = rgb_to_lab(curr_std[..., :3])

    # Target mapped LAB: (x - mu_src) * scale + mu_ref, with shift bounds
    eff_strength = min(1.0, max(0.0, float(strength)))
    scale = 1.0 + eff_strength * (raw_scale - 1.0)
    delta_mu = mu_ref - mu_src
    delta_mu_clamped = torch.stack([
        delta_mu[0].clamp(-max_luma_shift, max_luma_shift),
        delta_mu[1].clamp(-max_chroma_shift, max_chroma_shift),
        delta_mu[2].clamp(-max_chroma_shift, max_chroma_shift),
    ]) * eff_strength

    if mode == "luma_only":
        scale = torch.stack([scale[0], torch.tensor(1.0, device=device), torch.tensor(1.0, device=device)])
        delta_mu_clamped = torch.stack([delta_mu_clamped[0], torch.tensor(0.0, device=device), torch.tensor(0.0, device=device)])

    n_frames = int(curr_lab.shape[0])
    if decay_frames and int(decay_frames) > 0:
        n_decay = min(int(decay_frames), n_frames)
        weights = torch.zeros(n_frames, dtype=torch.float32, device=device)
        t = torch.linspace(0.0, 1.0, n_decay, dtype=torch.float32, device=device)
        weights[:n_decay] = 0.5 + 0.5 * torch.cos(math.pi * t)
        w_view = weights.view(n_frames, 1, 1, 1)
    else:
        w_view = torch.ones((n_frames, 1, 1, 1), dtype=torch.float32, device=device)

    frame_scale = 1.0 + w_view * (scale.view(1, 1, 1, 3) - 1.0)
    frame_shift = w_view * delta_mu_clamped.view(1, 1, 1, 3)

    mu_src_view = mu_src.view(1, 1, 1, 3)
    matched_lab = (curr_lab - mu_src_view) * frame_scale + mu_src_view + frame_shift
    matched_lab[..., 0] = matched_lab[..., 0].clamp(0.0, 100.0)
    matched_lab[..., 1:] = matched_lab[..., 1:].clamp(-127.0, 127.0)

    out_rgb = lab_to_rgb(matched_lab, target_dtype=orig_dtype)
    if has_alpha and alpha_ch is not None:
        out_rgb = torch.cat([out_rgb, alpha_ch], dim=-1)

    diag = {
        "applied": True,
        "mode": mode,
        "anchor_used": anchor_used,
        "delta_L": round(float(delta_mu_clamped[0].item()), 3),
        "delta_a": round(float(delta_mu_clamped[1].item()), 3),
        "delta_b": round(float(delta_mu_clamped[2].item()), 3),
        "scale_L": round(float(scale[0].item()), 4),
    }
    return out_rgb, diag


def _estimate_dense_flow_lk(frame_a: torch.Tensor, frame_b: torch.Tensor, window: int = 5) -> torch.Tensor:
    """Lightweight coarse-to-fine Lucas-Kanade dense optical flow from frame_a to frame_b.

    Pure PyTorch implementation (zero OpenCV dependency).
    Inputs: frame_a, frame_b of shape [B, H, W, 3] in [0, 1].
    Returns: flow [B, H, W, 2] in pixel coordinates (dx, dy) such that frame_b(x + dx, y + dy) ~ frame_a(x, y).
    """
    b_sz, h_orig, w_orig, _ = frame_a.shape
    la_full = _rgb_luminance(frame_a).unsqueeze(1)  # [B, 1, H, W]
    lb_full = _rgb_luminance(frame_b).unsqueeze(1)

    max_work_side = 160
    work_scale = min(1.0, float(max_work_side) / float(max(h_orig, w_orig, 1)))
    wh = max(16, int(round(h_orig * work_scale)))
    ww = max(16, int(round(w_orig * work_scale)))

    la_work = F.interpolate(la_full, size=(wh, ww), mode="bilinear", align_corners=False)
    lb_work = F.interpolate(lb_full, size=(wh, ww), mode="bilinear", align_corners=False)

    # 3-level pyramid: 1/4, 1/2, 1x of work resolution
    scales = [0.25, 0.5, 1.0]
    flow = torch.zeros((b_sz, 2, max(4, int(round(wh * scales[0]))), max(4, int(round(ww * scales[0])))),
                       dtype=torch.float32, device=frame_a.device)

    pad = window // 2
    avg_k = (window, window)

    for s in scales:
        sh = max(4, int(round(wh * s)))
        sw = max(4, int(round(ww * s)))
        ia = F.interpolate(la_work, size=(sh, sw), mode="bilinear", align_corners=False)
        ib = F.interpolate(lb_work, size=(sh, sw), mode="bilinear", align_corners=False)

        if flow.shape[-2:] != (sh, sw):
            scale_x = float(sw) / float(flow.shape[-1])
            scale_y = float(sh) / float(flow.shape[-2])
            flow = F.interpolate(flow, size=(sh, sw), mode="bilinear", align_corners=False)
            flow[:, 0] *= scale_x
            flow[:, 1] *= scale_y

        # Warp ib by current flow estimate
        yy, xx = torch.meshgrid(
            torch.linspace(-1.0, 1.0, sh, device=ia.device),
            torch.linspace(-1.0, 1.0, sw, device=ia.device),
            indexing="ij",
        )
        base_grid = torch.stack([xx, yy], dim=-1).unsqueeze(0).expand(b_sz, -1, -1, -1)
        norm_dx = flow[:, 0] * (2.0 / max(1.0, float(sw - 1)))
        norm_dy = flow[:, 1] * (2.0 / max(1.0, float(sh - 1)))
        sample_grid = base_grid + torch.stack([norm_dx, norm_dy], dim=-1)
        ib_warped = F.grid_sample(ib, sample_grid, mode="bilinear", padding_mode="border", align_corners=True)

        # Spatial & temporal derivatives
        ix = 0.5 * (F.pad(ia, (1, 1, 0, 0), mode="replicate")[..., :, 2:] - F.pad(ia, (1, 1, 0, 0), mode="replicate")[..., :, :-2])
        iy = 0.5 * (F.pad(ia, (0, 0, 1, 1), mode="replicate")[..., 2:, :] - F.pad(ia, (0, 0, 1, 1), mode="replicate")[..., :-2, :])
        it = ib_warped - ia

        ixx = F.avg_pool2d(ix * ix, kernel_size=avg_k, stride=1, padding=pad) + 1e-3
        iyy = F.avg_pool2d(iy * iy, kernel_size=avg_k, stride=1, padding=pad) + 1e-3
        ixy = F.avg_pool2d(ix * iy, kernel_size=avg_k, stride=1, padding=pad)
        ixt = F.avg_pool2d(ix * it, kernel_size=avg_k, stride=1, padding=pad)
        iyt = F.avg_pool2d(iy * it, kernel_size=avg_k, stride=1, padding=pad)

        det = (ixx * iyy - ixy * ixy).clamp_min(1e-6)
        du = -(iyy * ixt - ixy * iyt) / det
        dv = -(-ixy * ixt + ixx * iyt) / det
        max_step = max(1.5, 0.08 * max(sh, sw))
        flow[:, 0:1] = (flow[:, 0:1] + du.clamp(-max_step, max_step)).clamp(-0.15 * sw, 0.15 * sw)
        flow[:, 1:2] = (flow[:, 1:2] + dv.clamp(-max_step, max_step)).clamp(-0.15 * sh, 0.15 * sh)

    # Upsample flow back to original resolution
    if (wh, ww) != (h_orig, w_orig):
        sx = float(w_orig) / float(ww)
        sy = float(h_orig) / float(wh)
        flow = F.interpolate(flow, size=(h_orig, w_orig), mode="bilinear", align_corners=False)
        flow[:, 0] *= sx
        flow[:, 1] *= sy

    return flow.permute(0, 2, 3, 1)  # [B, H, W, 2]


def _warp_frames_with_flow(frames: torch.Tensor, flow: torch.Tensor) -> torch.Tensor:
    """Warp [B, H, W, C] frames using pixel-space displacement flow [B, H, W, 2]."""
    b_sz, h, w, _ = frames.shape
    yy, xx = torch.meshgrid(
        torch.linspace(-1.0, 1.0, h, device=frames.device, dtype=torch.float32),
        torch.linspace(-1.0, 1.0, w, device=frames.device, dtype=torch.float32),
        indexing="ij",
    )
    base_grid = torch.stack([xx, yy], dim=-1).unsqueeze(0).expand(b_sz, -1, -1, -1)
    norm_dx = flow[..., 0].float() * (2.0 / max(1.0, float(w - 1)))
    norm_dy = flow[..., 1].float() * (2.0 / max(1.0, float(h - 1)))
    grid = base_grid + torch.stack([norm_dx, norm_dy], dim=-1)
    warped = F.grid_sample(
        frames.float().permute(0, 3, 1, 2),
        grid,
        mode="bilinear",
        padding_mode="border",
        align_corners=True,
    )
    return warped.permute(0, 2, 3, 1).to(frames.dtype)


def blend_overlap_frames(
    prev_overlap: torch.Tensor,
    curr_overlap: torch.Tensor,
    blend_mode: str = "cosine_alpha",
) -> torch.Tensor:
    """Blend N overlapping frames [N, H, W, C] via cosine S-curve or bidirectional optical-flow warp."""
    n = int(prev_overlap.shape[0])
    if n <= 0:
        return prev_overlap
    t = torch.linspace(0.0, 1.0, n + 2, dtype=torch.float32, device=prev_overlap.device)[1:-1]
    alpha = (0.5 - 0.5 * torch.cos(math.pi * t)).view(n, 1, 1, 1)

    if blend_mode == "hard_cut":
        mid = n // 2
        return torch.cat([prev_overlap[:mid], curr_overlap[mid:]], dim=0)

    linear_cosine = (prev_overlap.float() * (1.0 - alpha) + curr_overlap.float() * alpha).clamp(0.0, 1.0)
    if blend_mode != "optical_flow_warp":
        return linear_cosine.to(prev_overlap.dtype)

    # Bidirectional flow alignment between synchronous overlap frames prev_overlap[i] and curr_overlap[i]
    flow_p2c = _estimate_dense_flow_lk(prev_overlap[..., :3].float(), curr_overlap[..., :3].float())
    flow_c2p = _estimate_dense_flow_lk(curr_overlap[..., :3].float(), prev_overlap[..., :3].float())

    # Warp prev by alpha * flow_p2c and curr by (1 - alpha) * flow_c2p toward intermediate trajectory
    prev_warped = _warp_frames_with_flow(prev_overlap.float(), flow_p2c * alpha)
    curr_warped = _warp_frames_with_flow(curr_overlap.float(), flow_c2p * (1.0 - alpha))
    flow_blended = (prev_warped * (1.0 - alpha) + curr_warped * alpha).clamp(0.0, 1.0)

    # Confidence gate: fall back smoothly toward cosine alpha where photometric residual is high (occlusions)
    photometric_err = (prev_warped[..., :3] - curr_warped[..., :3]).abs().mean(dim=-1, keepdim=True)
    conf = (1.0 - (photometric_err / 0.18)).clamp(0.0, 1.0)
    out = flow_blended * conf + linear_cosine * (1.0 - conf)
    return out.clamp(0.0, 1.0).to(prev_overlap.dtype)


# ---------------------------------------------------------------------------
# 3. Audio Sample-Rate Harmonizer & Zero-Click Cosine Crossfade
# ---------------------------------------------------------------------------

def harmonize_audio_format(
    audio: Optional[Dict[str, Any]],
    target_sr: int,
    target_channels: int,
    target_device: torch.device,
    target_dtype: torch.dtype,
) -> Optional[Dict[str, Any]]:
    """Resample and channel-match an AUDIO dict to (target_sr, target_channels) cleanly in PyTorch."""
    std = _standardize_audio_dict(audio, default_sr=target_sr)
    if std is None or "waveform" not in std:
        return None
    w = std["waveform"].to(device=target_device, dtype=torch.float32)  # [1, C, N]
    sr = int(std.get("sample_rate", target_sr))

    def _convert_tensor(wav: torch.Tensor) -> torch.Tensor:
        if wav.shape[1] != target_channels:
            if wav.shape[1] == 1 and target_channels == 2:
                wav = wav.repeat(1, 2, 1)
            elif wav.shape[1] > 1 and target_channels == 1:
                wav = wav.mean(dim=1, keepdim=True)
            elif wav.shape[1] > target_channels:
                wav = wav[:, :target_channels, :]
            else:
                reps = (target_channels + wav.shape[1] - 1) // wav.shape[1]
                wav = wav.repeat(1, reps, 1)[:, :target_channels, :]
        if sr != target_sr and wav.shape[-1] > 0:
            new_len = max(1, int(round(wav.shape[-1] * float(target_sr) / float(sr))))
            wav = F.interpolate(wav, size=new_len, mode="linear", align_corners=False)
        return wav.to(dtype=target_dtype)

    out = dict(std)
    out["waveform"] = _convert_tensor(w)
    out["sample_rate"] = int(target_sr)
    if "minimax_audio_context" in out and isinstance(out["minimax_audio_context"], torch.Tensor):
        ctx = out["minimax_audio_context"].to(device=target_device, dtype=torch.float32)
        if ctx.ndim == 2:
            ctx = ctx.unsqueeze(0)
        out["minimax_audio_context"] = _convert_tensor(ctx)
    return out


def zero_click_audio_stitch(
    prev_audio: Optional[Dict[str, Any]],
    curr_audio: Optional[Dict[str, Any]],
    prev_kept_frames: int,
    curr_total_frames: int,
    curr_trim_frames: int,
    crossfade_ms: float = 20.0,
    fps: float = 24.0,
) -> Optional[Dict[str, Any]]:
    """Sample-accurate, DC-declicked complementary cosine audio stitch across segments."""
    curr_std = _standardize_audio_dict(curr_audio)
    prev_std = _standardize_audio_dict(prev_audio)

    if curr_std is None and prev_std is None:
        return None
    if curr_std is None:
        if prev_std is None:
            return None
        sr = int(prev_std.get("sample_rate", 32000))
        want_p = int(round(prev_kept_frames / float(fps) * sr)) if prev_kept_frames > 0 else prev_std["waveform"].shape[-1]
        return {"waveform": fit_audio_length(prev_std["waveform"], want_p), "sample_rate": sr}

    curr_trim_frames = max(0, int(curr_trim_frames))
    curr_kept_frames = max(0, int(curr_total_frames) - curr_trim_frames)

    if prev_std is None or "waveform" not in prev_std or prev_std["waveform"].shape[-1] == 0:
        sr = int(curr_std.get("sample_rate", 32000))
        head_s = int(round(curr_trim_frames / float(fps) * sr))
        want_s = int(round(curr_kept_frames / float(fps) * sr))
        body = fit_audio_length(curr_std["waveform"][..., head_s:head_s + want_s], want_s)
        return {"waveform": body, "sample_rate": sr}

    # Harmonize curr_audio to prev_audio's sample_rate and channels so multi-engine workflows never crash
    target_sr = int(prev_std.get("sample_rate", 32000))
    prev_w_raw = prev_std["waveform"]
    curr_harm = harmonize_audio_format(
        curr_std,
        target_sr=target_sr,
        target_channels=int(prev_w_raw.shape[1]),
        target_device=prev_w_raw.device,
        target_dtype=prev_w_raw.dtype,
    )
    assert curr_harm is not None

    prev_want_s = int(round(prev_kept_frames / float(fps) * target_sr)) if prev_kept_frames > 0 else prev_w_raw.shape[-1]
    prev_w = fit_audio_length(prev_w_raw, prev_want_s)

    head_s = int(round(curr_trim_frames / float(fps) * target_sr))
    want_s = int(round(curr_kept_frames / float(fps) * target_sr))
    curr_w = curr_harm["waveform"]
    curr_body = fit_audio_length(curr_w[..., head_s:head_s + want_s], want_s)

    if want_s == 0:
        return {"waveform": prev_w, "sample_rate": target_sr}

    context = curr_w[..., :head_s] if 0 < head_s <= curr_w.shape[-1] else curr_harm.get("minimax_audio_context")
    fade_samples = max(0, int(round((float(crossfade_ms) / 1000.0) * target_sr)))
    c = min(fade_samples, context.shape[-1], prev_w.shape[-1]) if (context is not None and context.shape[-1] > 0) else 0

    if c > 0:
        prev_tail = prev_w[..., -c:].float()
        curr_ov = context[..., -c:].to(device=prev_w.device, dtype=torch.float32)

        # Remove any tiny DC bias mismatch across the overlap window to guarantee zero-click splice
        dc_diff = (prev_tail.mean(dim=-1, keepdim=True) - curr_ov.mean(dim=-1, keepdim=True)).clamp(-0.05, 0.05)
        t = torch.linspace(0.0, math.pi / 2.0, c, dtype=torch.float32, device=prev_w.device)
        fade_in = torch.sin(t).square().view(1, 1, c)
        fade_out = 1.0 - fade_in
        blended = (prev_tail * fade_out + (curr_ov + dc_diff * fade_out) * fade_in).to(prev_w.dtype)
        stitched = torch.cat([prev_w[..., :-c], blended, curr_body], dim=-1)
    else:
        # Contextless join: symmetric micro-declick ramp around seam without shortening total duration
        declick_n = min(max(0, fade_samples), prev_w.shape[-1], curr_body.shape[-1], max(16, int(0.005 * target_sr)))
        if declick_n > 1:
            ramp = (0.5 - 0.5 * torch.cos(torch.linspace(0.0, math.pi, declick_n, device=prev_w.device, dtype=torch.float32))).view(1, 1, declick_n)
            pw = prev_w.clone()
            cb = curr_body.clone()
            # Smoothly bridge boundary sample midpoint so waveform is C0 continuous at seam
            mid_val = 0.5 * (pw[..., -1:].float() + cb[..., :1].float())
            pw[..., -declick_n:] = (pw[..., -declick_n:].float() * (1.0 - ramp) + mid_val * ramp).to(prev_w.dtype)
            cb[..., :declick_n] = (mid_val * (1.0 - ramp) + cb[..., :declick_n].float() * ramp).to(curr_body.dtype)
            stitched = torch.cat([pw, cb], dim=-1)
        else:
            stitched = torch.cat([prev_w, curr_body], dim=-1)

    return {"waveform": stitched, "sample_rate": target_sr}


# ---------------------------------------------------------------------------
# 4. Unified Color-Match + Smart-Seam + Micro-Crossfade Pipeline
# ---------------------------------------------------------------------------

def color_match_and_stitch_segments(
    prev_images: Optional[torch.Tensor],
    curr_images: Optional[torch.Tensor],
    overlap_frames: int = 39,
    smart_seam: bool = True,
    search_radius: int = 12,
    max_phase_shift: int = 0,
    color_match_mode: str = "lab_mean_std",
    color_match_strength: float = 0.85,
    anchor_image: Optional[torch.Tensor] = None,
    anchor_pull_strength: float = 0.25,
    grade_decay_frames: int = 0,
    blend_mode: str = "cosine_alpha",
    blend_frames: int = 4,
    prev_audio: Optional[Dict[str, Any]] = None,
    curr_audio: Optional[Dict[str, Any]] = None,
    audio_crossfade_ms: float = 20.0,
    fps: float = 24.0,
) -> Tuple[torch.Tensor, Optional[Dict[str, Any]], Dict[str, Any]]:
    """End-to-end seamless video & audio stitcher with smart seam selection, CIE LAB color match, and micro-blend."""
    prev_std = _standardize_image_tensor(prev_images)
    curr_std = _standardize_image_tensor(curr_images)

    if curr_std is None or curr_std.shape[0] == 0:
        out_img = prev_std if prev_std is not None else torch.empty((0, 352, 608, 3), dtype=torch.float32)
        return out_img, _standardize_audio_dict(prev_audio), {"mode": "empty_curr", "total_frames": int(out_img.shape[0])}

    ov = max(0, int(overlap_frames))
    if prev_std is None or prev_std.shape[0] == 0:
        out_img = curr_std[ov:] if ov < curr_std.shape[0] else curr_std[:0]
        out_aud = zero_click_audio_stitch(
            prev_audio=None,
            curr_audio=curr_audio,
            prev_kept_frames=0,
            curr_total_frames=int(curr_std.shape[0]),
            curr_trim_frames=ov,
            crossfade_ms=audio_crossfade_ms,
            fps=fps,
        )
        return out_img, out_aud, {
            "mode": "initial_clip",
            "total_frames": int(out_img.shape[0]),
            "overlap_frames": ov,
        }

    curr_std = curr_std.to(device=prev_std.device, dtype=prev_std.dtype)
    fp = int(prev_std.shape[0])
    fc = int(curr_std.shape[0])

    if smart_seam and ov > 1:
        seam_plan = find_optimal_seam_cut(
            prev_images=prev_std,
            curr_images=curr_std,
            overlap_frames=ov,
            search_radius=search_radius,
            max_phase_shift=max_phase_shift,
        )
        prev_keep = seam_plan["prev_keep_frames"]
        curr_trim = seam_plan["curr_trim_frames"]
    else:
        w = min(ov, fp, fc)
        prev_keep = fp
        curr_trim = w
        seam_plan = {
            "overlap_window": w,
            "best_cut_k": w,
            "phase_shift": 0,
            "prev_keep_frames": prev_keep,
            "curr_trim_frames": curr_trim,
            "seam_cost": 0.0,
            "nominal_seam_cost": 0.0,
            "improvement_pct": 0.0,
        }

    # Color & luminance match curr_std against synchronous overlap region of prev_std
    analysis_n = min(max(1, curr_trim), prev_keep, 12)
    if curr_trim > 0 and analysis_n > 0:
        ref_ctx = prev_std[prev_keep - analysis_n:prev_keep]
        src_ctx = curr_std[curr_trim - analysis_n:curr_trim]
    else:
        ref_ctx = prev_std[max(0, prev_keep - 6):prev_keep]
        src_ctx = curr_std[:min(6, fc)]

    curr_graded, color_diag = apply_lab_color_match(
        curr_images=curr_std,
        ref_prev_context=ref_ctx,
        curr_overlap_context=src_ctx,
        anchor_image=anchor_image,
        mode=color_match_mode,
        strength=color_match_strength,
        anchor_pull_strength=anchor_pull_strength,
        decay_frames=grade_decay_frames,
    )

    prev_active = prev_std[:prev_keep]
    curr_body = curr_graded[curr_trim:]

    n_blend = 0 if blend_mode == "hard_cut" else min(max(0, int(blend_frames)), curr_trim, int(prev_active.shape[0]))
    if n_blend > 0:
        prev_prefix = prev_active[:-n_blend]
        prev_tail = prev_active[-n_blend:]
        curr_ov = curr_graded[curr_trim - n_blend:curr_trim]
        blended = blend_overlap_frames(prev_tail, curr_ov, blend_mode=blend_mode)
        stitched_img = torch.cat([prev_prefix, blended, curr_body], dim=0)
    else:
        stitched_img = torch.cat([prev_active, curr_body], dim=0)

    stitched_aud = zero_click_audio_stitch(
        prev_audio=prev_audio,
        curr_audio=curr_audio,
        prev_kept_frames=prev_keep,
        curr_total_frames=fc,
        curr_trim_frames=curr_trim,
        crossfade_ms=audio_crossfade_ms,
        fps=fps,
    )

    diagnostics = {
        "mode": "stitched",
        "total_frames": int(stitched_img.shape[0]),
        "duration_sec": round(int(stitched_img.shape[0]) / float(fps), 3),
        "blend_mode": blend_mode,
        "blend_frames_used": int(n_blend),
        "seam_plan": seam_plan,
        "color_match": color_diag,
    }
    return stitched_img, stitched_aud, diagnostics
