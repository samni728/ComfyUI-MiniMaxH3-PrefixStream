"""Segment Drift & Seam QA Probe for MiniMax H3 PrefixStream Chains.

Detects cumulative visual degradation BEFORE A12-style 13-segment collapse:
- seam_jump_score: Normalized boundary discontinuity vs natural intra-clip motion.
- sharpness_ratio_vs_anchor: High-frequency Laplacian detail ratio vs Shot-1 anchor (catches blur AND anti-burn blotching).
- color_drift_score: Perceptual CIE L*a*b* Delta-E76 / contrast drift vs anchor or previous clip.
- recommend_anchor_reset: Boolean circuit-breaker triggered when chain_segment_index >= max_safe_chain_length (default 3)
  or when seam/sharpness/color/black-frame thresholds are breached.
"""

from typing import Any, Dict, List, Optional
import math
import torch
import torch.nn.functional as F

try:
    from ..pipeline.seam_protector import _rgb_luminance, _standardize_audio_dict, _standardize_image_tensor
    from ..pipeline.smart_seam_color_stitch import _downsample_frames, _spatial_edge_magnitude, rgb_to_lab
    from .texture_diagnostics import texture_band_rms
except (ImportError, ValueError):
    from pipeline.seam_protector import _rgb_luminance, _standardize_audio_dict, _standardize_image_tensor
    from pipeline.smart_seam_color_stitch import _downsample_frames, _spatial_edge_magnitude, rgb_to_lab
    from engine.texture_diagnostics import texture_band_rms


def _laplacian_sharpness(frames: torch.Tensor) -> float:
    """Compute mean high-frequency 2D Laplacian RMS energy on interior luminance [F, H, W, C]."""
    if frames is None or frames.shape[0] == 0:
        return 0.0
    luma = _rgb_luminance(frames.float())  # [F, H, W]
    # Filter out solid black/flash frames (std < 1e-4) if valid content frames exist
    per_frame_std = luma.reshape(luma.shape[0], -1).std(dim=-1, unbiased=False)
    valid_mask = per_frame_std > 1e-4
    if bool(valid_mask.any()):
        luma = luma[valid_mask]
    luma_4d = luma.unsqueeze(1)
    if luma_4d.shape[-2] < 6 or luma_4d.shape[-1] < 6:
        return float(luma_4d.std().item())
    kernel = torch.tensor(
        [[[[0.0, 1.0, 0.0], [1.0, -4.0, 1.0], [0.0, 1.0, 0.0]]]],
        dtype=torch.float32,
        device=luma_4d.device,
    )
    lap = F.conv2d(luma_4d, kernel)  # valid interior [F, 1, H-2, W-2], zero border-padding bias
    if lap.shape[-2] > 4 and lap.shape[-1] > 4:
        lap = lap[:, :, 1:-1, 1:-1]
    return float(lap.square().mean().sqrt().item())


def _sample_representative_frames(frames: torch.Tensor, max_samples: int = 8) -> torch.Tensor:
    n = int(frames.shape[0])
    if n <= max_samples:
        return frames
    idx = torch.linspace(0, n - 1, max_samples, device=frames.device).round().long()
    return frames.index_select(0, idx)


def evaluate_prefix_drift_qa(
    curr_images: torch.Tensor,
    prev_images: Optional[torch.Tensor] = None,
    anchor_image: Optional[torch.Tensor] = None,
    curr_audio: Optional[Dict[str, Any]] = None,
    chain_segment_index: int = 1,
    overlap_frames: int = 0,
    max_safe_chain_length: int = 3,
    seam_jump_threshold: float = 0.12,
    sharpness_min_ratio: float = 0.72,
    sharpness_max_burn_ratio: float = 1.65,
    color_drift_threshold: float = 0.10,
) -> Dict[str, Any]:
    """Compute structured seam & cumulative drift QA metrics for a PrefixStream segment."""
    curr_std = _standardize_image_tensor(curr_images)
    prev_std = _standardize_image_tensor(prev_images)
    anc_std = _standardize_image_tensor(anchor_image)
    aud_std = _standardize_audio_dict(curr_audio)

    if curr_std is None or curr_std.shape[0] == 0:
        return {
            "chain_segment_index": int(chain_segment_index),
            "max_safe_chain_length": int(max_safe_chain_length),
            "seam_jump_score": 1.0,
            "sharpness_ratio_vs_anchor": 0.0,
            "color_drift_score": 1.0,
            "black_frame_count": 0,
            "frozen_frame_ratio": 0.0,
            "recommend_anchor_reset": True,
            "status": "FAIL",
            "reasons": ["empty_current_clip"],
        }

    device = curr_std.device
    fc = int(curr_std.shape[0])
    ov = max(0, int(overlap_frames))
    active_curr = curr_std[ov:] if 0 < ov < fc else curr_std

    # 1. Black frame & frozen frame checks (Gate B automated pre-screen)
    curr_small = _downsample_frames(active_curr, max_side=128)
    frame_lumas = _rgb_luminance(curr_small).mean(dim=(-2, -1))
    black_frame_count = int((frame_lumas < 0.015).sum().item())

    if curr_small.shape[0] >= 2:
        diffs = (curr_small[1:] - curr_small[:-1]).abs().mean(dim=(1, 2, 3))
        frozen_frame_ratio = float((diffs < 1e-4).float().mean().item())
        natural_step_diff = float(diffs.median().item())
    else:
        frozen_frame_ratio = 0.0
        natural_step_diff = 0.01

    # 2. Seam Jump Score
    seam_jump_score = 0.0
    if prev_std is not None and prev_std.shape[0] > 0:
        prev_small = _downsample_frames(prev_std[-min(8, prev_std.shape[0]):].to(device), max_side=128)
        seam_curr_idx = min(ov, fc - 1)
        curr_seam_frame = _downsample_frames(curr_std[seam_curr_idx:seam_curr_idx + 1], max_side=128)[0]
        prev_last_frame = prev_small[-1]

        raw_rgb_jump = float((curr_seam_frame - prev_last_frame).abs().mean().item())
        edge_p = _spatial_edge_magnitude(prev_last_frame.unsqueeze(0))[0]
        edge_c = _spatial_edge_magnitude(curr_seam_frame.unsqueeze(0))[0]
        edge_jump = float((edge_c - edge_p).abs().mean().item())

        # Subtract expected 1-frame motion so smooth camera pans aren't penalized as jump cuts
        excess_rgb_jump = max(0.0, raw_rgb_jump - 1.25 * natural_step_diff)
        seam_jump_score = float(min(1.0, 0.70 * excess_rgb_jump + 0.30 * max(0.0, edge_jump - natural_step_diff)))

    # 3. Reference selection for Sharpness & Color Drift (Anchor > Prev > Self-Head)
    if anc_std is not None and anc_std.shape[0] > 0:
        ref_frames = _sample_representative_frames(anc_std.to(device), max_samples=4)
        ref_source = "anchor_image"
    elif prev_std is not None and prev_std.shape[0] > 0:
        ref_frames = _sample_representative_frames(prev_std[:min(12, prev_std.shape[0])].to(device), max_samples=4)
        ref_source = "prev_images"
    else:
        ref_frames = curr_std[:max(1, min(4, fc))]
        ref_source = "self_head"

    curr_rep = _sample_representative_frames(active_curr, max_samples=8)

    ref_sharp = _laplacian_sharpness(ref_frames)
    curr_sharp = _laplacian_sharpness(curr_rep)
    if ref_sharp > 1e-6:
        sharpness_ratio = curr_sharp / ref_sharp
    else:
        sharpness_ratio = 1.0 if curr_sharp <= 1e-6 else 1.2

    # Optional mid-frequency texture drift probe (2-8px vs 8-32px blotch detector)
    mid_band_ratio = 1.0
    if min(curr_rep.shape[1], curr_rep.shape[2]) >= 16 and min(ref_frames.shape[1], ref_frames.shape[2]) >= 16:
        try:
            ref_tex = texture_band_rms(_downsample_frames(ref_frames[:1], max_side=128)[0])
            cur_tex = texture_band_rms(_downsample_frames(curr_rep[-1:], max_side=128)[0])
            ref_mid = ref_tex["band_rms"]["luma"]["8_32px"]
            cur_mid = cur_tex["band_rms"]["luma"]["8_32px"]
            if ref_mid > 1e-6:
                mid_band_ratio = float(cur_mid / ref_mid)
        except Exception:
            mid_band_ratio = 1.0

    # 4. Perceptual CIE LAB Color Drift Score
    ref_lab = rgb_to_lab(_downsample_frames(ref_frames, max_side=96)).reshape(-1, 3)
    cur_lab = rgb_to_lab(_downsample_frames(curr_rep, max_side=96)).reshape(-1, 3)

    mu_ref = ref_lab.mean(dim=0)
    mu_cur = cur_lab.mean(dim=0)
    std_ref = ref_lab.std(dim=0, unbiased=False)
    std_cur = cur_lab.std(dim=0, unbiased=False)

    delta_e = float((mu_cur - mu_ref).norm(p=2).item())  # CIE76 Delta-E (~2.3 JND, >10 visible drift)
    contrast_drift = abs(float(std_cur[0].item()) - float(std_ref[0].item())) / 100.0
    color_drift_score = float(min(1.0, (delta_e / 100.0) + 0.35 * contrast_drift))

    # 5. Audio clipping / silence sanity check if audio provided
    audio_peak = 0.0
    audio_clipping_ratio = 0.0
    if aud_std is not None and "waveform" in aud_std and aud_std["waveform"].numel() > 0:
        w = aud_std["waveform"].float()
        audio_peak = float(w.abs().max().item())
        audio_clipping_ratio = float((w.abs() >= 0.999).float().mean().item())

    # 6. Circuit-Breaker / Anchor Reset Decision
    reasons: List[str] = []
    seg_idx = max(1, int(chain_segment_index))
    safe_len = max(1, int(max_safe_chain_length))

    if seg_idx >= safe_len:
        reasons.append(f"chain_segment_index({seg_idx}) >= max_safe_chain_length({safe_len}): reset to fresh keyframe anchor next")
    if seam_jump_score > seam_jump_threshold:
        reasons.append(f"seam_jump_score({seam_jump_score:.4f}) > threshold({seam_jump_threshold:.4f})")
    if sharpness_ratio < sharpness_min_ratio:
        reasons.append(f"sharpness_ratio_vs_anchor({sharpness_ratio:.3f}) < min({sharpness_min_ratio:.3f}): detail blur detected")
    if sharpness_ratio > sharpness_max_burn_ratio or mid_band_ratio > 1.75:
        reasons.append(
            f"texture_burn_risk(sharpness_ratio={sharpness_ratio:.3f}, mid_band_ratio={mid_band_ratio:.3f}): "
            "cumulative anti-burn/blotch threshold exceeded"
        )
    if color_drift_score > color_drift_threshold:
        reasons.append(f"color_drift_score({color_drift_score:.4f}, dE={delta_e:.2f}) > threshold({color_drift_threshold:.4f})")
    if black_frame_count > 0:
        reasons.append(f"black_frames_detected({black_frame_count})")
    if frozen_frame_ratio > 0.50 and fc >= 8:
        reasons.append(f"frozen_frame_ratio({frozen_frame_ratio:.2f}) > 0.50")
    if audio_clipping_ratio > 0.02:
        reasons.append(f"audio_clipping_ratio({audio_clipping_ratio:.4f}) > 0.02")

    recommend_reset = len(reasons) > 0
    hard_defect = any(
        k in " ".join(reasons)
        for k in ("seam_jump_score", "sharpness_ratio_vs_anchor", "texture_burn_risk", "color_drift_score", "black_frames", "frozen_frame")
    )
    status = "FAIL_DRIFT" if hard_defect else ("RESET_ANCHOR_NEXT" if recommend_reset else "PASS")

    return {
        "status": status,
        "reference_source": ref_source,
        "chain_segment_index": seg_idx,
        "max_safe_chain_length": safe_len,
        "analyzed_frames": int(active_curr.shape[0]),
        "seam_jump_score": round(seam_jump_score, 5),
        "sharpness_ratio_vs_anchor": round(float(sharpness_ratio), 4),
        "mid_band_texture_ratio": round(float(mid_band_ratio), 4),
        "color_drift_score": round(color_drift_score, 5),
        "cie76_delta_e": round(delta_e, 3),
        "black_frame_count": black_frame_count,
        "frozen_frame_ratio": round(frozen_frame_ratio, 4),
        "audio_peak": round(audio_peak, 4),
        "audio_clipping_ratio": round(audio_clipping_ratio, 5),
        "recommend_anchor_reset": bool(recommend_reset),
        "reasons": reasons,
    }
