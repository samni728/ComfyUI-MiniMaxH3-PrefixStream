"""Dual-Engine MV Montage Stitcher: LongCat Vocal A-Roll ('唱整段') + H3 PrefixStream B-Roll ('拍场面').

Solves Solo MV A12/A30 synchronization & visual consistency bottlenecks:
1. Single Master Song Lock (`48kHz` / `44.1kHz` / `32kHz`): The output `AUDIO` tensor is sliced directly from
   `master_audio` to exact sample-locked video length (`round(total_frames / fps * sample_rate)`).
   Per-clip VAE audio drift is 0.0 ms across entire 3-minute songs!
2. Frame-Rate & Aspect Harmonization: Seamlessly converts LongCat 30fps (or arbitrary source FPS) A-Roll clips
   and H3 24fps B-Roll clips onto a unified master `fps` and `(target_height, target_width)` canvas (`cover_crop`, `letterbox`, `stretch`).
3. Cross-Engine CIE LAB Color Harmonization: Matches H3 B-Roll palette/exposure toward LongCat A-Roll (or `anchor_style_image`)
   so cutting between vocal performance and narrative scenery looks like a single camera package.
4. Beat-Accurate Cuts & Symmetric Micro-Crossfades: Supports crisp beat cuts (`transition_frames=0`) or centered
   cosine S-curve dissolves (`transition_frames=2..12`) without shifting segment start/end timestamps by even 1 frame!
"""

from typing import Any, Dict, List, Optional, Tuple
import json
import math
import torch
import torch.nn.functional as F

from .seam_protector import _standardize_audio_dict, _standardize_image_tensor, fit_audio_length
from .smart_seam_color_stitch import apply_lab_color_match, harmonize_audio_format


def resample_clip_fps(images: torch.Tensor, source_fps: float, target_fps: float) -> torch.Tensor:
    """Temporally resample [F, H, W, C] frames from source_fps to target_fps cleanly."""
    std = _standardize_image_tensor(images)
    if std is None or std.shape[0] <= 1:
        return std
    src_f = float(source_fps)
    tgt_f = float(target_fps)
    if src_f <= 0 or tgt_f <= 0 or abs(src_f - tgt_f) < 1e-3:
        return std
    n_in = int(std.shape[0])
    n_out = max(1, int(round(n_in * tgt_f / src_f)))
    if n_out == n_in:
        return std
    pos = torch.linspace(0.0, float(n_in - 1), n_out, device=std.device, dtype=torch.float32)
    idx0 = pos.floor().long().clamp(0, n_in - 1)
    idx1 = (idx0 + 1).clamp(0, n_in - 1)
    w = (pos - idx0.float()).view(n_out, 1, 1, 1)
    f0 = std.index_select(0, idx0).float()
    f1 = std.index_select(0, idx1).float()
    return (f0 * (1.0 - w) + f1 * w).to(std.dtype)


def fit_spatial_resolution(
    images: torch.Tensor,
    target_height: int,
    target_width: int,
    fit_mode: str = "cover_crop",
) -> torch.Tensor:
    """Resize/crop/pad [F, H, W, C] frames to exact (target_height, target_width, 3)."""
    std = _standardize_image_tensor(images)
    if std is None or std.shape[0] == 0:
        return torch.empty((0, target_height, target_width, 3), dtype=torch.float32)
    rgb = std[..., :3]
    f_cnt, h, w, _ = rgb.shape
    if h == target_height and w == target_width:
        return rgb

    chw = rgb.float().permute(0, 3, 1, 2)  # [F, 3, H, W]
    if fit_mode == "stretch":
        resized = F.interpolate(chw, size=(target_height, target_width), mode="bilinear", align_corners=False)
        return resized.permute(0, 2, 3, 1).clamp(0.0, 1.0).to(std.dtype)

    if fit_mode == "letterbox":
        scale = min(float(target_height) / float(max(1, h)), float(target_width) / float(max(1, w)))
        nh = max(1, int(round(h * scale)))
        nw = max(1, int(round(w * scale)))
        resized = F.interpolate(chw, size=(nh, nw), mode="bilinear", align_corners=False)
        pad_top = (target_height - nh) // 2
        pad_bottom = target_height - nh - pad_top
        pad_left = (target_width - nw) // 2
        pad_right = target_width - nw - pad_left
        padded = F.pad(resized, (pad_left, pad_right, pad_top, pad_bottom), mode="constant", value=0.0)
        return padded.permute(0, 2, 3, 1).clamp(0.0, 1.0).to(std.dtype)

    # Default: cover_crop (aspect-preserving scale + center crop, zero black bars!)
    scale = max(float(target_height) / float(max(1, h)), float(target_width) / float(max(1, w)))
    nh = max(target_height, int(math.ceil(h * scale)))
    nw = max(target_width, int(math.ceil(w * scale)))
    resized = F.interpolate(chw, size=(nh, nw), mode="bilinear", align_corners=False)
    y0 = max(0, (nh - target_height) // 2)
    x0 = max(0, (nw - target_width) // 2)
    cropped = resized[:, :, y0:y0 + target_height, x0:x0 + target_width]
    return cropped.permute(0, 2, 3, 1).clamp(0.0, 1.0).to(std.dtype)


def _extract_window_with_hold_or_loop(
    clip: torch.Tensor,
    start_frame: int,
    need_frames: int,
    pad_mode: str = "pingpong_or_hold",
) -> torch.Tensor:
    """Extract exact `need_frames` from `clip` starting at `start_frame` without crashing if clip is shorter."""
    if need_frames <= 0:
        return clip[:0]
    n = int(clip.shape[0])
    if n == 0:
        return clip[:0]
    sf = max(0, min(int(start_frame), n - 1))
    avail = clip[sf:]
    if avail.shape[0] >= need_frames:
        return avail[:need_frames]
    if pad_mode == "hold_last" or n == 1:
        rem = need_frames - int(avail.shape[0])
        return torch.cat([avail, avail[-1:].repeat(rem, 1, 1, 1)], dim=0)

    # Smooth ping-pong extension for slight B-Roll / A-Roll shortfall so motion never hard-freezes
    rev = clip.flip(0)
    cycle = torch.cat([avail, rev[1:] if n > 1 else rev], dim=0)
    out_parts = [avail]
    rem = need_frames - int(avail.shape[0])
    bounce = rev[1:] if n > 1 else rev
    forward = False
    while rem > 0:
        src = clip[1:] if (forward and n > 1) else (bounce if not forward else clip)
        if src.shape[0] == 0:
            src = clip
        take = min(rem, int(src.shape[0]))
        out_parts.append(src[:take])
        rem -= take
        forward = not forward
    return torch.cat(out_parts, dim=0)[:need_frames]


def _parse_timeline_schedule(
    timeline_spec_json: str,
    available_slots: Dict[str, torch.Tensor],
    master_duration_sec: float,
    fps: float,
    default_segment_sec: float = 4.0,
) -> List[Dict[str, Any]]:
    """Parse explicit JSON EDL or auto-build an alternating A-Roll / B-Roll montage schedule."""
    raw = (timeline_spec_json or "").strip()
    if raw and raw.lower() not in ("auto", "auto_alternating", "[]", "{}"):
        parsed = json.loads(raw)
        entries = parsed.get("segments", parsed.get("timeline", [])) if isinstance(parsed, dict) else parsed
        if not isinstance(entries, list):
            raise ValueError("timeline_spec_json must be a JSON list of segment dicts or an object with 'segments'")
        cleaned: List[Dict[str, Any]] = []
        cursor_sec = 0.0
        for idx, item in enumerate(entries):
            if not isinstance(item, dict):
                continue
            slot = str(item.get("clip_slot") or item.get("slot") or "").strip()
            role = str(item.get("role") or ("broll" if slot.startswith("broll") else "aroll")).lower()
            if not slot:
                # Auto-pick matching role slot
                candidates = [k for k in available_slots if k.startswith(role)]
                slot = candidates[idx % len(candidates)] if candidates else (next(iter(available_slots.keys()), ""))
            start_s = float(item.get("start_sec", cursor_sec))
            if "end_sec" in item:
                end_s = float(item["end_sec"])
            elif "duration_sec" in item:
                end_s = start_s + float(item["duration_sec"])
            else:
                clip_t = available_slots.get(slot)
                dur = (float(clip_t.shape[0]) / float(fps)) if clip_t is not None else default_segment_sec
                end_s = start_s + dur
            if end_s <= start_s:
                continue
            cleaned.append({
                "index": len(cleaned),
                "role": role,
                "clip_slot": slot,
                "start_sec": start_s,
                "end_sec": end_s,
                "source_offset_sec": float(item.get("source_offset_sec", 0.0)),
                "transition": str(item.get("transition", "crossfade")).lower(),
            })
            cursor_sec = end_s
        if cleaned:
            return cleaned

    # Auto alternating montage across provided slots up to master_duration_sec (or sum of clips if shorter)
    arolls = [k for k in ("aroll_1", "aroll_2", "aroll_3", "aroll_4") if k in available_slots]
    brolls = [k for k in ("broll_1", "broll_2", "broll_3", "broll_4") if k in available_slots]
    ordered_slots: List[Tuple[str, str]] = []
    max_pairs = max(len(arolls), len(brolls))
    for i in range(max_pairs):
        if i < len(arolls):
            ordered_slots.append(("aroll", arolls[i]))
        if i < len(brolls):
            ordered_slots.append(("broll", brolls[i]))

    if not ordered_slots:
        return []

    schedule: List[Dict[str, Any]] = []
    cursor = 0.0
    seg_dur = max(0.5, float(default_segment_sec))
    target_total = master_duration_sec if master_duration_sec > 0 else sum(
        float(available_slots[s].shape[0]) / float(fps) for _, s in ordered_slots
    )
    step_idx = 0
    # Cap auto schedule at target_total (or single pass if master_duration_sec <= 0)
    while cursor < target_total - (0.5 / float(fps)) and step_idx < 256:
        role, slot = ordered_slots[step_idx % len(ordered_slots)]
        clip_dur = float(available_slots[slot].shape[0]) / float(fps)
        use_dur = min(clip_dur if clip_dur > 0 else seg_dur, target_total - cursor)
        if use_dur <= 0:
            break
        schedule.append({
            "index": step_idx,
            "role": role,
            "clip_slot": slot,
            "start_sec": round(cursor, 4),
            "end_sec": round(cursor + use_dur, 4),
            "source_offset_sec": 0.0,
            "transition": "crossfade",
        })
        cursor += use_dur
        step_idx += 1
        # If clips aren't meant to repeat and we used all provided slots without explicit master song cap, stop if master_duration_sec was inferred
    return schedule


def stitch_dual_engine_mv_montage(
    master_audio: Dict[str, Any],
    fps: float = 24.0,
    target_width: int = 0,
    target_height: int = 0,
    fit_mode: str = "cover_crop",
    transition_frames: int = 3,
    color_harmonize_broll_to_aroll: bool = True,
    color_harmonize_strength: float = 0.65,
    timeline_spec_json: str = "auto_alternating",
    default_segment_sec: float = 4.0,
    aroll_source_fps: float = 30.0,
    broll_source_fps: float = 24.0,
    target_sample_rate: int = 0,
    master_start_sec: float = 0.0,
    anchor_style_image: Optional[torch.Tensor] = None,
    clip_slots: Optional[Dict[str, Optional[torch.Tensor]]] = None,
) -> Tuple[torch.Tensor, Dict[str, Any], int, Dict[str, Any]]:
    """Assemble LongCat A-Roll + H3 B-Roll segments locked to `master_audio` with 0.0ms audio drift."""
    aud_std = _standardize_audio_dict(master_audio, default_sr=44100)
    if aud_std is None or "waveform" not in aud_std or aud_std["waveform"].shape[-1] == 0:
        raise ValueError("H3LongCatMVMontageStitcher requires a non-empty master_audio input.")

    orig_sr = int(aud_std.get("sample_rate", 44100))
    out_sr = int(target_sample_rate) if int(target_sample_rate) > 0 else orig_sr
    if out_sr != orig_sr:
        aud_std = harmonize_audio_format(
            aud_std,
            target_sr=out_sr,
            target_channels=int(aud_std["waveform"].shape[1]),
            target_device=aud_std["waveform"].device,
            target_dtype=aud_std["waveform"].dtype,
        )
        assert aud_std is not None

    master_wav = aud_std["waveform"]
    audio_offset_samples = max(0, int(round(float(master_start_sec) * out_sr)))
    if audio_offset_samples > 0 and audio_offset_samples < master_wav.shape[-1]:
        master_wav = master_wav[..., audio_offset_samples:]
    master_dur_sec = float(master_wav.shape[-1]) / float(out_sr)

    raw_slots = clip_slots or {}
    prepared_slots: Dict[str, torch.Tensor] = {}
    first_valid: Optional[torch.Tensor] = None

    for slot_name, tensor in raw_slots.items():
        std_t = _standardize_image_tensor(tensor)
        if std_t is None or std_t.shape[0] == 0:
            continue
        src_fps = aroll_source_fps if slot_name.startswith("aroll") else broll_source_fps
        resampled = resample_clip_fps(std_t, source_fps=src_fps, target_fps=fps)
        prepared_slots[slot_name] = resampled
        if first_valid is None:
            first_valid = resampled

    if first_valid is None:
        raise ValueError("At least one non-empty A-Roll or B-Roll IMAGE clip must be connected.")

    device = first_valid.device
    dtype = first_valid.dtype
    out_h = int(target_height) if int(target_height) > 0 else int(first_valid.shape[1])
    out_w = int(target_width) if int(target_width) > 0 else int(first_valid.shape[2])

    for k in list(prepared_slots.keys()):
        prepared_slots[k] = fit_spatial_resolution(
            prepared_slots[k].to(device=device, dtype=dtype),
            target_height=out_h,
            target_width=out_w,
            fit_mode=fit_mode,
        )

    # Reference palette for cross-engine color harmonization (Anchor > first A-Roll > first clip)
    anc_std = _standardize_image_tensor(anchor_style_image)
    grade_ref: Optional[torch.Tensor] = None
    if anc_std is not None and anc_std.shape[0] > 0:
        grade_ref = fit_spatial_resolution(anc_std.to(device=device, dtype=dtype), out_h, out_w, fit_mode=fit_mode)
    else:
        for k in ("aroll_1", "aroll_2", "aroll_3", "aroll_4"):
            if k in prepared_slots:
                grade_ref = prepared_slots[k][:min(8, prepared_slots[k].shape[0])]
                break

    if color_harmonize_broll_to_aroll and grade_ref is not None and color_harmonize_strength > 1e-4:
        for k in list(prepared_slots.keys()):
            if k.startswith("broll") or anc_std is not None:
                graded, _ = apply_lab_color_match(
                    curr_images=prepared_slots[k],
                    ref_prev_context=grade_ref,
                    curr_overlap_context=prepared_slots[k][:min(8, prepared_slots[k].shape[0])],
                    anchor_image=grade_ref,
                    mode="lab_mean_std",
                    strength=color_harmonize_strength,
                    anchor_pull_strength=0.35,
                    decay_frames=0,
                )
                prepared_slots[k] = graded

    schedule = _parse_timeline_schedule(
        timeline_spec_json=timeline_spec_json,
        available_slots=prepared_slots,
        master_duration_sec=master_dur_sec,
        fps=fps,
        default_segment_sec=default_segment_sec,
    )
    if not schedule:
        raise ValueError("Montage schedule produced zero segments.")

    # Render exact frame-grid timeline so cumulative rounding drift is strictly 0 frames!
    trans_f = max(0, int(transition_frames))
    segment_tensors: List[torch.Tensor] = []
    segment_transitions: List[str] = []
    cut_log: List[Dict[str, Any]] = []
    aroll_frames_total = 0
    broll_frames_total = 0

    for idx, seg in enumerate(schedule):
        slot = seg["clip_slot"]
        if slot not in prepared_slots:
            fallback_key = next(iter(prepared_slots.keys()))
            slot = fallback_key
        src_clip = prepared_slots[slot]

        f_start = int(round(seg["start_sec"] * float(fps)))
        f_end = int(round(seg["end_sec"] * float(fps)))
        seg_frames = max(1, f_end - f_start)
        src_offset_f = max(0, int(round(seg.get("source_offset_sec", 0.0) * float(fps))))

        # Request extra tail/head margin for symmetric crossfade if available in source clip
        tail_extra = trans_f if idx < len(schedule) - 1 and seg.get("transition") != "cut" else 0
        extracted = _extract_window_with_hold_or_loop(src_clip, src_offset_f, seg_frames + tail_extra)
        body = extracted[:seg_frames]
        tail_handle = extracted[seg_frames:seg_frames + tail_extra]

        segment_tensors.append(body)
        segment_transitions.append(seg.get("transition", "crossfade"))

        if seg["role"] == "aroll":
            aroll_frames_total += seg_frames
        else:
            broll_frames_total += seg_frames

        cut_log.append({
            "index": idx,
            "role": seg["role"],
            "clip_slot": slot,
            "timeline_frames": [f_start, f_start + seg_frames],
            "frames": seg_frames,
            "duration_sec": round(seg_frames / float(fps), 3),
            "transition": seg.get("transition", "crossfade"),
            "_tail_handle": tail_handle,
        })

    # Assemble timeline with duration-invariant boundary dissolves
    assembled = segment_tensors[0]
    for i in range(1, len(segment_tensors)):
        nxt = segment_tensors[i]
        use_xfade = (trans_f > 0) and (segment_transitions[i - 1] != "cut")
        tail_handle = cut_log[i - 1].pop("_tail_handle", None)
        n_blend = min(trans_f, int(assembled.shape[0]), int(nxt.shape[0])) if use_xfade else 0

        if n_blend > 0 and tail_handle is not None and int(tail_handle.shape[0]) >= n_blend:
            # Blend tail_handle (overlapping in time with nxt[:n_blend]) into nxt[:n_blend] -> 0 frame loss!
            t = torch.linspace(0.0, 1.0, n_blend + 2, device=device, dtype=torch.float32)[1:-1]
            alpha = (0.5 - 0.5 * torch.cos(math.pi * t)).view(n_blend, 1, 1, 1)
            blended_head = (tail_handle[:n_blend].float() * (1.0 - alpha) + nxt[:n_blend].float() * alpha).to(dtype)
            assembled = torch.cat([assembled, blended_head, nxt[n_blend:]], dim=0)
        else:
            assembled = torch.cat([assembled, nxt], dim=0)

    if "_tail_handle" in cut_log[-1]:
        cut_log[-1].pop("_tail_handle", None)

    total_frames = int(assembled.shape[0])
    exact_audio_samples = int(round(total_frames / float(fps) * out_sr))
    locked_waveform = fit_audio_length(master_wav, exact_audio_samples)
    mv_audio = {
        "waveform": locked_waveform,
        "sample_rate": out_sr,
    }

    report = {
        "total_frames": total_frames,
        "fps": float(fps),
        "duration_sec": round(total_frames / float(fps), 4),
        "resolution": f"{out_w}x{out_h}",
        "audio_sample_rate": out_sr,
        "audio_samples": exact_audio_samples,
        "audio_video_drift_ms": 0.0,
        "aroll_frames": aroll_frames_total,
        "broll_frames": broll_frames_total,
        "aroll_ratio": round(aroll_frames_total / float(max(1, total_frames)), 3),
        "color_harmonized": bool(color_harmonize_broll_to_aroll and grade_ref is not None),
        "segments": cut_log,
    }
    return assembled, mv_audio, total_frames, report
