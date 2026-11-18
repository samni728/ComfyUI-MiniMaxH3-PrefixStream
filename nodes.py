"""ComfyUI custom nodes for MiniMax H3 masked AV continuation and chaining.

Exposes:
- MiniMaxPrefixCacheConfig: Select continuation mode and a visible-video context length.
- MiniMaxPrefixCacheApplier: Builds native AV masks or Safe Native fallback conditioning.
- MiniMaxTrimPrefixLatent: Automatically trims leading overlap frames from generated clips (AV unified).
- MiniMaxLongVideoStitcher: Smoothly stitches video and audio latents between clips in unified LATENT space.
- MiniMaxCacheMonitor: Real-time diagnostics for VRAM, memory footprint, and session progress.
"""

import os
import json
import logging
from typing import Dict, Any, Tuple, Optional
import torch

try:
    from safetensors import safe_open
    from safetensors.torch import load_file as st_load, save_file as st_save
except ImportError:
    try:
        import safetensors.torch
        st_load = safetensors.torch.load_file
        st_save = safetensors.torch.save_file
    except ImportError:
        st_load = None
        st_save = None

logger = logging.getLogger("minimax_prefix_stream")

try:
    from .engine.continuation_state import encode_state, decode_state, prepare_identity_references, ANCHOR_KEY
    from .engine.cache_manager import (
        KVCacheConfig,
        PrefixKVCacheManager,
        pixel_frames_to_latent_steps,
        latent_steps_to_pixel_frames,
    )
    from .pipeline.long_video_director import LongVideoSession
    from .pipeline.disk_stream import MiniMaxDiskVideoStreamNode
    from .pipeline.native_masked_av import apply_native_masked_av, latent_boundaries
    from .pipeline.context_noise import encode_noisy_prefix
    from .pipeline.seam_protector import (
        audio_equal_power_crossfade,
        latent_soft_blend,
        stitch_video_latents,
        stitch_audio_latents,
        trim_prefix_frames,
        trim_audio_latents,
        trim_audio_waveform,
        stitch_video_images,
        stitch_audio_waveforms,
        trim_images_and_audio,
        estimate_luminance_gain,
        apply_luminance_gain_fade,
        _standardize_image_tensor,
        _standardize_audio_dict,
    )
    from .engine.clip_bin_manager import (
        save_clip_asset,
        load_clip_asset,
        get_project_dir,
        list_projects,
        load_project_index,
        get_clips_for_selection,
        format_clip_label,
        pil_to_tensor,
        create_placeholder_card,
    )
    from .engine.timeline_session_manager import (
        get_or_create_timeline_session,
        slice_video_and_audio,
        render_timeline_indicator_image,
        get_input_video_files,
        resolve_video_path,
        standardize_audio_dict,
    )
    from .pipeline.smart_seam_color_stitch import (
        find_optimal_seam_cut,
        smart_select_seam_and_trim,
        apply_lab_color_match,
        color_match_and_stitch_segments,
        zero_click_audio_stitch,
    )
    from .engine.drift_qa_probe import evaluate_prefix_drift_qa
    from .engine.prompt_compiler import compile_h3_four_sentence_prompt, lint_h3_contradictions
    from .engine.vram_stage_gate import run_vram_stage_gate
    from .pipeline.mv_montage_stitcher import stitch_dual_engine_mv_montage
except (ImportError, ValueError):
    from engine.continuation_state import encode_state, decode_state, prepare_identity_references, ANCHOR_KEY
    from engine.cache_manager import (
        KVCacheConfig,
        PrefixKVCacheManager,
        pixel_frames_to_latent_steps,
        latent_steps_to_pixel_frames,
    )
    from pipeline.long_video_director import LongVideoSession
    from pipeline.disk_stream import MiniMaxDiskVideoStreamNode
    from pipeline.native_masked_av import apply_native_masked_av, latent_boundaries
    from pipeline.context_noise import encode_noisy_prefix
    from pipeline.seam_protector import (
        audio_equal_power_crossfade,
        latent_soft_blend,
        stitch_video_latents,
        stitch_audio_latents,
        trim_prefix_frames,
        trim_audio_latents,
        trim_audio_waveform,
        stitch_video_images,
        stitch_audio_waveforms,
        trim_images_and_audio,
        estimate_luminance_gain,
        apply_luminance_gain_fade,
        _standardize_image_tensor,
        _standardize_audio_dict,
    )
    from engine.clip_bin_manager import (
        save_clip_asset,
        load_clip_asset,
        get_project_dir,
        list_projects,
        load_project_index,
        get_clips_for_selection,
        format_clip_label,
        pil_to_tensor,
        create_placeholder_card,
    )
    from engine.timeline_session_manager import (
        get_or_create_timeline_session,
        slice_video_and_audio,
        render_timeline_indicator_image,
        get_input_video_files,
        resolve_video_path,
        standardize_audio_dict,
    )
    from pipeline.smart_seam_color_stitch import (
        find_optimal_seam_cut,
        smart_select_seam_and_trim,
        apply_lab_color_match,
        color_match_and_stitch_segments,
        zero_click_audio_stitch,
    )
    from engine.drift_qa_probe import evaluate_prefix_drift_qa
    from engine.prompt_compiler import compile_h3_four_sentence_prompt, lint_h3_contradictions
    from engine.vram_stage_gate import run_vram_stage_gate
    from pipeline.mv_montage_stitcher import stitch_dual_engine_mv_montage


class AnyType(str):
    """Wildcard type for ComfyUI input slots to accept multiple types (STRING, VHS_FILENAMES, etc.)."""
    def __ne__(self, __value: object) -> bool:
        return False

    def __eq__(self, __value: object) -> bool:
        return True


any_type = AnyType("*")


class MiniMaxPrefixCacheConfigNode:
    """Configures the user-facing continuation mode and video context length."""

    @classmethod
    def INPUT_TYPES(cls):

        return {
            "required": {
                "cache_mode": ([
                    "Native Masked AV (Recommended)",
                    "Safe Native (Fallback)"
                ], {
                    "default": "Native Masked AV (Recommended)",
                    "tooltip": "Native Masked AV 将上一段 AV latent 直接复制到目标开头，并用 ComfyUI 原生 video/audio denoise mask 分别保护；Safe Native 是短片段或旧工作流的兼容备选。"
                }),
                "continuation_frames": (["39", "5", "22", "56", "73", "90", "107", "124", "141", "158", "175", "192"], {
                    "default": "39",
                    "tooltip": "推荐默认 39 帧：1.625 秒 / 65 个音频 tick，音画时长精确对齐。5 帧仅约 0.208 秒 / 8 个 tick，音频上下文很短且需取整，声音接续可能不稳定。按 H3 的 17k+5 帧网格复制前缀。"
                }),
            },
            "optional": {
                "audio_tail_carryover": ([
                    "Match Video Handover",
                    "Full Previous Tail"
                ], {
                    "default": "Match Video Handover",
                    "tooltip": "Match Video Handover 按视频交接位置截取音频；Full Previous Tail 也保留被视频舍弃的尾部声音，可能导致音画交接位置不同。"
                }),
                "audio_feather_ticks": ("INT", {
                    "default": 0, "min": 0, "max": 16, "step": 1,
                    "tooltip": "仅 Masked Prefix 模式生效：音频前缀末尾允许重新生成的 tick 数。0 完全保护；大于 0 会释放部分音频，可能改变交接声音。开启 reference_noise 时强制为 0。"
                }),
                "video_context_denoise": ("FLOAT", {
                    "default": 0.0, "min": 0.0, "max": 1.0, "step": 0.01,
                    "tooltip": "实验项：0 完全冻结视频前缀；大于 0 使用原生软遮罩允许修正历史上下文。可对比 0/0.03/0.06；可能改善历史斑纹，也可能破坏接缝和运动一致性，尚无 H3 画质验证。"
                }),
                "reference_noise": ("BOOLEAN", {
                    "default": False, "label_on": "开启", "label_off": "关闭",
                    "tooltip": "参考帧渐变注噪：所有档位均从 0.4 平滑降至末帧 0。使用带噪/干净两次 VAE 编码的差值作用于原始 latent，避免直接替换造成额外编码偏差。继承的音频不注噪。需接视频 vae 和上一段完整 latent；仅支持 Native Masked AV。"
                }),
                "audio_continuation_mode": (["Masked Prefix", "Native Audio Guide"], {
                    "default": "Masked Prefix",
                    "tooltip": "Masked Prefix 冻结目标音频前缀；Native Audio Guide 通过 H3 原生音频引导条件继承前段声音，目标音频正常生成，不使用音频冻结或 feather。实验选项，需对照验证听感。"
                }),
            }
        }

    RETURN_TYPES = ("MINIMAX_CACHE_CONFIG",)
    RETURN_NAMES = ("cache_config",)
    FUNCTION = "create_config"
    CATEGORY = "MiniMaxH3/PrefixStream"

    def create_config(
        self,
        cache_mode: str = "Native Masked AV (Recommended)",
        continuation_frames: Any = "39",
        audio_tail_carryover: str = "Match Video Handover",
        audio_feather_ticks: int = 0,
        video_context_denoise: float = 0.0,
        reference_noise: bool = False,
        audio_continuation_mode: str = "Masked Prefix",
        **kwargs
    ) -> Tuple[KVCacheConfig]:
        # Accept a saved legacy rolling_frames value if an old workflow sends it.
        actual_rolling = kwargs.get("rolling_frames", continuation_frames)
        try:
            r_frames = int(actual_rolling)
        except (ValueError, TypeError):
            r_frames = 39

        config = KVCacheConfig(
            cache_mode=cache_mode,
            rolling_frames=r_frames,
            audio_tail_carryover=audio_tail_carryover,
            audio_feather_ticks=int(audio_feather_ticks),
            video_context_denoise=float(video_context_denoise),
            reference_noise=bool(reference_noise),
            audio_continuation_mode=audio_continuation_mode,
        )
        return (config,)



def _unpack_latent(latent_dict: Optional[Dict[str, Any]]) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
    """Unpacks video and audio tensors from an H3 latent dict, handling NestedTensor."""
    if latent_dict is None:
        return None, None
    samples = latent_dict.get("samples")
    if samples is None:
        return None, None
    if isinstance(samples, torch.Tensor) and not samples.is_nested:
        return samples, None
    if hasattr(samples, "unbind"):
        parts = list(samples.unbind())
        v = parts[0]
        a = parts[1] if len(parts) > 1 else None
    elif hasattr(samples, "tensors"):
        parts = samples.tensors
        v = parts[0]
        a = parts[1] if len(parts) > 1 else None
    elif isinstance(samples, (tuple, list)):
        v = samples[0]
        a = samples[1] if len(samples) > 1 else None
    elif isinstance(samples, torch.Tensor):
        v = samples
        a = None
    else:
        return None, None
    if v is not None and v.ndim == 4:
        v = v.unsqueeze(0)
    if a is not None and a.ndim == 3:
        a = a.unsqueeze(0)
    return v, a


def pack_av_latent(
    video: torch.Tensor,
    audio: Optional[torch.Tensor] = None,
    original_dict: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """Packs video and audio tensors back into an H3 latent dict matching ComfyUI conventions."""
    out = dict(original_dict) if original_dict is not None else {}
    if audio is None:
        out["samples"] = video
        return out

    try:
        import comfy.nested_tensor
        out["samples"] = comfy.nested_tensor.NestedTensor([video, audio])
    except (ImportError, AttributeError):
        out["samples"] = (video, audio)
    return out


def _pack_nested_streams(video: torch.Tensor, audio: torch.Tensor):
    """Pack two H3 streams without importing ComfyUI during standalone tests."""
    try:
        import comfy.nested_tensor
        return comfy.nested_tensor.NestedTensor((video, audio))
    except (ImportError, AttributeError):
        return (video, audio)


def _drop_head_keyframes(conditioning: Any, protected_frames: int):
    """Remove native keyframes that collide with the hard-protected masked head."""
    if not conditioning:
        return conditioning
    out = []
    for item in conditioning:
        if not isinstance(item, (list, tuple)) or len(item) != 2:
            out.append(item)
            continue
        embedding, extra = item
        updated = dict(extra)
        prior = updated.get("minimax_keyframes") or []
        updated["minimax_keyframes"] = [
            dict(kf) for kf in prior
            if float(kf.get("resolved_frame_index", 0)) >= float(protected_frames)
        ]
        out.append([embedding, updated])
    return out


def _require_native_masked_av_support() -> None:
    """Probe for ComfyUI's native MiniMax H3 per-stream mask implementation."""
    import inspect
    try:
        import comfy.model_base as model_base
        import comfy.ldm.minimax.model as h3_model
    except Exception as exc:
        raise RuntimeError(
            "Native Masked AV requires a current ComfyUI build with MiniMax H3 AV-mask support (PR #15375)."
        ) from exc

    base_cls = getattr(model_base, "MiniMaxH3", None)
    model_cls = getattr(h3_model, "MiniMaxH3Model", None)
    forward = getattr(model_cls, "forward", None) if model_cls is not None else None
    inner = getattr(model_cls, "_forward", None) if model_cls is not None else None
    scale = base_cls.__dict__.get("scale_latent_inpaint") if base_cls is not None else None
    extra_conds = getattr(base_cls, "extra_conds", None) if base_cls is not None else None

    def has_params(fn, *names):
        try:
            params = inspect.signature(fn).parameters
        except (TypeError, ValueError):
            return False
        return all(name in params for name in names)

    def code_names(fn):
        import types
        found = set()
        def walk(code):
            if not isinstance(code, types.CodeType):
                return
            found.update(code.co_names)
            found.update(value for value in code.co_consts if isinstance(value, str))
            for value in code.co_consts:
                if isinstance(value, types.CodeType):
                    walk(value)
        walk(getattr(fn, "__code__", None))
        return found

    extra_names = code_names(extra_conds) if callable(extra_conds) else set()

    available = all((
        base_cls is not None,
        callable(getattr(h3_model, "mask_row_values", None)),
        callable(forward) and has_params(forward, "denoise_mask", "audio_denoise_mask"),
        callable(inner) and has_params(inner, "denoise_mask", "audio_denoise_mask"),
        callable(base_cls.__dict__.get("_token_grid_masks")) if base_cls is not None else False,
        callable(base_cls.__dict__.get("_denoise_mask_conds")) if base_cls is not None else False,
        callable(scale) and has_params(scale, "x", "denoise_mask"),
        callable(extra_conds) and "denoise_mask" in extra_names and "_denoise_mask_conds" in extra_names,
    ))
    if not available:
        raise RuntimeError(
            "Native Masked AV requires a current ComfyUI build with MiniMax H3 AV-mask support from PR #15375. "
            "Update ComfyUI, restart it completely, and reload the workflow."
        )


class MiniMaxPrefixCacheApplierNode:
    """Builds Native Masked AV sampling input or Safe Native fallback conditioning."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("MODEL",),
                "conditioning": ("CONDITIONING",),
            },
            "optional": {
                "cache_config": ("MINIMAX_CACHE_CONFIG",),
                "context_latent": ("LATENT", {"tooltip": "上一段完整的 H3 音视频 LATENT。首段生成时留空。"}),
                "target_latent": ("LATENT", {"tooltip": "连接 MiniMaxH3ReferenceToVideo 的目标 LATENT；Native Masked AV 会输出带独立音视频 noise_mask 的采样 latent。"}),
                "identity_latent": ("LATENT", {"tooltip": "可选的固定人物/外观参考：原始参考图经 H3 VAE 编码的 latent，或首段完整 latent（仅取首帧）。续写时作为独立参考，不覆盖运动前缀。"}),
                "continuation_model": ("MODEL", {"tooltip": "可选，仅在有 context_latent 的续写段使用。可连接经原生 LoRA 节点加载纠偏 LoRA 的模型；首段仍使用 model。不会自动下载或加载 LoRA。"}),
                "vae": ("VAE", {"tooltip": "仅开启 reference_noise 时需要，连接原生 H3 视频 VAE。关闭注噪时不使用。"}),
            }
        }

    RETURN_TYPES = ("MODEL", "CONDITIONING", "MINIMAX_SESSION", "LATENT")
    RETURN_NAMES = ("model", "conditioning", "session", "masked_latent")
    FUNCTION = "apply_cache"
    CATEGORY = "MiniMaxH3/PrefixStream"

    def apply_cache(
        self,
        model: Any,
        conditioning: Any,
        cache_config: Optional[KVCacheConfig] = None,
        context_latent: Optional[Dict[str, Any]] = None,
        target_latent: Optional[Dict[str, Any]] = None,
        identity_latent: Optional[Dict[str, Any]] = None,
        continuation_model: Optional[Any] = None,
        vae: Optional[Any] = None,
    ) -> Tuple[Any, Any, LongVideoSession, Optional[Dict[str, Any]]]:
        cfg = cache_config or KVCacheConfig()
        sess = LongVideoSession(cfg)
        ctx_target = context_latent
        if cfg.reference_noise and ctx_target is not None:
            if not cfg.is_native_masked_av_mode():
                raise ValueError("参考帧注噪需要 Native Masked AV 模式")
            if vae is None:
                raise ValueError("开启 reference_noise 后，请将 H3 视频 VAE 连接到 Applier 的 vae 输入")
            if int(ctx_target.get("minimax_video_start_step", 0)) != 0:
                raise ValueError("参考帧注噪需要上一段完整 latent，请使用采样器完整输出而非裁切后的 latent")
        if ctx_target is not None and continuation_model is not None:
            model = continuation_model

        # Unpack video and audio from latents (supports ComfyUI NestedTensor from H3ContinuousLoadLatent)
        v_ctx, a_ctx_from_latent = _unpack_latent(ctx_target)
        a_ctx = a_ctx_from_latent
        anchor = None
        if identity_latent is not None:
            anchor = identity_latent.get(ANCHOR_KEY)
            if anchor is None:
                anchor, _ = _unpack_latent(identity_latent)
            if anchor is None or anchor.ndim != 5:
                raise ValueError("identity_latent requires an H3 video latent [B, C, T, H, W]")
            anchor = anchor[:, :, :1].clone()
        conditioning, reference_state = prepare_identity_references(conditioning, ctx_target, anchor)
        if target_latent is not None:
            target_latent = dict(target_latent)
            target_latent.pop(ANCHOR_KEY, None)
            target_latent["minimax_video_start_step"] = 0
            target_latent["minimax_audio_start_tick"] = 0
            target_latent.update(reference_state)

        def safe_native_result(reason: Optional[Exception] = None):
            """Run the compatibility path when a native AV mask cannot be built."""
            if reason is not None:
                logger.warning(
                    "[Native Masked AV] %s Falling back to Safe Native for this clip; "
                    "use at least 5 source frames and a target longer than 5 frames to enable native masks.",
                    reason,
                )
            patched_model, out_cond = sess.prepare_next_clip(
                model_patcher=model,
                conditioning=conditioning,
                previous_video_latent=v_ctx,
                previous_audio_latent=a_ctx,
                anchor_video_latent=None,
                source_start_step=int(ctx_target.get("minimax_video_start_step", 0)) if ctx_target else 0,
            )
            return (patched_model, out_cond, sess, target_latent)

        if cfg.is_native_masked_av_mode():
            if ctx_target is None:
                sess.last_rolling_steps = 0
                sess.last_rolling_frames = 0
                sess.is_current_clip_trimmed = False
                logger.info("[Native Masked AV] Initial clip: target latent passes through without a protected prefix.")
                return (model, conditioning, sess, target_latent)
            if target_latent is None:
                raise ValueError(
                    "Native Masked AV mode requires target_latent from MiniMaxH3ReferenceToVideo. "
                    "Connect the applier's masked_latent output to the sampler latent input."
                )
            if v_ctx is None or a_ctx_from_latent is None:
                raise ValueError("Native Masked AV requires a previous latent containing both video and audio streams")
            target_video, target_audio = _unpack_latent(target_latent)
            if target_video is None or target_audio is None:
                raise ValueError("Native Masked AV requires a target latent containing both video and audio streams")
            logger.info(
                "[Native Masked AV] Source %d video steps/%d frames, target %d video steps/%d frames.",
                v_ctx.shape[2], latent_steps_to_pixel_frames(v_ctx.shape[2]),
                target_video.shape[2], latent_steps_to_pixel_frames(target_video.shape[2]),
            )
            _require_native_masked_av_support()
            sess.is_current_clip_trimmed = False
            try:
                out_v, out_a, video_mask, audio_mask, plan = apply_native_masked_av(
                    target_video=target_video,
                    target_audio=target_audio,
                    source_video=v_ctx,
                    source_audio=a_ctx_from_latent,
                    context_frames=cfg.rolling_frames,
                    audio_tail_carryover=cfg.audio_tail_carryover,
                    audio_feather_ticks=0 if cfg.reference_noise else cfg.audio_feather_ticks,
                    source_start_step=int(ctx_target.get("minimax_video_start_step", 0)),
                    source_audio_start_tick=int(ctx_target.get("minimax_audio_start_tick", 0)),
                    video_context_denoise=cfg.video_context_denoise,
                )
            except ValueError as exc:
                if cfg.reference_noise:
                    raise
                geometry_errors = (
                    "at least 5 source frames",
                    "consume the whole target",
                    "has no phase-aligned",
                )
                if any(text in str(exc) for text in geometry_errors):
                    return safe_native_result(exc)
                raise
            if cfg.reference_noise:
                prefix = encode_noisy_prefix(vae, v_ctx, plan)
                out_v[:, :, :plan["context_steps"]] = prefix.to(out_v)
                logger.info("Reference noise: %d frames, cosine 0.4 to final pixel frame 0, paired-encode delta; inherited audio unchanged.",
                            plan["actual_context_frames"])
            out_cond = _drop_head_keyframes(conditioning, int(plan["actual_context_frames"]))
            if cfg.audio_continuation_mode == "Native Audio Guide":
                # Use the same native condition consumed by MiniMaxH3AddGuide.
                # Freeze only video: audio rows keep their stream timestep.
                guide = a_ctx_from_latent[..., plan["audio_start_tick"]:plan["audio_end_tick"]].clone()
                out_a = target_audio.clone()
                audio_mask = torch.ones_like(audio_mask)
                guided = []
                for embedding, extra in out_cond:
                    updated = dict(extra)
                    updated["minimax_keyframes"] = list(updated.get("minimax_keyframes") or []) + [
                        {"resolved_frame_index": 0, "audio_latent": guide}
                    ]
                    guided.append([embedding, updated])
                out_cond = guided
                logger.info("[Native Audio Guide] Conditioning on %d inherited audio ticks; target audio is generated normally.",
                            plan["audio_steps"])
            masked_latent = pack_av_latent(out_v, out_a, target_latent)
            masked_latent["noise_mask"] = _pack_nested_streams(video_mask, audio_mask)
            masked_latent["minimax_prefix_frames"] = int(plan["actual_context_frames"])
            sess.last_rolling_steps = int(plan["context_steps"])
            sess.last_rolling_frames = int(plan["actual_context_frames"])
            logger.info(
                "[Native Masked AV] Protected %d video frames; audio context %d ticks (%s); source latent %d:%d.",
                sess.last_rolling_frames,
                plan["audio_steps"],
                cfg.audio_continuation_mode,
                plan["start_t"],
                plan["end_t"],
            )
            return (model, out_cond, sess, masked_latent)

        # Safe Native fallback: inject grid-aligned keyframe conditioning only.
        return safe_native_result()


class MiniMaxTrimPrefixLatentNode:
    """Automatically trims redundant prefix overlap frames from video and audio.
    
    SUPPORTED MODES:
    1. Pixel & Waveform Space (RECOMMENDED): Connect decoded 'images' and 'audio'.
       Trims decoded frames without another VAE encode/decode pass.
    2. Latent Space: Connect 'latent'. Trims raw latent steps.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "trim_frames": ("INT", {
                    "default": 0, "min": 0, "max": 124, "step": 1,
                    "tooltip": "裁切的前置重叠帧数 (如 22 帧)。设为 0 且连接了 session/config 时将自动识别"
                }),
            },
            "optional": {
                "images": ("IMAGE", {"tooltip": "【强烈推荐】解码后的完整画面。在像素空间裁切，彻底杜绝 VAE 闪烁与偏色！"}),
                "audio": ("AUDIO", {"tooltip": "【强烈推荐】解码后的音频。精确同步毫秒级样本截断，杜绝音画不同步"}),
                "latent": ("LATENT", {"tooltip": "原始采样 latent (可选，若已连接 images/audio 则无需裁切 latent)"}),
                "session": ("MINIMAX_SESSION",),
                "cache_config": ("MINIMAX_CACHE_CONFIG",),
                "fps": ("FLOAT", {"default": 24.0, "min": 1.0, "max": 120.0}),
                "match_tail": ("BOOLEAN", {"default": True, "tooltip": "尾部时长严格对齐：消除 H3 40Hz 音频与 24fps 画面约8ms的网格舍入累积误差"}),
                "video_latent": ("LATENT",),  # Backward compatibility alias
            }
        }

    RETURN_TYPES = ("IMAGE", "AUDIO", "LATENT")
    RETURN_NAMES = ("trimmed_images", "trimmed_audio", "trimmed_latent")
    FUNCTION = "trim"
    CATEGORY = "MiniMaxH3/PrefixStream"

    def trim(
        self,
        trim_frames: int = 0,
        images: Optional[torch.Tensor] = None,
        audio: Optional[Dict[str, Any]] = None,
        latent: Optional[Dict[str, Any]] = None,
        video_latent: Optional[Dict[str, Any]] = None,
        session: Optional[LongVideoSession] = None,
        cache_config: Optional[KVCacheConfig] = None,
        fps: float = 24.0,
        match_tail: bool = True,
        **kwargs
    ) -> Tuple[Optional[torch.Tensor], Optional[Dict[str, Any]], Optional[Dict[str, Any]]]:
        # 1. Determine trim frame count
        actual_trim_frames = trim_frames
        if actual_trim_frames <= 0:
            if session is not None:
                # Strictly respect session: 0 for initial clip, >0 for continuation clips
                actual_trim_frames = session.last_rolling_frames
            elif cache_config is not None:
                # Protect initial clip from accidental trimming when session is not connected:
                # Only continuation clips contain 'noise_mask' in target_latent
                target_latent = latent if latent is not None else video_latent
                has_noise_mask = bool(target_latent is not None and isinstance(target_latent, dict) and "noise_mask" in target_latent)
                if has_noise_mask:
                    actual_trim_frames = int(target_latent.get("minimax_prefix_frames", cache_config.rolling_frames))
                else:
                    logger.info("[Trim AV] No session connected and no noise_mask in latent; preserving full initial clip (0 trim frames).")
                    actual_trim_frames = 0


        # 2. Pixel & audio waveform trimming (Golden Standard)
        out_images = None
        out_audio = None
        if images is not None:
            out_images, out_audio = trim_images_and_audio(
                images=images,
                audio=audio,
                trim_frames=actual_trim_frames,
                fps=fps,
                match_tail=match_tail
            )
            logger.info(
                "[Trim AV] Cleanly trimmed %d leading frames in pixel space. Output: %d frames (~%.2fs). Zero VAE flicker.",
                actual_trim_frames, out_images.shape[0], out_images.shape[0] / float(fps)
            )
        elif audio is not None:
            normalized = _standardize_audio_dict(audio)
            if normalized is not None:
                sr = int(normalized.get("sample_rate", 32000))
                head = max(0, round(actual_trim_frames / fps * sr))
                out_audio = dict(normalized, waveform=normalized["waveform"][..., head:])
                if head > 0:
                    out_audio.pop("minimax_audio_context", None)
                    if head <= normalized["waveform"].shape[-1]:
                        out_audio["minimax_audio_context"] = normalized["waveform"][..., :head]

        if actual_trim_frames > 0 and session is not None:
            session.is_current_clip_trimmed = True

        # 3. Latent trimming (fallback / passthrough)
        out_latent = None
        target_latent = latent if latent is not None else video_latent
        if target_latent is not None:
            trim_steps = 0
            if actual_trim_frames > 0:
                trim_steps = pixel_frames_to_latent_steps(actual_trim_frames)

            v, a_from_latent = _unpack_latent(target_latent)
            if v is None:
                v = target_latent.get("samples")

            if v is not None:
                source_step = int(target_latent.get("minimax_video_start_step", 0))
                if actual_trim_frames > 0:
                    bounds = latent_boundaries(v.shape[2], source_step)
                    trim_steps = next((i for i, end in enumerate(bounds) if end >= actual_trim_frames), v.shape[2])
                if trim_steps > 0:
                    trimmed_v = v[:, :, trim_steps:]
                else:
                    trimmed_v = v
                trimmed_a = None
                if a_from_latent is not None:
                    if trim_steps > 0:
                        end_frame = latent_boundaries(source_step + trim_steps)[-1]
                        audio_origin = int(target_latent.get("minimax_audio_start_tick", 0))
                        audio_trim_steps = min(max(0, round(end_frame * 40.0 / 24.0) - audio_origin), a_from_latent.shape[-1])
                        trimmed_a = trim_audio_latents(a_from_latent, audio_trim_steps)
                    else:
                        trimmed_a = a_from_latent
                out_latent = pack_av_latent(trimmed_v, trimmed_a, original_dict=target_latent)
                if trim_steps > 0:
                    out_latent["minimax_video_start_step"] = int(target_latent.get("minimax_video_start_step", 0)) + min(trim_steps, v.shape[2])
                    out_latent["minimax_audio_start_tick"] = int(target_latent.get("minimax_audio_start_tick", 0)) + (audio_trim_steps if a_from_latent is not None else 0)
                    out_latent.pop("noise_mask", None)
                    out_latent["minimax_prefix_frames"] = 0
            else:
                out_latent = target_latent

        if out_images is None:
            out_images = images if images is not None else torch.empty((0, 768, 1344, 3), dtype=torch.float32)
        if out_audio is None:
            out_audio = audio

        return (out_images, out_audio, out_latent)



class MiniMaxLongVideoStitcherNode:
    """Seamlessly stitches previous long video and current newly generated clip in pixel & waveform space.

    Applies:
    1. Zero-VAE-distortion pixel-space joining with luminance gain matching and smooth cosine S-curve crossfade.
    2. Waveform-space sample-accurate audio concatenation with complementary cosine overlap gains.
    3. Handles initial clip mode gracefully when prev_images or prev_audio is empty or None.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "trim_frames": ("INT", {
                    "default": 0, "min": 0, "max": 192, "step": 1,
                    "tooltip": "当前片段在缝合前需要裁切的前缀帧数。设为 0 时若连接了 session/config 将自动获取"
                }),
                "crossfade_frames": ("INT", {
                    "default": 4, "min": 0, "max": 30, "step": 1,
                    "tooltip": "画面重叠接缝处的余弦 S 曲线混合平滑过渡帧数 (推荐 2~6 帧)"
                }),
                "luminance_match": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "自动检测并平滑纠正前后片段的全局亮度色差，杜绝接缝闪光"
                }),
                "luminance_fade_frames": ("INT", {
                    "default": 16, "min": 1, "max": 60, "step": 1,
                    "tooltip": "亮度增益向原生亮度平滑回退过渡的帧数"
                }),
                "crossfade_ms": ("FLOAT", {
                    "default": 15.0, "min": 0.0, "max": 500.0, "step": 1.0,
                    "tooltip": "音频接缝余弦淡化时长（毫秒）。保留重叠上下文时交叉淡化，否则在接缝两侧淡出/淡入，保持音画时长。"
                }),
                "fps": ("FLOAT", {"default": 24.0, "min": 1.0, "max": 120.0}),
            },
            "optional": {
                "prev_images": ("IMAGE", {"tooltip": "前置已累积的长视频画面。首段生成时可留空"}),
                "curr_images": ("IMAGE", {"tooltip": "当前生成的片段画面 (来自 VAEDecode 或 TrimPrefix)"}),
                "prev_audio": ("AUDIO", {"tooltip": "前置已累积的音频流。首段生成时可留空"}),
                "curr_audio": ("AUDIO", {"tooltip": "当前生成的音频流 (来自 VAEDecodeAudio 或 TrimPrefix)"}),
                "session": ("MINIMAX_SESSION",),
                "cache_config": ("MINIMAX_CACHE_CONFIG",),
                "smart_seam_search": ("BOOLEAN", {
                    "default": False,
                    "tooltip": "可选：在重叠窗口内自动搜索像素/亮度/运动最平顺的最佳切点，替代固定第0帧盲切"
                }),
            }
        }

    RETURN_TYPES = ("IMAGE", "AUDIO", "INT")
    RETURN_NAMES = ("stitched_images", "stitched_audio", "total_frames")
    FUNCTION = "stitch"
    CATEGORY = "MiniMaxH3/PrefixStream"

    def stitch(
        self,
        trim_frames: int = 0,
        crossfade_frames: int = 4,
        luminance_match: bool = True,
        luminance_fade_frames: int = 16,
        crossfade_ms: float = 15.0,
        fps: float = 24.0,
        prev_images: Optional[torch.Tensor] = None,
        curr_images: Optional[torch.Tensor] = None,
        prev_audio: Optional[Dict[str, Any]] = None,
        curr_audio: Optional[Dict[str, Any]] = None,
        session: Optional[LongVideoSession] = None,
        cache_config: Optional[KVCacheConfig] = None,
        smart_seam_search: bool = False,
        **kwargs
    ) -> Tuple[Optional[torch.Tensor], Optional[Dict[str, Any]], int]:
        # Determine trim frames for curr_images/audio
        actual_trim_frames = trim_frames
        if actual_trim_frames <= 0:
            if session is not None and getattr(session, "is_current_clip_trimmed", False):
                logger.info(
                    "[Long Video Stitcher] Input clip was already trimmed upstream by MiniMaxTrimPrefixLatent. "
                    "Skipping duplicate trimming (0 trim frames) to prevent audio/video loss."
                )
                actual_trim_frames = 0
            elif session is not None:
                actual_trim_frames = session.last_rolling_frames
            elif cache_config is not None and prev_images is not None and prev_images.shape[0] > 0:
                actual_trim_frames = cache_config.rolling_frames

        if smart_seam_search and actual_trim_frames > 1:
            out_images, out_audio, _ = color_match_and_stitch_segments(
                prev_images=prev_images,
                curr_images=curr_images,
                overlap_frames=actual_trim_frames,
                smart_seam=True,
                color_match_mode="luma_only" if luminance_match else "none",
                grade_decay_frames=luminance_fade_frames,
                blend_mode="cosine_alpha" if crossfade_frames > 0 else "hard_cut",
                blend_frames=crossfade_frames,
                prev_audio=prev_audio,
                curr_audio=curr_audio,
                audio_crossfade_ms=crossfade_ms,
                fps=fps,
            )
            total_f = int(out_images.shape[0]) if out_images is not None else 0
            return (out_images, out_audio, total_f)

        # Standardize inputs
        prev_img_std = _standardize_image_tensor(prev_images)
        curr_img_std = _standardize_image_tensor(curr_images)
        prev_aud_std = _standardize_audio_dict(prev_audio)
        curr_aud_std = _standardize_audio_dict(curr_audio)

        out_images = None
        if curr_img_std is not None and curr_img_std.shape[0] > 0:
            out_images = stitch_video_images(
                prev_images=prev_img_std,
                curr_images=curr_img_std,
                trim_frames=actual_trim_frames,
                crossfade_frames=crossfade_frames,
                luminance_match=luminance_match,
                luminance_fade_frames=luminance_fade_frames,
            )
        elif prev_img_std is not None:
            out_images = prev_img_std
        else:
            out_images = torch.empty((0, 768, 1344, 3), dtype=torch.float32)

        curr_total_f = curr_img_std.shape[0] if curr_img_std is not None else 0
        out_audio = None
        if curr_aud_std is not None:
            out_audio = stitch_audio_waveforms(
                prev_audio=prev_aud_std,
                curr_audio=curr_aud_std,
                curr_total_frames=curr_total_f,
                trim_frames=actual_trim_frames,
                crossfade_ms=crossfade_ms,
                fps=fps,
            )
        else:
            out_audio = prev_aud_std

        total_f = int(out_images.shape[0]) if out_images is not None else 0
        logger.info(
            "[Long Video Stitcher] Seamlessly stitched. Total video: %d frames (~%.2fs).",
            total_f, total_f / float(fps)
        )
        return (out_images, out_audio, total_f)


class MiniMaxCacheMonitorNode:
    """Provides operational telemetry for the current continuation session."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "session": ("MINIMAX_SESSION",),
            }
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("telemetry_report",)
    FUNCTION = "report"
    CATEGORY = "MiniMaxH3/PrefixStream"

    def report(self, session: LongVideoSession) -> Tuple[str]:
        native_masked = session.config.is_native_masked_av_mode()
        status_str = "Native Masked AV" if native_masked else "Safe Native fallback"
        report_str = (
            f"=== MiniMax H3 Continuation Telemetry ===\n"
            f"Mode: {status_str}\n"
            f"Current Clip: #{session.current_clip_index}\n"
            f"Configured Protected Context: {session.config.rolling_frames} frames\n"
            f"Last Protected Context: {session.last_rolling_frames} frames ({session.last_rolling_steps} latent steps)\n"
            f"Accumulated Clips: {len(session.accumulated_video_latents)}"
        )
        return (report_str,)


class MiniMaxSaveLatentNode:
    """Saves the complete joint MiniMax H3 AV latent to safetensors for seamless continuation.
    
    Guarantees 100% independent, standalone operation without requiring external continuation suites.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "latent": ("LATENT", {"tooltip": "Sampler output joint AV latent to save."}),
                "filename_prefix": ("STRING", {
                    "default": "minimax_h3/clip",
                    "tooltip": "Subfolder and filename prefix in ComfyUI output directory."
                }),
                "clip_index": ("INT", {
                    "default": 1, "min": 0, "max": 99999, "step": 1,
                    "tooltip": "Fixed chain slot (1, 2, 3...). 0 = auto-incrementing."
                }),
            },
        }

    RETURN_TYPES = ("STRING", "STRING")
    RETURN_NAMES = ("saved_path", "latent_info")
    OUTPUT_NODE = True
    FUNCTION = "save"
    CATEGORY = "MiniMaxH3/PrefixStream"

    def save(
        self,
        latent: Dict[str, Any],
        filename_prefix: str = "minimax_h3/clip",
        clip_index: int = 1,
        **kwargs
    ) -> Tuple[str, str]:
        if latent is None:
            raise ValueError("MiniMaxSaveLatent: 'latent' input is required.")

        video, audio = _unpack_latent(latent)
        if video is None:
            raise ValueError("MiniMaxSaveLatent: latent contains no video samples.")

        video_cpu = video.detach().cpu().contiguous()
        audio_cpu = audio.detach().cpu().contiguous() if audio is not None else None

        try:
            import folder_paths
            base_dir = folder_paths.get_output_directory()
        except Exception:
            base_dir = "output"

        target_dir = os.path.join(base_dir, os.path.dirname(filename_prefix))
        os.makedirs(target_dir, exist_ok=True)

        base_name = os.path.basename(filename_prefix)
        if clip_index > 0:
            filename = f"{base_name}_{clip_index:05d}.safetensors"
        else:
            filename = f"{base_name}_temp.safetensors"
        full_path = os.path.join(target_dir, filename)

        tensors = {"video": video_cpu}
        if audio_cpu is not None:
            tensors["audio"] = audio_cpu
        state_tensors, state_metadata = encode_state(latent)
        tensors.update(state_tensors)

        frame_count = latent_steps_to_pixel_frames(video_cpu.shape[2])

        if st_save is not None:
            st_save(
                tensors,
                full_path,
                metadata={
                    "format": "minimax_h3_av_latent",
                    "frame_count": str(frame_count),
                    "clip_index": str(clip_index),
                    "video_shape": json.dumps(list(video_cpu.shape)),
                    "audio_shape": json.dumps(list(audio_cpu.shape)) if audio_cpu is not None else "none",
                    **state_metadata,
                }
            )
        else:
            tensors["minimax_metadata"] = state_metadata
            torch.save(tensors, full_path)

        info_str = f"{frame_count} frames | Video {tuple(video_cpu.shape)}"
        if audio_cpu is not None:
            info_str += f" | Audio {tuple(audio_cpu.shape)}"
        logger.info("[Save Latent] Successfully saved %s -> %s", info_str, full_path)
        return (full_path, info_str)


class MiniMaxLoadLatentNode:
    """Loads a saved MiniMax H3 joint AV latent for continuation and long-video stitching.
    
    Guarantees 100% independent, standalone operation: compatible with any saved H3 safetensors latent.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "latent_path": ("STRING", {
                    "default": "minimax_h3/clip",
                    "tooltip": "Folder or filepath relative to ComfyUI output, or absolute path."
                }),
                "clip_index": ("INT", {
                    "default": 1, "min": 0, "max": 99999, "step": 1,
                    "tooltip": "Clip index to load (e.g. 1 to continue Clip 2). 0 = latest file."
                }),
            }
        }

    RETURN_TYPES = ("LATENT", "STRING", "STRING")
    RETURN_NAMES = ("latent", "loaded_path", "latent_info")
    FUNCTION = "load"
    CATEGORY = "MiniMaxH3/PrefixStream"

    def load(
        self,
        latent_path: str = "minimax_h3/clip",
        clip_index: int = 1
    ) -> Tuple[Dict[str, Any], str, str]:
        try:
            import folder_paths
            base_dir = folder_paths.get_output_directory()
        except Exception:
            base_dir = "output"

        p = (latent_path or "").strip().strip('"').strip("'")
        if not os.path.isabs(p):
            full_target = os.path.join(base_dir, p)
        else:
            full_target = p

        if os.path.isfile(full_target):
            target_file = full_target
        else:
            parent_dir = os.path.dirname(full_target)
            base_name = os.path.basename(full_target)
            if os.path.isdir(parent_dir):
                if clip_index > 0:
                    candidates = [f for f in os.listdir(parent_dir) if f.startswith(base_name) and f.endswith(".safetensors")]
                    matched = [f for f in candidates if f"_{clip_index:05d}" in f or f"_{clip_index}." in f or f"_{clip_index}_" in f]
                    if matched:
                        target_file = os.path.join(parent_dir, matched[0])
                    else:
                        target_file = os.path.join(parent_dir, f"{base_name}_{clip_index:05d}.safetensors")
                else:
                    candidates = [os.path.join(parent_dir, f) for f in os.listdir(parent_dir) if f.endswith(".safetensors")]
                    if candidates:
                        target_file = max(candidates, key=os.path.getmtime)
                    else:
                        target_file = full_target
            else:
                target_file = full_target

        if not os.path.exists(target_file):
            raise FileNotFoundError(f"MiniMaxLoadLatent: file not found at '{target_file}'")

        if st_load is not None:
            tensors = st_load(target_file, device="cpu")
            with safe_open(target_file, framework="pt", device="cpu") as stored:
                state_metadata = stored.metadata() or {}
        else:
            try:
                tensors = torch.load(target_file, map_location="cpu")
            except RuntimeError as exc:
                if "safetensors is not installed" in str(exc):
                    with open(target_file, "rb") as f:
                        tensors = torch.load(f, map_location="cpu")
                else:
                    raise

        if "video" not in tensors:
            raise ValueError(f"MiniMaxLoadLatent: '{target_file}' does not contain 'video' tensor.")

        video = tensors["video"]
        audio = tensors.get("audio", None)

        out_latent = pack_av_latent(video, audio)
        out_latent.update(decode_state(tensors, state_metadata if st_load is not None else tensors.get("minimax_metadata", {})))
        frame_count = latent_steps_to_pixel_frames(video.shape[2])
        info_str = f"{frame_count} frames | Video {tuple(video.shape)}"
        if audio is not None:
            info_str += f" | Audio {tuple(audio.shape)}"

        logger.info("[Load Latent] Successfully loaded %s (%s)", target_file, info_str)
        return (out_latent, target_file, info_str)


class MiniMaxClipBinSaverNode:
    """Saves a unified MiniMax H3 AV Latent into the Clip Bin media pool with keyframes, preview card, and rich metadata."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "latent": ("LATENT", {"tooltip": "【核心音画潜空间】采样器输出的原生联合音画 Latent (支持 MiniMax H3 官方 NestedTensor，包含完整视频与音频潜空间)"}),
                "project_name": ("STRING", {
                    "default": "Default_Project",
                    "tooltip": "【项目/分镜箱名称】指定当前镜头归档的项目库（例如：科幻短片、广告场景1）。不同项目之间素材完全隔离，方便多故事独立管理"
                }),
                "shot_tag": ("STRING", {
                    "default": "Auto (自动编号)",
                    "tooltip": "【镜头标签/备注】镜头的编号或简要动作描述（如：'Shot 1'、'男主回眸'、'远景空镜'）。填 'Auto (自动编号)' 时系统将根据项目内已有镜头数量自动顺延递增为 Shot 1, Shot 2..."
                }),
                "rating": ("INT", {
                    "default": 4, "min": 1, "max": 5, "step": 1,
                    "tooltip": "【镜头星标打分】1~5 星质量评级。后续使用 Clip Bin Picker 加载接力时，可按星级一键过滤掉废案镜头，只接力高分镜头"
                }),
            },
            "optional": {
                "images": ("IMAGE", {"tooltip": "【渲染像素画面】连接当前片段解码后的画面 (来自 VAEDecode 或 TrimPrefix)。连接后系统将自动截取真实的首帧与尾帧，生成超高清并排缩略图卡片！"}),
                "audio": ("AUDIO", {"tooltip": "【音频流】连接当前片段的音频 (来自 TrimPrefix 或 VAEDecodeAudio)。当自动编码保存 MP4 视频时，将作为音轨同步封装"}),
                "prompt": ("STRING", {"default": "", "tooltip": "【本段正向提示词】连接输入文本 (Input Text/Prompt)。自动入库保存到 meta.json，以便后续回顾镜头剧情与接力参考"}),
                "parent_clip_id": ("STRING", {"default": "", "tooltip": "【父镜头血缘ID】连接上一段 Clip Bin Picker 输出的 clip_id。用于在元数据中清晰记录多版本分支历史与承接血缘"}),
                "video_file_name": (any_type, {"default": "", "tooltip": "【关联合成视频名】连接当前片段合成保存节点 (VHS_VideoCombine) 的 Filenames 输出，或手动输入关联的 MP4 文件名，系统将自动将该视频归档到资产包中"}),
                "save_video": ("BOOLEAN", {"default": True, "tooltip": "【归档完整视频】是否在资产包内归档或编码生成完整 MP4 视频文件。开启后 Clip Bin Picker 画廊将支持悬停实时微动播放与声画视听弹窗！"}),
            }
        }

    RETURN_TYPES = ("STRING", "IMAGE", "STRING")
    RETURN_NAMES = ("clip_id", "preview_image", "bin_path")
    OUTPUT_NODE = True
    FUNCTION = "save_clip"
    CATEGORY = "MiniMaxH3/ClipBin"

    def save_clip(
        self,
        latent: Dict[str, Any],
        project_name: str = "Default_Project",
        shot_tag: str = "Auto (自动编号)",
        rating: int = 4,
        images: Optional[torch.Tensor] = None,
        audio: Optional[Dict[str, Any]] = None,
        prompt: str = "",
        parent_clip_id: str = "",
        video_file_name: Any = "",
        save_video: bool = True,
        **kwargs
    ) -> Dict[str, Any]:
        if latent is None:
            raise ValueError("MiniMaxClipBinSaver: 'latent' input is required.")

        video, audio_lat = _unpack_latent(latent)
        if video is None:
            raise ValueError("MiniMaxClipBinSaver: latent contains no video samples.")

        images = _standardize_image_tensor(images)
        audio = _standardize_audio_dict(audio)

        actual_shot = (shot_tag or "").strip()
        if actual_shot.startswith("Auto") or not actual_shot:
            idx = load_project_index(project_name)
            actual_shot = f"Shot {len(idx.get('clips', [])) + 1}"

        # Handle video_file_name if passed as list/tuple from VHS_VideoCombine Filenames
        resolved_video_name = ""
        def _extract_filename(val: Any) -> str:
            if isinstance(val, (list, tuple)):
                if not val:
                    return ""
                # Prioritize video extensions in list/tuple
                for item in val:
                    if isinstance(item, (list, tuple)):
                        extracted = _extract_filename(item)
                        if extracted and any(extracted.lower().endswith(e) for e in (".mp4", ".webm", ".mov", ".mkv")):
                            return extracted
                    elif isinstance(item, str) and any(item.lower().endswith(e) for e in (".mp4", ".webm", ".mov", ".mkv")):
                        return item.strip()
                return _extract_filename(val[-1])
            s = str(val).strip()
            return s if s.lower() not in ("true", "false", "none") else ""

        if video_file_name is not None:
            resolved_video_name = _extract_filename(video_file_name)
            # If it's a full path, keep basename for friendly display
            if resolved_video_name:
                resolved_video_name = os.path.basename(resolved_video_name)

        meta_obj, clip_dir, preview_pil = save_clip_asset(
            video_tensor=video,
            audio_tensor=audio_lat,
            images=images,
            project_name=project_name,
            shot_tag=actual_shot,
            prompt=prompt if isinstance(prompt, str) else str(prompt),
            rating=rating,
            parent_clip_id=parent_clip_id if isinstance(parent_clip_id, str) else str(parent_clip_id),
            associated_video_path=resolved_video_name,
            raw_video_source=video_file_name,
            audio_dict=audio,
            save_video=save_video,
            continuation_state=latent,
        )

        preview_tensor = pil_to_tensor(preview_pil)

        try:
            import folder_paths
            base_dir = folder_paths.get_output_directory()
            subfolder = os.path.relpath(clip_dir, base_dir)
        except Exception:
            subfolder = ""

        ui_images = [{
            "filename": "preview.png",
            "subfolder": subfolder,
            "type": "output"
        }]

        logger.info("[Clip Bin Saver] Stored clip '%s' in '%s' (%s frames | ⭐%s | tag: %s | video: '%s')",
                    meta_obj.clip_id, project_name, meta_obj.frames, meta_obj.rating, actual_shot, resolved_video_name)

        return {
            "ui": {"images": ui_images},
            "result": (meta_obj.clip_id, preview_tensor, clip_dir)
        }


class MiniMaxClipBinPickerNode:
    """Visually browses, filters, and loads clips from the Clip Bin with instant tail-frame output."""

    @classmethod
    def INPUT_TYPES(cls):
        projects = list_projects()
        default_proj = projects[0] if projects else "Default_Project"
        return {
            "required": {
                "project_name": ("STRING", {
                    "default": default_proj,
                    "tooltip": "【选择项目库】要读取素材的项目文件夹名称（如 Default_Project）。可在 ComfyUI 运行控制台查看已存在的项目名称列表"
                }),
                "mode": ([
                    "Auto (首段全新 / 后续自动接力)",
                    "Force Initial (强制新建首段，无上下文)",
                    "Strict Chaining (必须接力指定或最新镜头)"
                ], {
                    "default": "Auto (首段全新 / 后续自动接力)",
                    "tooltip": "【运行工作模式】\n• Auto（强烈推荐）：若项目库为空自动作为首段全新生成；后续运行时全自动接续上一段，无需任何拔线或手动操作！\n• Force Initial：强制开辟首段，忽略库内所有历史素材。\n• Strict Chaining：严格接力模式，库内无镜头时直接报错提示"
                }),
                "filter_rating": ([
                    "All (1-5 ⭐)",
                    "⭐⭐⭐+ (3+ ⭐)",
                    "⭐⭐⭐⭐+ (4+ ⭐)",
                    "⭐⭐⭐⭐⭐ (5 ⭐)"
                ], {
                    "default": "All (1-5 ⭐)",
                    "tooltip": "【星级过滤器】只读取大于等于该评级的镜头（如过滤掉 1~3 星的测试废案，只接续 4 星或 5 星的满意镜头）"
                }),
                "clip_selection": ("STRING", {
                    "default": "latest",
                    "tooltip": "【镜头定位】\n• 填 'latest'（默认）：自动调取最新生成的优质镜头进行无缝接续\n• 填 'initial'（树根节点）：Auto 模式下全新生成，无历史上下文\n• 填 clip_id（如 clip_20260908...）：精确跳转或回溯到指定的历史镜头开启新分支\n• 填 shot 名称：按镜头标签名称匹配"
                }),
            },
            "optional": {
                "custom_clip_path": ("STRING", {
                    "default": "",
                    "tooltip": "【自定义物理路径覆盖】可选高级选项。填入绝对路径可直接载入任意磁盘目录下的 Clip Bin 镜头文件夹"
                }),
                "view_mode": ([
                    "Deck (卡片流)",
                    "Tree (关系树)"
                ], {
                    "default": "Deck (卡片流)",
                    "tooltip": "【界面展现模式】卡片横向滚动流 或 分支血缘拓扑树（亦可在节点界面顶部一键切换）"
                }),
            }
        }

    RETURN_TYPES = ("LATENT", "IMAGE", "IMAGE", "STRING", "STRING")
    RETURN_NAMES = ("latent", "tail_frame", "first_frame", "prompt", "clip_id")
    FUNCTION = "pick_clip"
    CATEGORY = "MiniMaxH3/ClipBin"


    def pick_clip(
        self,
        project_name: str = "Default_Project",
        mode: str = "Auto (首段全新 / 后续自动接力)",
        filter_rating: str = "All (1-5 ⭐)",
        clip_selection: str = "latest",
        custom_clip_path: str = "",
        view_mode: str = "Deck (卡片流)",
        **kwargs
    ) -> Dict[str, Any]:
        p_name = (project_name or "Default_Project").strip()
        custom_p = (custom_clip_path or "").strip().strip('"').strip("'")

        if custom_p and os.path.isdir(custom_p):
            target_clip_dir = custom_p
            p_name = os.path.basename(os.path.dirname(custom_p)) or p_name
            target_clip_id = os.path.basename(custom_p)
        else:
            idx = load_project_index(p_name)
            clips = idx.get("clips", [])

            # Explicit root selection must not resolve to the latest saved clip.
            selected_initial = (clip_selection or "").strip().lower() == "initial"
            if selected_initial and mode.startswith("Strict Chaining"):
                raise ValueError("Strict Chaining requires a clip; choose Auto or Force Initial to generate a new root.")
            is_initial_mode = mode.startswith("Force Initial") or (
                mode.startswith("Auto") and (selected_initial or not clips)
            )

            if is_initial_mode:
                logger.info("[Clip Bin Picker] Operating in Initial Generation mode for project '%s' (Zero prior context).", p_name)
                card = create_placeholder_card("✨ Initial Clip Mode", f"Project: {p_name} | Ready for First Clip (No Context)")
                placeholder_tensor = pil_to_tensor(card)
                return {
                    "ui": {"images": []},
                    "result": (None, placeholder_tensor, placeholder_tensor, "", "[INITIAL_GENERATION]")
                }

            if not clips:
                raise ValueError(f"MiniMaxClipBinPicker: No clips found in project '{p_name}'. "
                                 f"Switch mode to 'Auto' to generate the first clip.")

            # Parse star rating filter
            min_stars = 1
            if filter_rating.startswith("⭐⭐⭐⭐⭐"):
                min_stars = 5
            elif filter_rating.startswith("⭐⭐⭐⭐"):
                min_stars = 4
            elif filter_rating.startswith("⭐⭐⭐"):
                min_stars = 3

            filtered = [c for c in clips if c.get("rating", 3) >= min_stars]
            if not filtered:
                logger.warning("[Clip Bin Picker] No clips match rating >= %s in '%s', falling back to all clips.",
                               min_stars, p_name)
                filtered = clips

            sel = (clip_selection or "latest").strip()
            if sel.lower() in ("latest", "", "0", "auto", "default"):
                target_clip = filtered[0]
                target_clip_id = target_clip["clip_id"]
            else:
                # Substring / exact match
                matched = [c for c in clips if sel in c.get("clip_id", "") or sel in c.get("shot_tag", "")]
                if matched:
                    target_clip_id = matched[0]["clip_id"]
                else:
                    target_clip_id = sel

        video, audio, tail_tensor, first_tensor, meta_dict = load_clip_asset(p_name, target_clip_id)
        out_latent = pack_av_latent(video, audio)
        out_latent.update(decode_state(meta_dict, meta_dict))

        prompt_str = meta_dict.get("prompt", "")
        frames = meta_dict.get("frames", latent_steps_to_pixel_frames(video.shape[2]))
        logger.info("[Clip Bin Picker] Loaded clip '%s' (%s frames | ⭐%s | tag: '%s')",
                    target_clip_id, frames, meta_dict.get("rating", 3), meta_dict.get("shot_tag", ""))

        return {
            "ui": {"images": []},
            "result": (out_latent, tail_tensor, first_tensor, prompt_str, target_clip_id)
        }




class MiniMaxClipBinTreePickerNode(MiniMaxClipBinPickerNode):
    """Visual Tree & DAG Lineage Graph Picker for MiniMax Clip Bin.
    
    Identical pipeline compatibility with MiniMaxClipBinPickerNode, but defaults to
    an interactive branching DAG tree view for exploring non-linear story branches.
    """

    @classmethod
    def INPUT_TYPES(cls):
        types = super().INPUT_TYPES()
        import copy
        types = copy.deepcopy(types)
        types["optional"]["view_mode"] = ([
            "Tree (关系树)",
            "Deck (卡片流)"
        ], {
            "default": "Tree (关系树)",
            "tooltip": "【界面展现模式】分支血缘拓扑树 或 卡片横向滚动流（亦可在节点界面顶部一键切换）"
        })
        return types

    CATEGORY = "MiniMaxH3/ClipBin"


class MiniMaxSafeVAEDecodeNode:
    """Safe VAE Video Decoder that gracefully handles None in Initial Clip Mode.
    
    When samples is None (e.g. initial generation with zero prior context),
    it safely returns an empty IMAGE tensor instead of crashing with TypeError.
    When samples is present, it delegates to vae.decode() with full fidelity.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "vae": ("VAE",),
            },
            "optional": {
                "samples": ("LATENT",),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "decode"
    CATEGORY = "MiniMaxH3/ClipBin"

    def decode(self, vae: Any, samples: Optional[Dict[str, Any]] = None) -> Tuple[torch.Tensor]:
        if samples is None:
            logger.info("[Safe VAE Decode] No previous video latent provided (Initial Clip Mode). Passing through empty.")
            return (torch.empty((0, 768, 1344, 3), dtype=torch.float32),)

        v, _ = _unpack_latent(samples)
        if v is None:
            raw_s = samples.get("samples")
            if raw_s is None:
                return (torch.empty((0, 768, 1344, 3), dtype=torch.float32),)
            v = raw_s

        try:
            images = vae.decode(v)
            images = _standardize_image_tensor(images)
            return (images,)
        except Exception as e:
            raise RuntimeError(f"Video VAE decode failed for shape {getattr(v, 'shape', None)}") from e


class MiniMaxSafeVAEDecodeAudioNode:
    """Safe VAE Audio Decoder that gracefully handles None in Initial Clip Mode.
    
    When samples is None (e.g. initial generation with zero prior context),
    it safely returns an empty AUDIO dict instead of crashing with TypeError.
    When samples is present, it delegates to vae.decode() with full audio fidelity.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "vae": ("VAE",),
            },
            "optional": {
                "samples": ("LATENT",),
            }
        }

    RETURN_TYPES = ("AUDIO",)
    FUNCTION = "decode"
    CATEGORY = "MiniMaxH3/ClipBin"

    def decode(self, vae: Any, samples: Optional[Dict[str, Any]] = None) -> Tuple[Optional[Dict[str, Any]]]:
        if samples is None:
            logger.info("[Safe VAE Decode Audio] No previous audio latent provided (Initial Clip Mode). Passing through None.")
            return (None,)

        _, a = _unpack_latent(samples)
        if a is None:
            # Check if samples itself is an audio latent or dictionary
            raw_s = samples.get("samples")
            if raw_s is not None and hasattr(raw_s, "ndim") and raw_s.ndim <= 4:
                a = raw_s
            else:
                logger.info("[Safe VAE Decode Audio] No audio stream found in latent. Passing through None.")
                return (None,)

        try:
            audio = vae.decode(a)
            if isinstance(audio, torch.Tensor):
                # ComfyUI VAE.decode returns [batch, samples, channels].
                # AUDIO consumers require [batch, channels, samples].
                audio = audio.movedim(-1, 1)
                std = torch.std(audio, dim=[1, 2], keepdim=True) * 5.0
                audio = audio / std.clamp_min(1.0)
                sr = samples.get("sample_rate", getattr(vae, "audio_sample_rate_output",
                                 getattr(vae, "audio_sample_rate", 32000)))
                audio = {"waveform": audio, "sample_rate": int(sr)}
            else:
                audio = _standardize_audio_dict(audio)
            return (audio,)
        except Exception as e:
            raise RuntimeError(f"Audio VAE decode failed for shape {getattr(a, 'shape', None)}") from e


class MiniMaxVideoChunkSlicerNode:
    """Intelligent frame & audio slicer for long video editing (>15s).

    Supports:
    1. Direct on-demand video file streaming (Lazy Chunk Decoding) - Zero RAM/VRAM explosion!
    2. Optional connected images Tensor (100% backward compatible).
    3. Built-in target resolution scaling and force_fps locking.
    4. Emits slice_context so downstream reassembler seamlessly replaces original frames.
    """
    @classmethod
    def INPUT_TYPES(cls):
        video_files = get_input_video_files()
        return {
            "required": {
                "video_file": (video_files,),
                "project_name": ("STRING", {"default": "Video_Edit_Project"}),
                "chunk_length": ("INT", {"default": 124, "min": 16, "max": 2048, "step": 1}),
                "chunk_index": ("INT", {"default": 0, "min": 0, "max": 9999, "step": 1}),
                "target_width": ("INT", {"default": 0, "min": 0, "max": 7680, "step": 8}),
                "target_height": ("INT", {"default": 0, "min": 0, "max": 4320, "step": 8}),
                "force_fps": ("FLOAT", {"default": 24.0, "min": 1.0, "max": 120.0, "step": 0.01}),
                "slice_mode": ([
                    "Auto Chunk Grid (网格切分)",
                    "Custom Range (自由区间)",
                    "Next Unedited (自动下一未编辑段)",
                ],),
                "custom_start_frame": ("INT", {"default": 0, "min": 0, "max": 999999, "step": 1}),
                "custom_end_frame": ("INT", {"default": 124, "min": 1, "max": 999999, "step": 1}),
                "auto_advance": ([
                    "None (手动控制)",
                    "Next Chunk (顺序下一段)",
                    "Next Unedited (跳至下一未编辑)",
                ], {"default": "None (手动控制)"}),
                "prev_ref_frames_count": ("INT", {"default": 16, "min": 1, "max": 128, "step": 1}),
                "first_chunk_ref_mode": ([
                    "Current Chunk First Frame (当前片段首帧)",
                    "Black / Zero Frame (全黑空帧)",
                ], {"default": "Current Chunk First Frame (当前片段首帧)"}),
            },
            "optional": {
                "images": ("IMAGE",),
                "audio": ("AUDIO",),
                "fps": ("FLOAT", {"forceInput": True}),
                "optional_first_frame_ref": ("IMAGE",),
                "optional_prev_chunk_result": ("IMAGE",),
            }
        }

    @classmethod
    def VALIDATE_INPUTS(cls, **kwargs):
        # Allow backward-compatibility with saved workflows that had shifted widget indices
        return True

    RETURN_TYPES = (
        "IMAGE",
        "AUDIO",
        "SLICE_CONTEXT",
        "VHS_VIDEOINFO",
        "IMAGE",
        "FLOAT",
        "INT",
        "STRING",
        "IMAGE",
        "IMAGE",
        "IMAGE",
        "IMAGE",
        "IMAGE",
        "IMAGE",
    )
    RETURN_NAMES = (
        "chunk_images",
        "chunk_audio",
        "slice_context",
        "video_info",
        "timeline_preview",
        "fps",
        "frame_count",
        "slice_info",
        "prev_last_frame",
        "prev_ref_frames",
        "prev_edited_last_frame",
        "prev_orig_last_frame",
        "prev_edited_ref_frames",
        "prev_orig_ref_frames",
    )
    FUNCTION = "slice_chunk"
    CATEGORY = "MiniMaxH3/VideoEdit"

    def slice_chunk(
        self,
        video_file: str,
        project_name: str,
        chunk_length: int = 124,
        chunk_index: int = 0,
        target_width: int = 0,
        target_height: int = 0,
        force_fps: float = 24.0,
        slice_mode: str = "Auto Chunk Grid (网格切分)",
        custom_start_frame: int = 0,
        custom_end_frame: int = 124,
        auto_advance: Any = "None (手动控制)",
        prev_ref_frames_count: int = 16,
        first_chunk_ref_mode: str = "Current Chunk First Frame (当前片段首帧)",
        images: Optional[torch.Tensor] = None,
        audio: Optional[Dict[str, Any]] = None,
        fps: Optional[Any] = None,
        optional_first_frame_ref: Optional[torch.Tensor] = None,
        optional_prev_chunk_result: Optional[torch.Tensor] = None,
        **kwargs,
    ):
        # Sanitize auto_advance (gracefully handle legacy workflows where 24 was stored)
        valid_modes = [
            "None (手动控制)",
            "Next Chunk (顺序下一段)",
            "Next Unedited (跳至下一未编辑)",
        ]
        if not isinstance(auto_advance, str) or auto_advance not in valid_modes:
            auto_advance = "None (手动控制)"

        try:
            prev_ref_frames_count = max(1, int(prev_ref_frames_count))
        except Exception:
            prev_ref_frames_count = 16

        valid_first_modes = [
            "Current Chunk First Frame (当前片段首帧)",
            "Black / Zero Frame (全黑空帧)",
        ]
        if not isinstance(first_chunk_ref_mode, str) or first_chunk_ref_mode not in valid_first_modes:
            first_chunk_ref_mode = "Current Chunk First Frame (当前片段首帧)"

        # Robust FPS resolution
        effective_fps = 24.0
        try:
            if fps is not None and str(fps).strip() != "" and float(fps) > 0:
                effective_fps = float(fps)
            elif force_fps is not None and float(force_fps) > 0:
                effective_fps = float(force_fps)
        except (ValueError, TypeError):
            try:
                effective_fps = float(force_fps) if force_fps else 24.0
            except Exception:
                effective_fps = 24.0
        (
            chunk_imgs,
            chunk_aud,
            ctx,
            info,
            prev_last_frame,
            prev_ref_frames,
            prev_edited_last_frame,
            prev_orig_last_frame,
            prev_edited_ref_frames,
            prev_orig_ref_frames,
        ) = slice_video_and_audio(
            project_name=project_name,
            video_file=video_file,
            images=images,
            audio=audio,
            fps=effective_fps,
            chunk_length=chunk_length,
            chunk_index=chunk_index,
            slice_mode=slice_mode,
            custom_start_frame=custom_start_frame,
            custom_end_frame=custom_end_frame,
            target_width=target_width,
            target_height=target_height,
            prev_ref_frames_count=prev_ref_frames_count,
            first_chunk_ref_mode=first_chunk_ref_mode,
            optional_first_frame_ref=optional_first_frame_ref,
            optional_prev_chunk_result=optional_prev_chunk_result,
            return_ref_frames=True,
        )
        ctx["auto_advance"] = auto_advance
        session = get_or_create_timeline_session(project_name)
        preview_card = render_timeline_indicator_image(
            session=session,
            active_chunk_idx=ctx["chunk_index"],
            active_start=ctx["start_frame"],
            active_end=ctx["end_frame"],
        )

        # Construct standard VHS_VIDEOINFO compatible dictionary for downstream nodes (e.g. VHS_VideoInfo)
        total_f = session.meta.get("total_frames", chunk_imgs.shape[0])
        src_w = session.meta.get("width", chunk_imgs.shape[2])
        src_h = session.meta.get("height", chunk_imgs.shape[1])
        src_fps = float(session.meta.get("fps", effective_fps))
        cur_f = int(chunk_imgs.shape[0])
        cur_w = int(chunk_imgs.shape[2])
        cur_h = int(chunk_imgs.shape[1])

        video_info = {
            "source_fps": src_fps,
            "source_frame_count": total_f,
            "source_duration": round(total_f / src_fps, 4) if src_fps > 0 else 0.0,
            "source_width": src_w,
            "source_height": src_h,
            "loaded_fps": float(effective_fps),
            "loaded_frame_count": cur_f,
            "loaded_duration": round(cur_f / effective_fps, 4) if effective_fps > 0 else 0.0,
            "loaded_width": cur_w,
            "loaded_height": cur_h,
        }

        return (
            chunk_imgs,
            chunk_aud,
            ctx,
            video_info,
            preview_card,
            float(effective_fps),
            cur_f,
            info,
            prev_last_frame,
            prev_ref_frames,
            prev_edited_last_frame,
            prev_orig_last_frame,
            prev_edited_ref_frames,
            prev_orig_ref_frames,
        )


class MiniMaxVideoPatchReassemblerNode:
    """Seamless in-place video patch reassembler & timeline progress tracker.

    Takes generated/edited clip and automatically replaces original frames in master video.
    Detects unedited gaps, supports seam micro-blending, and exports assembled long video.
    """
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "edited_images": ("IMAGE",),
                "slice_context": ("SLICE_CONTEXT",),
                "seam_blend_frames": ("INT", {"default": 2, "min": 0, "max": 16, "step": 1}),
                "output_mode": ([
                    "Full Assembled Video (完整拼装长视频)",
                    "Current Patch Only (仅当前片段)",
                ],),
                "save_to_session": ("BOOLEAN", {"default": True}),
            },
            "optional": {
                "edited_audio": ("AUDIO",),
            }
        }

    RETURN_TYPES = ("IMAGE", "AUDIO", "BOOLEAN", "STRING")
    RETURN_NAMES = ("output_images", "output_audio", "is_all_completed", "timeline_summary")
    FUNCTION = "reassemble_patch"
    CATEGORY = "MiniMaxH3/VideoEdit"

    def reassemble_patch(
        self,
        edited_images: torch.Tensor,
        slice_context: Dict[str, Any],
        seam_blend_frames: int = 2,
        output_mode: str = "Full Assembled Video (完整拼装长视频)",
        save_to_session: bool = True,
        edited_audio: Optional[Dict[str, Any]] = None,
    ):
        project_name = slice_context.get("project_name", "Video_Edit_Project")
        chunk_index = slice_context.get("chunk_index", 0)
        start_frame = slice_context.get("start_frame", 0)
        end_frame = slice_context.get("end_frame", edited_images.shape[0])

        session = get_or_create_timeline_session(project_name)

        if save_to_session:
            session.patch_chunk(
                chunk_index=chunk_index,
                start_frame=start_frame,
                end_frame=end_frame,
                edited_images=edited_images,
                edited_audio=edited_audio,
                seam_blend_frames=seam_blend_frames,
            )

        # Decide output
        if "Current Patch Only" in output_mode:
            out_imgs = edited_images
            raw_aud = edited_audio or (session.master_audio if session.master_audio else None)
        else:
            # Output master assembled frames
            out_imgs = session.get_assembled_frames()
            raw_aud = session.master_audio

        fps = float(session.meta.get("fps", 24.0))
        dur_samples = int(round(out_imgs.shape[0] / fps * 44100))
        out_aud = standardize_audio_dict(raw_aud, default_sr=44100, fallback_samples=dur_samples)

        is_all_done = bool(session.meta.get("is_fully_assembled", False))
        cov_pct = session.meta.get("coverage_ratio", 0.0) * 100
        total_frames = session.meta.get("total_frames", out_imgs.shape[0])
        fps = session.meta.get("fps", 24.0)
        gaps = session.meta.get("gaps", [])

        if is_all_done:
            status_text = (
                f"✅ [全部完成 100%] 项目 '{project_name}' 已完成全部 {total_frames} 帧 "
                f"({total_frames/fps:.2f}s) 替换组装，无任何遗漏！"
            )
        else:
            gap_desc = ", ".join(f"#{g['chunk_index']}({g['start_frame']}~{g['end_frame']}帧)" for g in gaps[:3])
            if len(gaps) > 3:
                gap_desc += f"...共{len(gaps)}处"
            status_text = (
                f"⚡ [拼装进度: {cov_pct:.1f}%] 项目 '{project_name}' 已成功回填 Chunk #{chunk_index}！"
                f"\n⚠️ 尚未完成/漏编辑分段: {gap_desc}"
            )

        return (out_imgs, out_aud, is_all_done, status_text)


# ===========================================================================
# Upgrade 1 — Seamless PrefixStream Video Stitching & Anti-Drift QA Nodes
# ===========================================================================

class H3PrefixSmartSeamSelector:
    """Searches within the PrefixStream overlap window for the minimal-discontinuity frame cut.

    Instead of blindly cutting at exact frame 0 / trim_frames (where H3 unmasked denoising starts),
    evaluates pixel, luminance, Sobel edge, and motion-velocity continuity across candidate overlap
    frames while strictly preserving total AV duration when max_phase_shift == 0.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "curr_images": ("IMAGE", {"tooltip": "当前续写生成的完整画面（裁切前）"}),
                "overlap_frames": ("INT", {
                    "default": 39, "min": 0, "max": 192, "step": 1,
                    "tooltip": "重叠前缀帧数（如 39 或 22）。设为 0 且连接 session/cache_config 时自动识别"
                }),
                "search_radius": ("INT", {
                    "default": 12, "min": 0, "max": 64, "step": 1,
                    "tooltip": "在重叠窗口内向回搜索最佳切点的帧半径"
                }),
                "max_phase_shift": ("INT", {
                    "default": 0, "min": 0, "max": 6, "step": 1,
                    "tooltip": "允许的微相位偏移帧数。0 = 严格锁死音画总时长（MV/对口型必选 0）；>0 适合纯空镜运动对齐"
                }),
                "fps": ("FLOAT", {"default": 24.0, "min": 1.0, "max": 120.0}),
            },
            "optional": {
                "prev_images": ("IMAGE", {"tooltip": "上一段累积画面（首段留空时直接原样透传）"}),
                "prev_audio": ("AUDIO", {"tooltip": "上一段累积音频（同步对齐裁切点）"}),
                "curr_audio": ("AUDIO", {"tooltip": "当前段完整音频（同步对齐裁切点并保留交叉淡化上下文）"}),
                "session": ("MINIMAX_SESSION",),
                "cache_config": ("MINIMAX_CACHE_CONFIG",),
            }
        }

    RETURN_TYPES = ("IMAGE", "IMAGE", "AUDIO", "AUDIO", "INT", "FLOAT", "STRING")
    RETURN_NAMES = (
        "aligned_prev_images",
        "trimmed_curr_images",
        "aligned_prev_audio",
        "trimmed_curr_audio",
        "best_cut_offset",
        "seam_cost",
        "seam_report_json",
    )
    FUNCTION = "select_seam"
    CATEGORY = "MiniMaxH3/SeamlessStitch"

    def select_seam(
        self,
        curr_images: torch.Tensor,
        overlap_frames: int = 39,
        search_radius: int = 12,
        max_phase_shift: int = 0,
        fps: float = 24.0,
        prev_images: Optional[torch.Tensor] = None,
        prev_audio: Optional[Dict[str, Any]] = None,
        curr_audio: Optional[Dict[str, Any]] = None,
        session: Optional[LongVideoSession] = None,
        cache_config: Optional[KVCacheConfig] = None,
    ):
        actual_ov = int(overlap_frames)
        if actual_ov <= 0:
            if session is not None and getattr(session, "is_current_clip_trimmed", False):
                actual_ov = 0
            elif session is not None:
                actual_ov = int(session.last_rolling_frames)
            elif cache_config is not None and prev_images is not None and prev_images.shape[0] > 0:
                actual_ov = int(cache_config.rolling_frames)

        aligned_prev, trimmed_curr, aligned_prev_aud, trimmed_curr_aud, report = smart_select_seam_and_trim(
            prev_images=prev_images,
            curr_images=curr_images,
            overlap_frames=actual_ov,
            search_radius=search_radius,
            max_phase_shift=max_phase_shift,
            prev_audio=prev_audio,
            curr_audio=curr_audio,
            fps=fps,
        )
        if session is not None and report.get("curr_trim_frames", 0) > 0:
            session.is_current_clip_trimmed = True

        if aligned_prev is None:
            aligned_prev = torch.empty((0, curr_images.shape[1], curr_images.shape[2], 3), dtype=curr_images.dtype, device=curr_images.device)

        return (
            aligned_prev,
            trimmed_curr,
            aligned_prev_aud,
            trimmed_curr_aud,
            int(report.get("curr_trim_frames", 0)),
            float(report.get("seam_cost", 0.0)),
            json.dumps(report, ensure_ascii=False, indent=2),
        )


class H3PrefixColorMatchStitch:
    """Eliminates subtle exposure/tint jumps across stitched H3 segments via CIE LAB transfer + micro-blend.

    Combines:
    1. Optional Smart Seam Selection inside the overlap window.
    2. CIE L*a*b* mean/std luma + chroma transfer across the overlap zone, plus optional Shot-1 anchor pull
       so multi-segment chains never accumulate warm/cool/contrast drift.
    3. 2–6 frame Optical-Flow Warp Blend or Cosine S-Curve Alpha Blend.
    4. Zero-click DC-balanced cosine audio crossfade with automatic sample-rate harmonization.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "curr_images": ("IMAGE", {"tooltip": "当前新生成的片段画面"}),
                "overlap_frames": ("INT", {
                    "default": 0, "min": 0, "max": 192, "step": 1,
                    "tooltip": "重叠前缀帧数。设为 0 且连接了 session/cache_config 时自动读取（如 39 帧）"
                }),
                "smart_seam": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "自动在重叠区内挑选像素/运动最平顺的接缝帧对"
                }),
                "color_match_mode": (["lab_mean_std", "luma_only", "none"], {
                    "default": "lab_mean_std",
                    "tooltip": "lab_mean_std 同时消除明暗曝光跳变与冷暖色温漂移；luma_only 仅匹配明度"
                }),
                "color_match_strength": ("FLOAT", {
                    "default": 0.85, "min": 0.0, "max": 1.0, "step": 0.05,
                    "tooltip": "LAB 色彩/明度迁移强度"
                }),
                "anchor_pull_strength": ("FLOAT", {
                    "default": 0.25, "min": 0.0, "max": 1.0, "step": 0.05,
                    "tooltip": "若连接了 anchor_image（定妆/首镜基准图），将色彩统计向首镜拉回的比例，防止多段累积偏色"
                }),
                "grade_decay_frames": ("INT", {
                    "default": 0, "min": 0, "max": 120, "step": 1,
                    "tooltip": "0 = 整段保持色准锁定（推荐，防中段反弹偏色）；>0 = 在指定帧数内平滑衰减回原片色彩"
                }),
                "blend_mode": (["cosine_alpha", "optical_flow_warp", "hard_cut"], {
                    "default": "cosine_alpha",
                    "tooltip": "optical_flow_warp 使用纯 PyTorch 光流双向对齐融合（防运动重影）；cosine_alpha 为余弦平滑混合"
                }),
                "blend_frames": ("INT", {
                    "default": 4, "min": 0, "max": 16, "step": 1,
                    "tooltip": "接缝微融合帧数（推荐 2~6 帧）"
                }),
                "audio_crossfade_ms": ("FLOAT", {
                    "default": 20.0, "min": 0.0, "max": 500.0, "step": 1.0,
                    "tooltip": "零爆音余弦音频交叉淡化窗口（毫秒）"
                }),
                "fps": ("FLOAT", {"default": 24.0, "min": 1.0, "max": 120.0}),
            },
            "optional": {
                "prev_images": ("IMAGE", {"tooltip": "前序已拼接画面（首段留空）"}),
                "anchor_image": ("IMAGE", {"tooltip": "可选：首镜定妆/场景锚点图，用于抑制多段累积色偏"}),
                "prev_audio": ("AUDIO", {"tooltip": "前序已拼接音频（首段留空）"}),
                "curr_audio": ("AUDIO", {"tooltip": "当前段音频"}),
                "session": ("MINIMAX_SESSION",),
                "cache_config": ("MINIMAX_CACHE_CONFIG",),
                "search_radius": ("INT", {"default": 12, "min": 0, "max": 64, "step": 1}),
            }
        }

    RETURN_TYPES = ("IMAGE", "AUDIO", "INT", "STRING")
    RETURN_NAMES = ("stitched_images", "stitched_audio", "total_frames", "stitch_diagnostics_json")
    FUNCTION = "stitch"
    CATEGORY = "MiniMaxH3/SeamlessStitch"

    def stitch(
        self,
        curr_images: torch.Tensor,
        overlap_frames: int = 0,
        smart_seam: bool = True,
        color_match_mode: str = "lab_mean_std",
        color_match_strength: float = 0.85,
        anchor_pull_strength: float = 0.25,
        grade_decay_frames: int = 0,
        blend_mode: str = "cosine_alpha",
        blend_frames: int = 4,
        audio_crossfade_ms: float = 20.0,
        fps: float = 24.0,
        prev_images: Optional[torch.Tensor] = None,
        anchor_image: Optional[torch.Tensor] = None,
        prev_audio: Optional[Dict[str, Any]] = None,
        curr_audio: Optional[Dict[str, Any]] = None,
        session: Optional[LongVideoSession] = None,
        cache_config: Optional[KVCacheConfig] = None,
        search_radius: int = 12,
    ):
        actual_ov = int(overlap_frames)
        if actual_ov <= 0:
            if session is not None and getattr(session, "is_current_clip_trimmed", False):
                actual_ov = 0
            elif session is not None:
                actual_ov = int(session.last_rolling_frames)
            elif cache_config is not None and prev_images is not None and prev_images.shape[0] > 0:
                actual_ov = int(cache_config.rolling_frames)

        stitched_img, stitched_aud, diag = color_match_and_stitch_segments(
            prev_images=prev_images,
            curr_images=curr_images,
            overlap_frames=actual_ov,
            smart_seam=smart_seam,
            search_radius=search_radius,
            max_phase_shift=0,
            color_match_mode=color_match_mode,
            color_match_strength=color_match_strength,
            anchor_image=anchor_image,
            anchor_pull_strength=anchor_pull_strength,
            grade_decay_frames=grade_decay_frames,
            blend_mode=blend_mode,
            blend_frames=blend_frames,
            prev_audio=prev_audio,
            curr_audio=curr_audio,
            audio_crossfade_ms=audio_crossfade_ms,
            fps=fps,
        )
        total_f = int(stitched_img.shape[0]) if stitched_img is not None else 0
        return (stitched_img, stitched_aud, total_f, json.dumps(diag, ensure_ascii=False, indent=2))


class H3PrefixDriftQAProbe:
    """Structured Segment Drift & Seam QA Probe (Gate B in-graph circuit breaker).

    Emits `seam_jump_score`, `sharpness_ratio_vs_anchor`, `color_drift_score`, and `recommend_anchor_reset`
    directly inside ComfyUI so PrefixStream micro-chains auto-warn before A12-style segment 13 collapse!
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "curr_images": ("IMAGE", {"tooltip": "当前生成的片段画面"}),
                "chain_segment_index": ("INT", {
                    "default": 1, "min": 1, "max": 999, "step": 1,
                    "tooltip": "当前处于连续接力链的第几段（推荐 <=3 段为一微链，满 3 段后用新关键帧重置锚点）"
                }),
                "overlap_frames": ("INT", {
                    "default": 0, "min": 0, "max": 192, "step": 1,
                    "tooltip": "未裁切画面开头包含的前缀帧数（已裁切则填 0）"
                }),
                "max_safe_chain_length": ("INT", {
                    "default": 3, "min": 1, "max": 20, "step": 1,
                    "tooltip": "最大安全连续续写段数（默认 3 段，防止 13 段长链崩坏）"
                }),
                "seam_jump_threshold": ("FLOAT", {"default": 0.12, "min": 0.01, "max": 1.0, "step": 0.01}),
                "sharpness_min_ratio": ("FLOAT", {"default": 0.72, "min": 0.10, "max": 1.50, "step": 0.02}),
                "color_drift_threshold": ("FLOAT", {"default": 0.10, "min": 0.01, "max": 1.0, "step": 0.01}),
            },
            "optional": {
                "prev_images": ("IMAGE", {"tooltip": "上一段画面（用于计算接缝跳变 seam_jump_score）"}),
                "anchor_image": ("IMAGE", {"tooltip": "首镜关键帧/定妆锚点图（用于计算清晰度衰减与色偏）"}),
                "curr_audio": ("AUDIO", {"tooltip": "可选音频流（用于检测削波爆音）"}),
            }
        }

    RETURN_TYPES = ("STRING", "FLOAT", "FLOAT", "FLOAT", "BOOLEAN")
    RETURN_NAMES = (
        "qa_report_json",
        "seam_jump_score",
        "sharpness_ratio_vs_anchor",
        "color_drift_score",
        "recommend_anchor_reset",
    )
    FUNCTION = "probe"
    CATEGORY = "MiniMaxH3/QA"

    def probe(
        self,
        curr_images: torch.Tensor,
        chain_segment_index: int = 1,
        overlap_frames: int = 0,
        max_safe_chain_length: int = 3,
        seam_jump_threshold: float = 0.12,
        sharpness_min_ratio: float = 0.72,
        color_drift_threshold: float = 0.10,
        prev_images: Optional[torch.Tensor] = None,
        anchor_image: Optional[torch.Tensor] = None,
        curr_audio: Optional[Dict[str, Any]] = None,
    ):
        res = evaluate_prefix_drift_qa(
            curr_images=curr_images,
            prev_images=prev_images,
            anchor_image=anchor_image,
            curr_audio=curr_audio,
            chain_segment_index=chain_segment_index,
            overlap_frames=overlap_frames,
            max_safe_chain_length=max_safe_chain_length,
            seam_jump_threshold=seam_jump_threshold,
            sharpness_min_ratio=sharpness_min_ratio,
            color_drift_threshold=color_drift_threshold,
        )
        return (
            json.dumps(res, ensure_ascii=False, indent=2),
            float(res["seam_jump_score"]),
            float(res["sharpness_ratio_vs_anchor"]),
            float(res["color_drift_score"]),
            bool(res["recommend_anchor_reset"]),
        )


# ===========================================================================
# Upgrade 2 — H3 150-Char 4-Sentence Prompt Compiler & Contradiction Guard
# ===========================================================================

class H3FourSentencePromptNode:
    """Empirical 91%-obedience 150-char 4-sentence H3 Prompt Compiler + Contradiction Guard.

    Enforces:
    - Sentence 1: Subject + locked visual anchor (<=30 chars), separated from long Setting Bible lore.
    - Sentence 2: Concrete filmable micro-action + scene (rewrites vague emotion words; A30 closed-lip guard).
    - Sentence 3: Camera language strictly <=2 of 3 dimensions (shot_size / camera_movement / lighting).
    - Sentence 4: Unified texture & palette closure.
    - Negative constraints capped at <=2 items + contradiction detection/auto-resolution.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "scene_preset": ([
                    "drama",
                    "mv_broll",
                    "listening_reaction",
                    "talking_head",
                    "commercial",
                ], {
                    "default": "drama",
                    "tooltip": "场景预设：drama(漫剧四句全开) / mv_broll(MV拍场面) / listening_reaction(A30闭唇倾听防张嘴) / talking_head(口播省镜头) / commercial(广告精确入画秒数)"
                }),
                "subject_anchor": ("STRING", {
                    "default": "穿米色风衣的短发女孩",
                    "multiline": False,
                    "tooltip": "第1句 主体+外观锚点（<=30字写死特征，或填设定库编号如 R01）"
                }),
                "action_and_scene": ("STRING", {
                    "default": "天台边缘缓步转身，风吹动发梢，身后是夜色城市灯海",
                    "multiline": True,
                    "tooltip": "第2句 动作+场景（自动将'很纠结/很伤心'转为具体可拍微动作）"
                }),
                "shot_size": ("STRING", {
                    "default": "近景",
                    "tooltip": "第3句维度A：景别（如 特写 / 近景 / 中景 / 全景；留空表示不指定）"
                }),
                "camera_movement": ("STRING", {
                    "default": "Dolly in 缓慢推近",
                    "tooltip": "第3句维度B：专业运镜术语（如 Dolly in / Orbit 环绕 / Tracking 跟拍 / 固定镜头）"
                }),
                "lighting": ("STRING", {
                    "default": "侧逆光",
                    "tooltip": "第3句维度C：光线（自动执行景别/运镜/光线三选二铁律，防指令淹没）"
                }),
                "style_closure": ("STRING", {
                    "default": "胶片颗粒，冷调夜景，情绪克制",
                    "tooltip": "第4句 风格收尾：统一质感与色调"
                }),
                "contradiction_policy": (["auto_resolve", "warn_only"], {
                    "default": "auto_resolve",
                    "tooltip": "检测到'逆光剪影+面部清晰'或'闭唇倾听+开口唱歌'等互杀指令时自动消解或仅告警"
                }),
            },
            "optional": {
                "setting_bible_json": ("STRING", {
                    "default": "",
                    "multiline": True,
                    "tooltip": "【设定与指令分离仓库】JSON格式存放角色小传/场景模板，生成时仅按 ID 提取精简视觉锚点，绝不把800字设定塞进单镜提示词"
                }),
                "character_id": ("STRING", {"default": "", "tooltip": "可选角色锚点编号（如 R01）"}),
                "scene_id": ("STRING", {"default": "", "tooltip": "可选场景锚点编号（如 S03）"}),
                "style_id": ("STRING", {"default": "", "tooltip": "可选风格锚点编号（如 ST01）"}),
                "negative_constraints": ("STRING", {
                    "default": "",
                    "tooltip": "负面约束（自动截断保留最关键 <=2 条，杜绝负面清单污染）"
                }),
                "product_entry_second": ("FLOAT", {
                    "default": 0.0, "min": 0.0, "max": 30.0, "step": 0.1,
                    "tooltip": "仅 commercial 广告模式生效：产品精确入画秒数"
                }),
                "ref_image_tag": ("STRING", {"default": "", "tooltip": "可选多模态图引用前缀（如 <Picture 1>）"}),
                "ref_audio_tag": ("STRING", {"default": "", "tooltip": "可选多模态音频引用前缀（如 <Audio 1>）"}),
            }
        }

    RETURN_TYPES = ("STRING", "STRING", "INT", "BOOLEAN", "STRING")
    RETURN_NAMES = (
        "compiled_prompt",
        "negative_prompt",
        "estimated_obedience",
        "has_contradiction",
        "diagnostics_json",
    )
    FUNCTION = "compile_prompt"
    CATEGORY = "MiniMaxH3/Prompt"

    def compile_prompt(
        self,
        scene_preset: str = "drama",
        subject_anchor: str = "穿米色风衣的短发女孩",
        action_and_scene: str = "天台边缘缓步转身，风吹动发梢，身后是夜色城市灯海",
        shot_size: str = "近景",
        camera_movement: str = "Dolly in 缓慢推近",
        lighting: str = "侧逆光",
        style_closure: str = "胶片颗粒，冷调夜景，情绪克制",
        contradiction_policy: str = "auto_resolve",
        setting_bible_json: str = "",
        character_id: str = "",
        scene_id: str = "",
        style_id: str = "",
        negative_constraints: str = "",
        product_entry_second: float = 0.0,
        ref_image_tag: str = "",
        ref_audio_tag: str = "",
    ):
        result = compile_h3_four_sentence_prompt(
            subject_anchor=subject_anchor,
            action_and_scene=action_and_scene,
            shot_size=shot_size,
            camera_movement=camera_movement,
            lighting=lighting,
            style_closure=style_closure,
            scene_preset=scene_preset,
            setting_bible_json=setting_bible_json,
            character_id=character_id,
            scene_id=scene_id,
            style_id=style_id,
            negative_constraints=negative_constraints,
            contradiction_policy=contradiction_policy,
            product_entry_second=product_entry_second,
            ref_image_tag=ref_image_tag,
            ref_audio_tag=ref_audio_tag,
        )
        return (
            result["compiled_prompt"],
            result["negative_prompt"],
            int(result["estimated_obedience"]),
            bool(result["has_contradiction"]),
            json.dumps(result, ensure_ascii=False, indent=2),
        )


# ===========================================================================
# Upgrade 3 — RTX 4060 Ti (8GB VRAM) Survival Kit & Solo MV Montage Stitcher
# ===========================================================================

class H3VRAMStageGate8GB:
    """8GB VRAM Stage-Gate Cleaner & Sequential Eviction Barrier for RTX 4060 Ti workflows.

    Forces `comfy.model_management.unload_all_models()`, `soft_empty_cache()`, `gc.collect()`,
    and `torch.cuda.empty_cache()` (plus optional CPU offload of passthrough tensors) between heavy stages:
    Seed-VC -> FLUX.3 Klein Keyframe -> LongCat A-Roll -> H3 PrefixStream B-Roll -> Tiled VAE Decode.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "passthrough": (any_type, {"tooltip": "需要跨阶段传递的数据（支持 IMAGE / LATENT / AUDIO / CONDITIONING / 任意类型）"}),
                "stage_name": ("STRING", {
                    "default": "Post-Sampler -> Pre-TiledVAE (8GB Evict)",
                    "tooltip": "当前显存清理关卡名称（写入日志与诊断报告）"
                }),
                "unload_models": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "强制卸载 ComfyUI 驻留模型（防止 Qwen3VL / FLUX / H3 同时占满 8GB 显存）"
                }),
                "empty_cuda_cache": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "清空 PyTorch CUDA 缓存块与 Python GC 循环引用"
                }),
                "offload_passthrough_to_cpu": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "将透传的 IMAGE/LATENT/AUDIO 张量移至系统内存，确保下一阶段加载模型前 GPU 显存处于最低基线"
                }),
                "vram_budget_mb": ("INT", {
                    "default": 8188, "min": 2048, "max": 81920, "step": 1,
                    "tooltip": "目标显卡总显存预算（RTX 4060 Ti 8GB 实测为 8188 MiB）"
                }),
            },
            "optional": {
                "wait_for": (any_type, {"tooltip": "可选执行顺序栅栏：连接上一阶段末端输出，确保上一阶段完全结束后再清理显存"}),
                "always_execute": ("BOOLEAN", {"default": True}),
            }
        }

    RETURN_TYPES = (any_type, "STRING", "FLOAT")
    RETURN_NAMES = ("passthrough_out", "vram_report_json", "free_vram_mb")
    FUNCTION = "clean_stage"
    CATEGORY = "MiniMaxH3/VRAM_8GB"

    @classmethod
    def IS_CHANGED(cls, always_execute: bool = True, **kwargs):
        if always_execute:
            return float("nan")
        return False

    def clean_stage(
        self,
        passthrough: Any,
        stage_name: str = "Post-Sampler -> Pre-TiledVAE (8GB Evict)",
        unload_models: bool = True,
        empty_cuda_cache: bool = True,
        offload_passthrough_to_cpu: bool = True,
        vram_budget_mb: int = 8188,
        wait_for: Any = None,
        always_execute: bool = True,
    ):
        _ = wait_for
        _ = always_execute
        out_val, report, free_mb = run_vram_stage_gate(
            passthrough=passthrough,
            stage_name=stage_name,
            unload_models=unload_models,
            empty_cuda_cache=empty_cuda_cache,
            offload_passthrough_to_cpu=offload_passthrough_to_cpu,
            vram_budget_mb=vram_budget_mb,
        )
        return (out_val, json.dumps(report, ensure_ascii=False, indent=2), float(free_mb))


class H3LongCatMVMontageStitcher:
    """Dual-Engine Solo MV Timeline Stitcher: LongCat Singing A-Roll ('唱整段') + H3 B-Roll ('拍场面').

    Locks all visual cuts directly to a single authoritative `master_audio` song tensor (`48kHz`/`44.1kHz`)
    with automatic FPS conversion (e.g. LongCat 30fps -> Master 24fps), aspect-safe spatial framing,
    cross-engine CIE LAB color harmonization, and 0.0ms cumulative audio drift!
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "master_audio": ("AUDIO", {"tooltip": "唯一权威主音轨（如 48kHz/44.1kHz 原曲或 Seed-VC 歌声转换母带），全程零漂移锁死"}),
                "fps": ("FLOAT", {"default": 24.0, "min": 1.0, "max": 120.0, "step": 0.01}),
                "target_width": ("INT", {"default": 0, "min": 0, "max": 4096, "step": 8, "tooltip": "0 = 自动跟随首段接入视频宽度"}),
                "target_height": ("INT", {"default": 0, "min": 0, "max": 4096, "step": 8, "tooltip": "0 = 自动跟随首段接入视频高度"}),
                "fit_mode": (["cover_crop", "letterbox", "stretch"], {
                    "default": "cover_crop",
                    "tooltip": "双引擎画幅比例不完全一致时的对齐策略：cover_crop 等比居中裁切无黑边"
                }),
                "transition_frames": ("INT", {
                    "default": 3, "min": 0, "max": 16, "step": 1,
                    "tooltip": "A/B 轨切换处的微溶解过渡帧数（0 = 卡鼓点硬切；2~4 = 柔和过渡且绝不改变总帧数）"
                }),
                "color_harmonize_broll_to_aroll": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "自动将 H3 场面镜头与 LongCat 演唱主镜头的 LAB 色温/明暗拉齐，消除双模型切换违和感"
                }),
                "timeline_spec_json": ("STRING", {
                    "default": "auto_alternating",
                    "multiline": True,
                    "tooltip": "填 'auto_alternating' 自动交替编排已接入的 A-Roll / B-Roll；或填入精确时间轴 JSON 数组 [{'role':'aroll','clip_slot':'aroll_1','start_sec':0,'end_sec':4}]"
                }),
            },
            "optional": {
                "aroll_images_1": ("IMAGE", {"tooltip": "LongCat 唱整段 A-Roll 镜头 1"}),
                "broll_images_1": ("IMAGE", {"tooltip": "H3 PrefixStream 拍场面 B-Roll 镜头 1"}),
                "aroll_images_2": ("IMAGE", {"tooltip": "LongCat 唱整段 A-Roll 镜头 2"}),
                "broll_images_2": ("IMAGE", {"tooltip": "H3 PrefixStream 拍场面 B-Roll 镜头 2"}),
                "aroll_images_3": ("IMAGE", {"tooltip": "LongCat 唱整段 A-Roll 镜头 3"}),
                "broll_images_3": ("IMAGE", {"tooltip": "H3 PrefixStream 拍场面 B-Roll 镜头 3"}),
                "aroll_images_4": ("IMAGE", {"tooltip": "LongCat 唱整段 A-Roll 镜头 4"}),
                "broll_images_4": ("IMAGE", {"tooltip": "H3 PrefixStream 拍场面 B-Roll 镜头 4"}),
                "anchor_style_image": ("IMAGE", {"tooltip": "可选全局色彩定妆基准图（如 FLUX 首帧关键图）"}),
                "aroll_source_fps": ("FLOAT", {"default": 30.0, "min": 1.0, "max": 120.0, "tooltip": "LongCat A-Roll 原始帧率（LongCat 默认 30fps，自动重采样至主时间轴 fps）"}),
                "broll_source_fps": ("FLOAT", {"default": 24.0, "min": 1.0, "max": 120.0, "tooltip": "H3 B-Roll 原始帧率（H3 默认 24fps）"}),
                "target_sample_rate": ("INT", {"default": 0, "min": 0, "max": 96000, "tooltip": "0 = 保持母带原始采样率（如 48000 或 44100）"}),
                "master_start_sec": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 3600.0, "step": 0.01}),
                "default_segment_sec": ("FLOAT", {"default": 4.0, "min": 0.5, "max": 60.0, "step": 0.1}),
                "color_harmonize_strength": ("FLOAT", {"default": 0.65, "min": 0.0, "max": 1.0, "step": 0.05}),
            }
        }

    RETURN_TYPES = ("IMAGE", "AUDIO", "INT", "STRING")
    RETURN_NAMES = ("mv_images", "mv_audio", "total_frames", "montage_report_json")
    FUNCTION = "assemble_mv"
    CATEGORY = "MiniMaxH3/MV_Studio"

    def assemble_mv(
        self,
        master_audio: Dict[str, Any],
        fps: float = 24.0,
        target_width: int = 0,
        target_height: int = 0,
        fit_mode: str = "cover_crop",
        transition_frames: int = 3,
        color_harmonize_broll_to_aroll: bool = True,
        timeline_spec_json: str = "auto_alternating",
        aroll_images_1: Optional[torch.Tensor] = None,
        broll_images_1: Optional[torch.Tensor] = None,
        aroll_images_2: Optional[torch.Tensor] = None,
        broll_images_2: Optional[torch.Tensor] = None,
        aroll_images_3: Optional[torch.Tensor] = None,
        broll_images_3: Optional[torch.Tensor] = None,
        aroll_images_4: Optional[torch.Tensor] = None,
        broll_images_4: Optional[torch.Tensor] = None,
        anchor_style_image: Optional[torch.Tensor] = None,
        aroll_source_fps: float = 30.0,
        broll_source_fps: float = 24.0,
        target_sample_rate: int = 0,
        master_start_sec: float = 0.0,
        default_segment_sec: float = 4.0,
        color_harmonize_strength: float = 0.65,
    ):
        slots = {
            "aroll_1": aroll_images_1,
            "broll_1": broll_images_1,
            "aroll_2": aroll_images_2,
            "broll_2": broll_images_2,
            "aroll_3": aroll_images_3,
            "broll_3": broll_images_3,
            "aroll_4": aroll_images_4,
            "broll_4": broll_images_4,
        }
        mv_img, mv_aud, total_f, report = stitch_dual_engine_mv_montage(
            master_audio=master_audio,
            fps=fps,
            target_width=target_width,
            target_height=target_height,
            fit_mode=fit_mode,
            transition_frames=transition_frames,
            color_harmonize_broll_to_aroll=color_harmonize_broll_to_aroll,
            color_harmonize_strength=color_harmonize_strength,
            timeline_spec_json=timeline_spec_json,
            default_segment_sec=default_segment_sec,
            aroll_source_fps=aroll_source_fps,
            broll_source_fps=broll_source_fps,
            target_sample_rate=target_sample_rate,
            master_start_sec=master_start_sec,
            anchor_style_image=anchor_style_image,
            clip_slots=slots,
        )
        return (mv_img, mv_aud, int(total_f), json.dumps(report, ensure_ascii=False, indent=2))


NODE_CLASS_MAPPINGS = {
    "MiniMaxDiskVideoStream": MiniMaxDiskVideoStreamNode,
    "MiniMaxPrefixCacheConfig": MiniMaxPrefixCacheConfigNode,
    "MiniMaxPrefixCacheApplier": MiniMaxPrefixCacheApplierNode,
    "MiniMaxTrimPrefix": MiniMaxTrimPrefixLatentNode,
    "MiniMaxTrimPrefixLatent": MiniMaxTrimPrefixLatentNode,
    "MiniMaxLongVideoStitcher": MiniMaxLongVideoStitcherNode,
    "MiniMaxCacheMonitor": MiniMaxCacheMonitorNode,
    "MiniMaxSaveLatent": MiniMaxSaveLatentNode,
    "MiniMaxLoadLatent": MiniMaxLoadLatentNode,
    "MiniMaxClipBinSaver": MiniMaxClipBinSaverNode,
    "MiniMaxClipBinPicker": MiniMaxClipBinPickerNode,
    "MiniMaxClipBinTreePicker": MiniMaxClipBinTreePickerNode,
    "MiniMaxSafeVAEDecode": MiniMaxSafeVAEDecodeNode,
    "MiniMaxSafeVAEDecodeAudio": MiniMaxSafeVAEDecodeAudioNode,
    "MiniMaxVideoChunkSlicer": MiniMaxVideoChunkSlicerNode,
    "MiniMaxVideoPatchReassembler": MiniMaxVideoPatchReassemblerNode,
    "H3PrefixSmartSeamSelector": H3PrefixSmartSeamSelector,
    "H3PrefixColorMatchStitch": H3PrefixColorMatchStitch,
    "H3PrefixDriftQAProbe": H3PrefixDriftQAProbe,
    "H3FourSentencePromptNode": H3FourSentencePromptNode,
    "H3VRAMStageGate8GB": H3VRAMStageGate8GB,
    "H3LongCatMVMontageStitcher": H3LongCatMVMontageStitcher,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "MiniMaxDiskVideoStream": "MiniMax H3 Disk Video Stream",
    "MiniMaxPrefixCacheConfig": "MiniMax H3 Continuation Config",
    "MiniMaxPrefixCacheApplier": "MiniMax H3 Continuation Applier",
    "MiniMaxTrimPrefix": "MiniMax H3 Trim Prefix (AV Master, Zero Flicker)",
    "MiniMaxTrimPrefixLatent": "MiniMax H3 Trim Prefix Latent (AV Master)",
    "MiniMaxLongVideoStitcher": "MiniMax H3 Long Video Stitcher (AV Seamless)",
    "MiniMaxCacheMonitor": "MiniMax H3 Cache Telemetry Monitor",
    "MiniMaxSaveLatent": "MiniMax H3 Save AV Latent (Standalone)",
    "MiniMaxLoadLatent": "MiniMax H3 Load AV Latent (Standalone)",
    "MiniMaxClipBinSaver": "MiniMax H3 Clip Bin Saver (Media Pool)",
    "MiniMaxClipBinPicker": "MiniMax H3 Clip Bin Picker (Gallery Loader)",
    "MiniMaxClipBinTreePicker": "MiniMax H3 Clip Bin Tree Picker (Lineage Graph)",
    "MiniMaxSafeVAEDecode": "MiniMax H3 Safe VAE Decode (Video)",
    "MiniMaxSafeVAEDecodeAudio": "MiniMax H3 Safe VAE Decode (Audio)",
    "MiniMaxVideoChunkSlicer": "🎬 MiniMax Video Chunk Slicer (长视频智能切片器)",
    "MiniMaxVideoPatchReassembler": "🧩 MiniMax Video Patch Reassembler (视频回填缝合器)",
    "H3PrefixSmartSeamSelector": "✂️ H3 Prefix Smart Seam Selector (智能最小跳变寻缝)",
    "H3PrefixColorMatchStitch": "🎨 H3 Prefix LAB Color-Match & Flow Stitcher (无痕色彩/光流缝合)",
    "H3PrefixDriftQAProbe": "🔬 H3 Prefix Drift & Seam QA Probe (长链劣化熔断探针)",
    "H3FourSentencePromptNode": "✍️ H3 150-Char 4-Sentence Prompt Compiler (91%服从度四句编译器)",
    "H3VRAMStageGate8GB": "🛡️ H3 RTX 4060 Ti 8GB VRAM Stage-Gate Cleaner (跨阶段显存清道夫)",
    "H3LongCatMVMontageStitcher": "🎵 H3 × LongCat Dual-Engine MV Montage Stitcher (音画零漂移母带混剪)",
}


