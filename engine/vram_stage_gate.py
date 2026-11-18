"""RTX 4060 Ti (8GB VRAM) Stage-Gate Cleaner & Sequential Eviction Barrier.

Why this exists (validated on `stone-winpc` RTX 4060 Ti 8188 MiB WDDM):
- Windows 11 desktop + WDDM reserves ~1.0–1.5 GiB baseline; H3 Turbo INT8 608x352 peaks at ~7343–7373 MiB.
- Running multi-stage Solo MV graphs (Seed-VC -> FLUX.3 Klein INT8 -> Qwen3-VL-4B -> H3 PrefixStream / LongCat -> Tiled VAE)
  causes immediate CUDA OOM if any prior model or intermediate GPU tensor stays resident across stage boundaries.

Capabilities:
1. Execution-order barrier (`passthrough` + optional `wait_for` dependency trigger).
2. Calls `comfy.model_management.unload_all_models()`, `soft_empty_cache()`, `gc.collect()`, and `torch.cuda.empty_cache()`
   with 100% safe CPU/MPS fallback when running without an NVIDIA GPU or standalone in unit tests.
3. Optional CPU tensor offload (`offload_passthrough_to_cpu`) for `IMAGE`, `LATENT` (incl. NestedTensor), and `AUDIO`
   so stage handoff payloads occupy 0 MB of GPU VRAM while the next heavy model loads.
"""

from typing import Any, Dict, Optional, Tuple
import gc
import logging
import torch

logger = logging.getLogger("minimax_prefix_stream")


def _offload_value_to_cpu(obj: Any) -> Any:
    """Recursively detach and move PyTorch tensors (including NestedTensor / dicts / tuples) to CPU RAM."""
    if obj is None:
        return None
    if isinstance(obj, torch.Tensor):
        if obj.device.type == "cpu":
            return obj
        return obj.detach().cpu()
    # ComfyUI NestedTensor support
    if hasattr(obj, "unbind") and hasattr(obj, "to"):
        try:
            return obj.to("cpu")
        except Exception:
            parts = [_offload_value_to_cpu(t) for t in obj.unbind()]
            try:
                import comfy.nested_tensor
                return comfy.nested_tensor.NestedTensor(parts)
            except Exception:
                return tuple(parts)
    if hasattr(obj, "tensors") and isinstance(getattr(obj, "tensors"), (list, tuple)):
        parts = [_offload_value_to_cpu(t) for t in obj.tensors]
        try:
            return type(obj)(parts)
        except Exception:
            return tuple(parts)
    if isinstance(obj, dict):
        return {k: _offload_value_to_cpu(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_offload_value_to_cpu(v) for v in obj]
    if isinstance(obj, tuple):
        return tuple(_offload_value_to_cpu(v) for v in obj)
    return obj


def _probe_vram_mb(vram_budget_mb: int = 8188) -> Dict[str, float]:
    """Query CUDA VRAM metrics in MiB with graceful fallback on CPU/MPS machines."""
    mib = float(1024 * 1024)
    if torch.cuda.is_available():
        try:
            alloc = float(torch.cuda.memory_allocated()) / mib
            reserved = float(torch.cuda.memory_reserved()) / mib
            free_bytes, total_bytes = torch.cuda.mem_get_info()
            free_mb = float(free_bytes) / mib
            total_mb = float(total_bytes) / mib
            return {
                "backend": "cuda",
                "allocated_mb": round(alloc, 2),
                "reserved_mb": round(reserved, 2),
                "free_mb": round(free_mb, 2),
                "total_mb": round(total_mb, 2),
            }
        except Exception:
            pass
    if hasattr(torch, "mps") and hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        try:
            alloc = float(torch.mps.current_allocated_memory()) / mib if hasattr(torch.mps, "current_allocated_memory") else 0.0
        except Exception:
            alloc = 0.0
        total_mb = float(vram_budget_mb)
        return {
            "backend": "mps_simulated_8gb",
            "allocated_mb": round(alloc, 2),
            "reserved_mb": round(alloc, 2),
            "free_mb": round(max(0.0, total_mb - alloc), 2),
            "total_mb": round(total_mb, 2),
        }
    total_mb = float(vram_budget_mb)
    return {
        "backend": "cpu_fallback",
        "allocated_mb": 0.0,
        "reserved_mb": 0.0,
        "free_mb": round(total_mb, 2),
        "total_mb": round(total_mb, 2),
    }


def run_vram_stage_gate(
    passthrough: Any,
    stage_name: str = "Stage Boundary -> Evict Resident Models",
    unload_models: bool = True,
    empty_cuda_cache: bool = True,
    offload_passthrough_to_cpu: bool = True,
    vram_budget_mb: int = 8188,
    min_required_free_mb: float = 1500.0,
) -> Tuple[Any, Dict[str, Any], float]:
    """Execute full model eviction & VRAM cache purge across a ComfyUI stage gate."""
    before = _probe_vram_mb(vram_budget_mb=vram_budget_mb)
    out_payload = _offload_value_to_cpu(passthrough) if offload_passthrough_to_cpu else passthrough

    comfy_unloaded = False
    comfy_cache_cleared = False
    try:
        import comfy.model_management as mm  # type: ignore
        if unload_models and hasattr(mm, "unload_all_models"):
            mm.unload_all_models()
            comfy_unloaded = True
        if unload_models and hasattr(mm, "cleanup_models_gc"):
            mm.cleanup_models_gc()
        elif unload_models and hasattr(mm, "cleanup_models"):
            mm.cleanup_models()
        if empty_cuda_cache and hasattr(mm, "soft_empty_cache"):
            try:
                mm.soft_empty_cache(True)
            except TypeError:
                mm.soft_empty_cache()
            comfy_cache_cleared = True
    except ImportError:
        pass
    except Exception as exc:
        logger.warning("[VRAM Stage-Gate 8GB] comfy.model_management cleanup warning: %s", exc)

    gc_collected = int(gc.collect())

    if empty_cuda_cache:
        if torch.cuda.is_available():
            try:
                torch.cuda.empty_cache()
                if hasattr(torch.cuda, "ipc_collect"):
                    torch.cuda.ipc_collect()
            except Exception:
                pass
        elif hasattr(torch, "mps") and hasattr(torch.mps, "empty_cache") and hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            try:
                torch.mps.empty_cache()
            except Exception:
                pass

    after = _probe_vram_mb(vram_budget_mb=vram_budget_mb)
    reclaimed_alloc = round(max(0.0, before["allocated_mb"] - after["allocated_mb"]), 2)
    reclaimed_reserved = round(max(0.0, before["reserved_mb"] - after["reserved_mb"]), 2)
    free_mb = float(after["free_mb"])
    safe_for_next = free_mb >= float(min_required_free_mb)

    report: Dict[str, Any] = {
        "stage_name": stage_name,
        "backend": after["backend"],
        "vram_budget_mb": int(vram_budget_mb),
        "before_allocated_mb": before["allocated_mb"],
        "after_allocated_mb": after["allocated_mb"],
        "before_reserved_mb": before["reserved_mb"],
        "after_reserved_mb": after["reserved_mb"],
        "reclaimed_allocated_mb": reclaimed_alloc,
        "reclaimed_reserved_mb": reclaimed_reserved,
        "free_vram_mb": free_mb,
        "total_vram_mb": after["total_mb"],
        "comfy_models_unloaded": comfy_unloaded,
        "comfy_cache_cleared": comfy_cache_cleared,
        "passthrough_offloaded_to_cpu": bool(offload_passthrough_to_cpu),
        "gc_objects_collected": gc_collected,
        "safe_for_next_stage": safe_for_next,
        "recommended_4060ti_profile": {
            "cli_flags": "--reserve-vram 0.5 --lowvram",
            "h3_draft_res": "608x352 or 752x416 (124 frames, 4 steps INT8 Turbo)",
            "max_prefix_chain_segments": 3,
            "vae_decode": "VAEDecodeTiled",
        },
    }

    logger.info(
        "[VRAM Stage-Gate 8GB] %s | backend=%s | alloc %.1f->%.1f MiB | free=%.1f/%.1f MiB | safe=%s",
        stage_name,
        after["backend"],
        before["allocated_mb"],
        after["allocated_mb"],
        free_mb,
        after["total_mb"],
        safe_for_next,
    )
    return out_payload, report, free_mb
