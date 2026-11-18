"""Comprehensive unit tests for ComfyUI-MiniMaxH3-PrefixStream Global Upgrade V2:
- Upgrade 1: H3PrefixSmartSeamSelector, H3PrefixColorMatchStitch (CIE LAB + Optical Flow + Zero-Click Audio), H3PrefixDriftQAProbe
- Upgrade 2: H3FourSentencePromptNode (150-char 4-sentence sweet spot, Setting Bible separation, A30 closed-lip guard, Contradiction Firewall)
- Upgrade 3: H3VRAMStageGate8GB (CPU/ComfyUI fallback, NestedTensor offload), H3LongCatMVMontageStitcher (30fps LongCat A-Roll + 24fps H3 B-Roll locked to 48kHz/44.1kHz master audio)
"""

import json
import math
import sys
import types
import unittest
from unittest.mock import MagicMock, patch

import torch

from nodes import (
    H3FourSentencePromptNode,
    H3LongCatMVMontageStitcher,
    H3PrefixColorMatchStitch,
    H3PrefixDriftQAProbe,
    H3PrefixSmartSeamSelector,
    H3VRAMStageGate8GB,
    MiniMaxLongVideoStitcherNode,
    MiniMaxPrefixCacheConfigNode,
    NODE_CLASS_MAPPINGS,
    NODE_DISPLAY_NAME_MAPPINGS,
)
from pipeline.long_video_director import LongVideoSession
from pipeline.smart_seam_color_stitch import lab_to_rgb, rgb_to_lab
from engine.prompt_compiler import score_h3_prompt_obedience


def _make_panning_clip(num_frames: int, h: int = 48, w: int = 64, start_phase: int = 0) -> torch.Tensor:
    """Generate a smooth deterministic textured panning clip [F, H, W, 3] in [0, 1]."""
    yy = torch.linspace(0.0, 1.0, h).view(1, h, 1, 1)
    xx = torch.linspace(0.0, 1.0, w).view(1, 1, w, 1)
    tt = torch.arange(start_phase, start_phase + num_frames, dtype=torch.float32).view(num_frames, 1, 1, 1)
    # Textured moving bars + fine detail + color gradient
    pan_x = xx + 0.012 * tt
    fine = 0.08 * torch.sin(2.0 * math.pi * 14.0 * pan_x) * torch.cos(2.0 * math.pi * 14.0 * yy)
    r = 0.45 + 0.22 * torch.sin(2.0 * math.pi * (3.0 * pan_x + yy)) + fine
    g = 0.45 + 0.22 * torch.cos(2.0 * math.pi * (2.0 * pan_x - 2.0 * yy)) + fine
    b = 0.50 + 0.18 * torch.sin(2.0 * math.pi * (4.0 * yy + 0.5 * pan_x)) + fine
    return torch.cat([r, g, b], dim=-1).clamp(0.0, 1.0)


def _make_sine_audio(frames: int, fps: float = 24.0, sr: int = 32000, freq: float = 440.0, start_frame: int = 0):
    n_samples = int(round(frames / fps * sr))
    t0 = start_frame / fps
    t = t0 + torch.arange(n_samples, dtype=torch.float32) / float(sr)
    wave = torch.stack([0.4 * torch.sin(2.0 * math.pi * freq * t), 0.4 * torch.cos(2.0 * math.pi * freq * t)], dim=0).unsqueeze(0)
    return {"waveform": wave, "sample_rate": sr}


class TestNodeRegistry(unittest.TestCase):
    def test_all_six_new_nodes_registered_and_valid(self):
        expected = [
            "H3PrefixSmartSeamSelector",
            "H3PrefixColorMatchStitch",
            "H3PrefixDriftQAProbe",
            "H3FourSentencePromptNode",
            "H3VRAMStageGate8GB",
            "H3LongCatMVMontageStitcher",
        ]
        for key in expected:
            self.assertIn(key, NODE_CLASS_MAPPINGS)
            self.assertIn(key, NODE_DISPLAY_NAME_MAPPINGS)
            cls = NODE_CLASS_MAPPINGS[key]
            schema = cls.INPUT_TYPES()
            self.assertIn("required", schema)
            self.assertTrue(hasattr(cls, cls.FUNCTION))
            self.assertEqual(len(cls.RETURN_TYPES), len(cls.RETURN_NAMES))


class TestSmartSeamAndColorStitch(unittest.TestCase):
    def test_cie_lab_round_trip_accuracy(self):
        grid = torch.linspace(0.0, 1.0, 16)
        rr, gg, bb = torch.meshgrid(grid, grid, grid, indexing="ij")
        rgb = torch.stack([rr, gg, bb], dim=-1)
        recovered = lab_to_rgb(rgb_to_lab(rgb))
        max_err = float((rgb - recovered).abs().max().item())
        self.assertLess(max_err, 1e-4)

    def test_smart_seam_avoids_boundary_pop_and_preserves_exact_duration(self):
        # Prev: frames 0..59; Curr overlaps 22 frames (frames 38..97)
        prev = _make_panning_clip(60, start_phase=0)
        curr = _make_panning_clip(60, start_phase=38).clone()

        # Inject a severe denoiser flash/pop right at the nominal boundary curr[21..22] and prev[59]
        curr[22] = (curr[22] + 0.45).clamp(0.0, 1.0)
        prev[59] = (prev[59] * 0.35).clamp(0.0, 1.0)

        prev_aud = _make_sine_audio(60, fps=24.0, sr=32000, start_frame=0)
        curr_aud = _make_sine_audio(60, fps=24.0, sr=32000, start_frame=38)

        selector = H3PrefixSmartSeamSelector()
        al_prev, tr_curr, al_paud, tr_caud, best_cut, cost, report_json = selector.select_seam(
            curr_images=curr,
            overlap_frames=22,
            search_radius=12,
            max_phase_shift=0,
            fps=24.0,
            prev_images=prev,
            prev_audio=prev_aud,
            curr_audio=curr_aud,
        )
        report = json.loads(report_json)
        # Must NOT choose the corrupted nominal boundary k=22
        self.assertLess(best_cut, 22)
        self.assertGreater(report["improvement_pct"], 50.0)
        # Strict timeline invariance: total frames == 60 + 60 - 22 == 98!
        self.assertEqual(al_prev.shape[0] + tr_curr.shape[0], 98)
        expected_samples = int(round(98 / 24.0 * 32000))
        actual_samples = al_paud["waveform"].shape[-1] + tr_caud["waveform"].shape[-1]
        self.assertAlmostEqual(actual_samples, expected_samples, delta=2)

    def test_smart_seam_sets_session_trimmed_guard(self):
        cfg = MiniMaxPrefixCacheConfigNode().create_config(continuation_frames="22")[0]
        sess = LongVideoSession(cfg)
        sess.last_rolling_frames = 22
        prev = _make_panning_clip(40, start_phase=0)
        curr = _make_panning_clip(40, start_phase=18)

        selector = H3PrefixSmartSeamSelector()
        al_prev, tr_curr, _, _, _, _, _ = selector.select_seam(
            curr_images=curr,
            overlap_frames=0,
            prev_images=prev,
            session=sess,
        )
        self.assertTrue(sess.is_current_clip_trimmed)
        # Downstream stitcher should see session.is_current_clip_trimmed and NOT double-trim!
        stitcher = H3PrefixColorMatchStitch()
        stitched, _, total_f, _ = stitcher.stitch(
            curr_images=tr_curr,
            overlap_frames=0,
            prev_images=al_prev,
            session=sess,
            blend_frames=0,
        )
        self.assertEqual(total_f, 40 + 40 - 22)
        self.assertEqual(stitched.shape[0], total_f)

    def test_color_match_eliminates_warm_tint_and_exposure_jump(self):
        anchor = _make_panning_clip(8, start_phase=0)
        prev = _make_panning_clip(48, start_phase=0)
        curr_raw = _make_panning_clip(48, start_phase=26)

        # Simulate strong cool/dim tint shift on continuation clip
        curr_tinted = curr_raw.clone()
        curr_tinted[..., 0] = (curr_tinted[..., 0] * 0.78).clamp(0.0, 1.0)  # drop red
        curr_tinted[..., 2] = (curr_tinted[..., 2] * 1.18 + 0.05).clamp(0.0, 1.0)  # boost blue

        node = H3PrefixColorMatchStitch()
        stitched_unmatched, _, _, _ = node.stitch(
            curr_images=curr_tinted,
            prev_images=prev,
            overlap_frames=22,
            smart_seam=False,
            color_match_mode="none",
            blend_mode="hard_cut",
            blend_frames=0,
        )
        stitched_matched, _, total_f, diag_json = node.stitch(
            curr_images=curr_tinted,
            prev_images=prev,
            anchor_image=anchor,
            overlap_frames=22,
            smart_seam=True,
            color_match_mode="lab_mean_std",
            color_match_strength=0.95,
            anchor_pull_strength=0.30,
            blend_mode="optical_flow_warp",
            blend_frames=4,
        )
        self.assertEqual(total_f, 48 + 48 - 22)
        self.assertTrue(torch.isfinite(stitched_matched).all())
        self.assertGreaterEqual(float(stitched_matched.min().item()), 0.0)
        self.assertLessEqual(float(stitched_matched.max().item()), 1.0)

        # Compare tail LAB distance vs anchor: matched must be much closer to original palette than unmatched
        gt_tail = rgb_to_lab(_make_panning_clip(26, start_phase=48)).mean(dim=(0, 1, 2))
        err_unmatched = float((rgb_to_lab(stitched_unmatched[-20:]).mean(dim=(0, 1, 2)) - gt_tail).norm().item())
        err_matched = float((rgb_to_lab(stitched_matched[-20:]).mean(dim=(0, 1, 2)) - gt_tail).norm().item())
        self.assertLess(err_matched, err_unmatched * 0.5)
        diag = json.loads(diag_json)
        self.assertTrue(diag["color_match"]["applied"])
        self.assertEqual(diag["blend_mode"], "optical_flow_warp")

    def test_cross_sample_rate_zero_click_audio_stitch(self):
        prev = _make_panning_clip(48, start_phase=0)
        curr = _make_panning_clip(48, start_phase=26)
        # Mismatched sample rates: prev is 48kHz stereo, curr is 32kHz mono with DC offset!
        prev_aud = _make_sine_audio(48, fps=24.0, sr=48000, start_frame=0)
        curr_aud = _make_sine_audio(48, fps=24.0, sr=32000, start_frame=26)
        curr_aud["waveform"] = curr_aud["waveform"][:, :1, :] + 0.03

        node = H3PrefixColorMatchStitch()
        _, stitched_aud, total_f, _ = node.stitch(
            curr_images=curr,
            prev_images=prev,
            prev_audio=prev_aud,
            curr_audio=curr_aud,
            overlap_frames=22,
            audio_crossfade_ms=25.0,
            fps=24.0,
        )
        self.assertEqual(stitched_aud["sample_rate"], 48000)
        self.assertEqual(stitched_aud["waveform"].shape[1], 2)
        expected_samples = int(round(total_f / 24.0 * 48000))
        self.assertEqual(stitched_aud["waveform"].shape[-1], expected_samples)

    def test_legacy_stitcher_smart_seam_toggle_works(self):
        prev = _make_panning_clip(40, start_phase=0)
        curr = _make_panning_clip(40, start_phase=18)
        st = MiniMaxLongVideoStitcherNode()
        imgs, _, total_f = st.stitch(
            trim_frames=22,
            prev_images=prev,
            curr_images=curr,
            smart_seam_search=True,
        )
        self.assertEqual(total_f, 58)
        self.assertEqual(imgs.shape[0], 58)


class TestDriftQAProbe(unittest.TestCase):
    def test_clean_segment_passes_and_segment_3_recommends_anchor_reset(self):
        anchor = _make_panning_clip(4, h=64, w=64, start_phase=0)
        prev = _make_panning_clip(40, h=64, w=64, start_phase=0)
        curr = _make_panning_clip(40, h=64, w=64, start_phase=18)

        probe = H3PrefixDriftQAProbe()
        rep1_json, seam1, sharp1, col1, reset1 = probe.probe(
            curr_images=curr,
            prev_images=prev,
            anchor_image=anchor,
            chain_segment_index=1,
            overlap_frames=22,
            max_safe_chain_length=3,
        )
        rep1 = json.loads(rep1_json)
        self.assertEqual(rep1["status"], "PASS")
        self.assertFalse(reset1)
        self.assertLess(seam1, 0.08)
        self.assertGreater(sharp1, 0.85)
        self.assertLess(col1, 0.05)

        # At segment 3 of a <=3 micro-chain, probe proactively recommends resetting anchor next
        rep3_json, _, _, _, reset3 = probe.probe(
            curr_images=curr,
            prev_images=prev,
            anchor_image=anchor,
            chain_segment_index=3,
            overlap_frames=22,
            max_safe_chain_length=3,
        )
        rep3 = json.loads(rep3_json)
        self.assertTrue(reset3)
        self.assertEqual(rep3["status"], "RESET_ANCHOR_NEXT")

    def test_detects_blur_color_drift_seam_pop_and_black_frames(self):
        anchor = _make_panning_clip(4, h=64, w=64, start_phase=0)
        prev = _make_panning_clip(30, h=64, w=64, start_phase=0)
        # Blurred + color-shifted + black tail frame (replicate pad avoids fake zero-border spikes)
        padded_src = torch.nn.functional.pad(
            _make_panning_clip(30, h=64, w=64, start_phase=30).permute(0, 3, 1, 2),
            (3, 3, 3, 3),
            mode="replicate",
        )
        degraded = torch.nn.functional.avg_pool2d(padded_src, kernel_size=7, stride=1).permute(0, 2, 3, 1)
        degraded[..., 0] = (degraded[..., 0] + 0.30).clamp(0.0, 1.0)
        degraded[0] = 0.95  # seam flash
        degraded[-1] = 0.0  # black frame

        probe = H3PrefixDriftQAProbe()
        rep_json, seam, sharp, col, reset = probe.probe(
            curr_images=degraded,
            prev_images=prev,
            anchor_image=anchor,
            chain_segment_index=2,
            overlap_frames=0,
        )
        rep = json.loads(rep_json)
        self.assertTrue(reset)
        self.assertEqual(rep["status"], "FAIL_DRIFT")
        self.assertGreater(seam, 0.12)
        self.assertLess(sharp, 0.72)
        self.assertGreater(col, 0.10)
        self.assertEqual(rep["black_frame_count"], 1)


class TestFourSentencePromptNode(unittest.TestCase):
    def test_sweet_spot_compilation_and_setting_bible_separation(self):
        bible = json.dumps({
            "characters": {
                "R01": {
                    "visual_anchor": "穿深青布衫的青年男子，短黑发",
                    "bio": "这是一个长达五百字的人物背景故事，讲述他在1985年离开乾头村远赴南方打工的经历……" * 8,
                }
            },
            "scenes": {
                "S03": {"anchor": "秋日古村石桥边，红柿满枝"}
            },
            "styles": {
                "ST01": {"anchor": "80年代胶片质感，暖秋柔光，情绪内敛"}
            },
        }, ensure_ascii=False)

        node = H3FourSentencePromptNode()
        prompt, neg, obedience, has_contra, diag_json = node.compile_prompt(
            scene_preset="drama",
            subject_anchor="R01",
            action_and_scene="站在桥头很纠结地望向远方溪水",
            shot_size="近景",
            camera_movement="Dolly in 缓慢推近",
            lighting="侧逆光",  # 3rd camera dim -> auto-trimmed to 2-of-3!
            style_closure="",
            setting_bible_json=bible,
            character_id="R01",
            scene_id="S03",
            style_id="ST01",
            negative_constraints="字幕，水印，多余路人，手指变形，画面闪烁",
        )
        diag = json.loads(diag_json)
        self.assertFalse(has_contra)
        self.assertGreaterEqual(obedience, 90)
        self.assertEqual(diag["length_zone"], "sweet_spot")
        # Long biography must NOT leak into single-shot prompt
        self.assertNotIn("1985年", prompt)
        self.assertIn("穿深青布衫的青年男子", prompt)
        self.assertIn("秋日古村石桥边", prompt)
        # Abstract '很纠结' rewritten to concrete micro-action
        self.assertIn("指尖轻敲桌面", prompt)
        self.assertNotIn("很纠结", prompt)
        # Negative list capped at <=2 items
        self.assertEqual(len(neg.split("，")), 2)
        # 2-of-3 camera rule enforced
        self.assertTrue(any("2-of-3" in n for n in diag["notes"]))

    def test_contradiction_guard_and_a30_listening_reaction_preset(self):
        node = H3FourSentencePromptNode()
        # Warn-only detects silhouette + clear face contradiction
        _, _, obed_warn, has_contra_warn, _ = node.compile_prompt(
            scene_preset="mv_broll",
            subject_anchor="穿白裙的长发女主",
            action_and_scene="窗前逆光剪影站立，面部清晰展现眼神细节",
            contradiction_policy="warn_only",
        )
        self.assertTrue(has_contra_warn)
        self.assertLess(obed_warn, 82)

        # Auto-resolve removes contradiction and restores high obedience
        prompt_fixed, _, obed_fixed, has_contra_fixed, _ = node.compile_prompt(
            scene_preset="mv_broll",
            subject_anchor="穿白裙的长发女主",
            action_and_scene="窗前逆光剪影站立，面部清晰展现眼神细节",
            contradiction_policy="auto_resolve",
        )
        self.assertFalse(has_contra_fixed)
        self.assertGreaterEqual(obed_fixed, 89)
        self.assertNotIn("逆光剪影", prompt_fixed)
        self.assertIn("面部清晰", prompt_fixed)

        # A30 listening_reaction preset strips singing verbs and locks closed lips
        prompt_a30, _, _, contra_a30, _ = node.compile_prompt(
            scene_preset="listening_reaction",
            subject_anchor="麻花辫蓝布衫女孩",
            action_and_scene="坐在柿子树下开口歌唱并望向男主",
            contradiction_policy="auto_resolve",
        )
        self.assertFalse(contra_a30)
        self.assertIn("双唇自然闭合安静倾听", prompt_a30)
        self.assertNotIn("开口歌唱", prompt_a30)

    def test_empirical_curve_800_char_overload_penalty(self):
        overloaded_text = ("穿风衣的角色在雨夜街道奔跑并同时特写瞳孔与大远景航拍，" * 32)
        scored = score_h3_prompt_obedience(overloaded_text, sentence_count=12, contradiction_count=1)
        self.assertEqual(scored["length_zone"], "overloaded")
        self.assertEqual(scored["slowdown_risk_pct"], 40)
        self.assertLessEqual(scored["estimated_obedience"], 60)

    def test_commercial_entry_second_and_talking_head_camera_omit(self):
        node = H3FourSentencePromptNode()
        p_comm, _, _, _, _ = node.compile_prompt(
            scene_preset="commercial",
            subject_anchor="磨砂黑咖啡机",
            action_and_scene="木质吧台萃取出浓郁油脂",
            product_entry_second=1.8,
        )
        self.assertIn("第1.8秒核心产品清晰入画", p_comm)

        _, _, _, _, diag_th = node.compile_prompt(
            scene_preset="talking_head",
            subject_anchor="穿西装的男主播",
            action_and_scene="书房桌前自然手势讲解",
            shot_size="",
            camera_movement="",
            lighting="",
        )
        self.assertEqual(json.loads(diag_th)["sentences"]["s3_camera_language"], "")


class TestVRAMStageGateAndMVMontage(unittest.TestCase):
    def test_vram_stage_gate_fallback_and_comfy_unload_hooks(self):
        gate = H3VRAMStageGate8GB()
        latent_payload = {
            "samples": (torch.randn(1, 16, 4, 8, 8), torch.randn(1, 32, 2, 16)),
            "minimax_prefix_frames": 39,
        }

        # 1. Works cleanly with zero ComfyUI / CUDA installed
        out_clean, rep_json, free_mb = gate.clean_stage(
            passthrough=latent_payload,
            stage_name="Seed-VC -> FLUX Keyframe",
            vram_budget_mb=8188,
        )
        rep = json.loads(rep_json)
        self.assertGreater(free_mb, 0.0)
        self.assertEqual(out_clean["minimax_prefix_frames"], 39)
        self.assertEqual(out_clean["samples"][0].device.type, "cpu")
        self.assertEqual(rep["vram_budget_mb"], 8188)

        # 2. Invokes comfy.model_management unload & soft_empty_cache when present
        mock_mm = types.SimpleNamespace(
            unload_all_models=MagicMock(),
            cleanup_models_gc=MagicMock(),
            soft_empty_cache=MagicMock(),
        )
        mock_comfy = types.SimpleNamespace(model_management=mock_mm)
        with patch.dict(sys.modules, {"comfy": mock_comfy, "comfy.model_management": mock_mm}):
            _, rep2_json, _ = gate.clean_stage(passthrough=torch.zeros(2, 16, 16, 3))
            rep2 = json.loads(rep2_json)
            self.assertTrue(rep2["comfy_models_unloaded"])
            self.assertTrue(rep2["comfy_cache_cleared"])
            mock_mm.unload_all_models.assert_called_once()
            mock_mm.cleanup_models_gc.assert_called_once()
            mock_mm.soft_empty_cache.assert_called_once()

    def test_dual_engine_mv_montage_locks_30fps_longcat_and_24fps_h3_to_48khz_master(self):
        # Master song: 6.0 seconds at 48kHz stereo (288,000 samples)
        master_audio = _make_sine_audio(frames=144, fps=24.0, sr=48000, freq=330.0)
        self.assertEqual(master_audio["waveform"].shape[-1], 288000)

        # LongCat A-Roll singing at 30fps (90 frames = 3.0s) @ 360x640
        aroll_1 = _make_panning_clip(90, h=64, w=96, start_phase=0)
        aroll_2 = _make_panning_clip(90, h=64, w=96, start_phase=50)

        # H3 PrefixStream B-Roll scenery at 24fps (72 frames = 3.0s) @ different aspect 48x80 and warmer tint
        broll_1 = (_make_panning_clip(72, h=48, w=80, start_phase=10) * 0.85).clamp(0.0, 1.0)

        timeline = json.dumps([
            {"role": "aroll", "clip_slot": "aroll_1", "start_sec": 0.0, "end_sec": 2.0, "transition": "crossfade"},
            {"role": "broll", "clip_slot": "broll_1", "start_sec": 2.0, "end_sec": 4.0, "transition": "cut"},
            {"role": "aroll", "clip_slot": "aroll_2", "start_sec": 4.0, "end_sec": 6.0, "transition": "crossfade"},
        ])

        stitcher = H3LongCatMVMontageStitcher()
        mv_images, mv_audio, total_frames, report_json = stitcher.assemble_mv(
            master_audio=master_audio,
            fps=24.0,
            target_width=96,
            target_height=64,
            fit_mode="cover_crop",
            transition_frames=4,
            color_harmonize_broll_to_aroll=True,
            timeline_spec_json=timeline,
            aroll_images_1=aroll_1,
            broll_images_1=broll_1,
            aroll_images_2=aroll_2,
            aroll_source_fps=30.0,
            broll_source_fps=24.0,
        )
        rep = json.loads(report_json)
        # 6.0s * 24fps = 144 exact frames (micro-crossfades MUST NOT shrink timeline!)
        self.assertEqual(total_frames, 144)
        self.assertEqual(tuple(mv_images.shape), (144, 64, 96, 3))
        self.assertEqual(mv_audio["sample_rate"], 48000)
        self.assertEqual(mv_audio["waveform"].shape[-1], 288000)
        self.assertEqual(rep["audio_video_drift_ms"], 0.0)
        self.assertEqual(rep["aroll_frames"], 96)
        self.assertEqual(rep["broll_frames"], 48)
        # Verify master audio waveform is bit-identical to original song (zero VAE re-encode distortion!)
        self.assertTrue(torch.allclose(mv_audio["waveform"], master_audio["waveform"]))

    def test_dual_engine_mv_auto_alternating_44_1khz_mode(self):
        master_audio = _make_sine_audio(frames=96, fps=24.0, sr=44100, freq=220.0)  # 4.0s
        aroll = _make_panning_clip(60, h=48, w=64, start_phase=0)  # 2.0s @ 30fps
        broll = _make_panning_clip(48, h=48, w=64, start_phase=20)  # 2.0s @ 24fps

        stitcher = H3LongCatMVMontageStitcher()
        mv_images, mv_audio, total_frames, rep_json = stitcher.assemble_mv(
            master_audio=master_audio,
            fps=24.0,
            timeline_spec_json="auto_alternating",
            aroll_images_1=aroll,
            broll_images_1=broll,
            aroll_source_fps=30.0,
            broll_source_fps=24.0,
        )
        self.assertEqual(total_frames, 96)
        self.assertEqual(mv_images.shape[0], 96)
        self.assertEqual(mv_audio["sample_rate"], 44100)
        self.assertEqual(mv_audio["waveform"].shape[-1], 4 * 44100)
        self.assertEqual(json.loads(rep_json)["audio_video_drift_ms"], 0.0)


class TestWorkflowTemplatesIntegrity(unittest.TestCase):
    def test_v2_workflow_templates_are_valid_json_and_reference_registered_nodes(self):
        from pathlib import Path

        repo_root = Path(__file__).resolve().parent.parent
        wf_4060 = repo_root / "examples" / "MiniMaxH3_4060Ti_8GB_3Seg_PrefixStream_MicroChain.json"
        wf_mv = repo_root / "examples" / "H3_LongCat_DualEngine_SoloMV_Master_Template.json"

        for wf_path in (wf_4060, wf_mv):
            self.assertTrue(wf_path.is_file(), f"Missing workflow template: {wf_path}")
            data = json.loads(wf_path.read_text(encoding="utf-8"))
            self.assertIn("nodes", data)
            self.assertIn("links", data)
            node_ids = {n["id"] for n in data["nodes"]}
            self.assertEqual(len(node_ids), len(data["nodes"]))
            for link in data["links"]:
                _, src_id, _, dst_id, _, _ = link
                self.assertIn(src_id, node_ids)
                self.assertIn(dst_id, node_ids)
            custom_types = {
                n["type"] for n in data["nodes"] if n["type"].startswith(("H3", "MiniMax")) and n["type"] != "MiniMaxH3ReferenceToVideo"
            }
            for ctype in custom_types:
                self.assertIn(ctype, NODE_CLASS_MAPPINGS)


if __name__ == "__main__":
    unittest.main()
