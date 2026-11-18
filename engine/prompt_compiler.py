"""MiniMax H3 Empirical 150-Char 4-Sentence Prompt Compiler & Contradiction Guard.

Grounded in SI Video Lab's 50/150/400/800-char H3 benchmark and MiniMax-H3 official guide:
- 50 chars (1 sentence): ~72% obedience (stable, but vague words cause guessing)
- 150 chars (4-sentence structure): ~91% obedience sweet spot, fastest inference, highest hit rate
- 400 chars (long narrative): ~78% obedience (focus buried, camera terms ignored)
- 800+ chars (stacked requirements): <=60% obedience (contradictions self-destruct, frame tearing, +40% slower)

Core Rules:
1. Sentence 1 (Subject + Appearance Anchor, <=30 chars): Lock concrete visual traits; keep long lore/biography in Setting Bible.
2. Sentence 2 (Concrete Action + Scene): Observable micro-actions instead of abstract emotion words; A30 closed-lip guard for listening reactions; exact entry second for commercials.
3. Sentence 3 (Camera Language, strictly <=2 of 3 dimensions): Shot size / Camera movement (Dolly/Orbit/Tracking) / Lighting (Rim light/Side key). Omit for simple talking_head if unrequested.
4. Sentence 4 (Style & Texture Closure): Unified visual palette & mood.
5. Negative constraints capped at <=2 critical items; mutually contradictory instructions detected and optionally resolved before GPU spend.
"""

from typing import Any, Dict, List, Optional, Tuple
import json
import re


# Abstract emotion -> concrete observable micro-action dictionary (Chinese + English)
EMOTION_MICRO_ACTION_MAP: List[Tuple[str, str]] = [
    ("非常纠结", "指尖轻敲桌面，目光微垂"),
    ("很纠结", "指尖轻敲桌面，目光微垂"),
    ("十分纠结", "指尖轻敲桌面，眉头微蹙"),
    ("纠结", "指尖轻敲桌面，目光微垂"),
    ("非常伤心", "眼眶微红，低头轻抿嘴唇"),
    ("很伤心", "眼眶微红，低头轻抿嘴唇"),
    ("十分悲伤", "眼角泛光，视线缓缓垂落"),
    ("很悲伤", "眼角泛光，指尖轻颤"),
    ("非常紧张", "手指收紧衣角，呼吸微促"),
    ("很紧张", "手指收紧衣角，喉结微动"),
    ("非常愤怒", "下颌收紧，指节攥白"),
    ("很生气", "眉峰压低，目光冷冽直视"),
    ("很开心", "嘴角自然扬起，眉眼舒展"),
    ("非常开心", "眉眼弯起，露出克制浅笑"),
    ("沉浸聆听", "双唇自然闭合，目光安静凝视"),
    ("陷入沉思", "托腮凝视窗外，手指轻顿"),
    ("很迷茫", "视线游移失焦，脚步微顿"),
    ("very conflicted", "fingers lightly tapping the table, gaze lowered"),
    ("very sad", "eyes glistening, head slightly bowed with lips pressed"),
    ("very nervous", "fingers tightening on coat hem, shallow breath"),
]

# Mutually destructive instruction pairs (category_id, group_a_keywords, group_b_keywords, explanation, keep_side_by_default)
CONTRADICTION_RULES: List[Tuple[str, Tuple[str, ...], Tuple[str, ...], str]] = [
    (
        "silhouette_vs_clear_face",
        ("逆光剪影", "全剪影", "纯黑剪影", "silhouette", "backlit silhouette"),
        ("面部清晰", "五官清晰", "眼神细节", "看清表情", "面部特写清晰", "clear face", "clear facial features", "detailed eyes"),
        "逆光剪影与面部五官清晰互相冲突（必删其一，否则画面撕裂或曝光闪烁）",
    ),
    (
        "closed_lips_vs_vocalizing",
        ("闭唇", "双唇紧闭", "双唇自然闭合", "沉默不语", "安静倾听", "不说话", "lips closed", "silent listening", "mouth closed"),
        ("开口歌唱", "高声歌唱", "大声说话", "念台词", "开口说话", "张嘴演唱", "singing", "speaking aloud", "belting lyrics"),
        "闭唇沉默倾听与开口歌唱/念白直接冲突（A30 口型污染高危项）",
    ),
    (
        "extreme_motion_vs_static_freeze",
        ("极速甩镜", "剧烈奔跑", "高速追逐", "急速旋转", "whip pan", "fast sprint", "high-speed chase"),
        ("静止定格", "完全静止", "微距定格", "纹丝不动", "static freeze", "completely motionless", "still frame"),
        "剧烈高速运动/甩镜与静止定格冲突",
    ),
    (
        "extreme_wide_vs_macro_detail",
        ("大远景", "航拍全景", "全身极小", "微小身影", "extreme wide shot", "aerial establishing"),
        ("瞳孔特写", "睫毛微颤", "毛孔细节", "指尖微距", "extreme close-up", "pupil macro", "eyelash detail"),
        "大远景全身小人与微距五官/瞳孔细节无法在同一单镜同时呈现",
    ),
    (
        "pitch_dark_vs_high_key",
        ("全黑无光", "伸手不见五指", "pitch black", "complete darkness"),
        ("高调明亮", "通透明亮细节", "high-key bright", "sunlit bright clarity"),
        "全黑无光与高调明亮细节互相抵消",
    ),
]

VOCALIZING_STRIP_PATTERNS = (
    "开口歌唱", "高声歌唱", "张嘴演唱", "开口说话", "大声说话", "念台词", "对口型演唱",
    "singing aloud", "belting lyrics", "speaking dialogue",
)


def _clean_clause(text: str) -> str:
    if not text:
        return ""
    cleaned = re.sub(r"\s+", " ", str(text)).strip()
    cleaned = cleaned.strip("。；;，,.!！ ")
    return cleaned


def _parse_setting_bible(setting_bible_json: str) -> Dict[str, Dict[str, Any]]:
    """Parse optional character/scene/style anchor registry JSON without failing on plain text."""
    raw = (setting_bible_json or "").strip()
    if not raw:
        return {"characters": {}, "scenes": {}, "styles": {}}
    try:
        data = json.loads(raw)
    except Exception:
        return {"characters": {}, "scenes": {}, "styles": {}}
    if not isinstance(data, dict):
        return {"characters": {}, "scenes": {}, "styles": {}}

    out: Dict[str, Dict[str, Any]] = {"characters": {}, "scenes": {}, "styles": {}}
    for bucket in ("characters", "scenes", "styles"):
        val = data.get(bucket, {})
        if isinstance(val, dict):
            out[bucket] = val
        elif isinstance(val, list):
            mapped = {}
            for item in val:
                if isinstance(item, dict) and item.get("id"):
                    mapped[str(item["id"])] = item
            out[bucket] = mapped
    # Also allow flat ID map like {"R01": "穿米色风衣的短发女孩", "S03": "黄昏天台边缘"}
    for k, v in data.items():
        if k in ("characters", "scenes", "styles"):
            continue
        ks = str(k).upper()
        if ks.startswith("R") or ks.startswith("C"):
            out["characters"][str(k)] = v
        elif ks.startswith("ST"):
            out["styles"][str(k)] = v
        elif ks.startswith("S"):
            out["scenes"][str(k)] = v
    return out


def _extract_anchor_text(entry: Any, max_chars: int = 30) -> str:
    if isinstance(entry, str):
        txt = _clean_clause(entry)
    elif isinstance(entry, dict):
        txt = _clean_clause(
            entry.get("visual_anchor")
            or entry.get("anchor")
            or entry.get("appearance")
            or entry.get("summary")
            or entry.get("name")
            or ""
        )
    else:
        txt = ""
    if max_chars > 0 and len(txt) > max_chars:
        # Cut cleanly at nearest comma/space before max_chars if possible
        sub = txt[:max_chars]
        for sep in ("，", ",", "、", " "):
            pos = sub.rfind(sep)
            if pos >= max_chars // 2:
                return sub[:pos].strip()
        return sub.strip()
    return txt


def resolve_setting_anchors(
    subject_anchor: str,
    action_and_scene: str,
    style_closure: str,
    setting_bible_json: str = "",
    character_id: str = "",
    scene_id: str = "",
    style_id: str = "",
) -> Tuple[str, str, str, List[str]]:
    """Resolve compact IDs (e.g. R01, S03, ST01) from Setting Bible while keeping long lore out of single-shot prompts."""
    bible = _parse_setting_bible(setting_bible_json)
    notes: List[str] = []

    subj = _clean_clause(subject_anchor)
    act = _clean_clause(action_and_scene)
    sty = _clean_clause(style_closure)

    cid = (character_id or "").strip()
    if cid and cid in bible["characters"]:
        resolved = _extract_anchor_text(bible["characters"][cid], max_chars=30)
        if resolved:
            subj = resolved if not subj or subj == cid else f"{resolved}"
            notes.append(f"Resolved character {cid} -> '{resolved}' (stripped long biography)")

    # Inline token replacement e.g. "角色R01" or "R01"
    for k, v in bible["characters"].items():
        token_anchor = _extract_anchor_text(v, max_chars=30)
        if not token_anchor:
            continue
        for pat in (f"角色{k}", k):
            if pat in subj:
                subj = subj.replace(pat, token_anchor)
                notes.append(f"Expanded {pat} -> '{token_anchor}'")

    sid = (scene_id or "").strip()
    if sid and sid in bible["scenes"]:
        s_anchor = _extract_anchor_text(bible["scenes"][sid], max_chars=36)
        if s_anchor and s_anchor not in act:
            act = f"{s_anchor}，{act}" if act else s_anchor
            notes.append(f"Resolved scene {sid} -> '{s_anchor}'")

    for k, v in bible["scenes"].items():
        s_anchor = _extract_anchor_text(v, max_chars=36)
        if not s_anchor:
            continue
        for pat in (f"场景{k}", k):
            if pat in act:
                act = act.replace(pat, s_anchor)
                notes.append(f"Expanded {pat} -> '{s_anchor}'")

    stid = (style_id or "").strip()
    if stid and stid in bible["styles"]:
        st_anchor = _extract_anchor_text(bible["styles"][stid], max_chars=40)
        if st_anchor:
            sty = st_anchor if not sty else sty
            notes.append(f"Resolved style {stid} -> '{st_anchor}'")

    if len(subj) > 30:
        original_len = len(subj)
        subj = _extract_anchor_text(subj, max_chars=30)
        notes.append(f"Trimmed Sentence 1 subject anchor from {original_len} -> {len(subj)} chars (<=30 char lock)")

    return subj, act, sty, notes


def rewrite_abstract_emotions(action_text: str, scene_preset: str = "drama") -> Tuple[str, List[str]]:
    """Replace vague emotion adjectives with observable filmable micro-actions."""
    out = _clean_clause(action_text)
    rewrites: List[str] = []
    for abstract_kw, concrete_action in EMOTION_MICRO_ACTION_MAP:
        if abstract_kw in out:
            out = out.replace(abstract_kw, concrete_action)
            rewrites.append(f"'{abstract_kw}' -> '{concrete_action}'")

    if scene_preset == "listening_reaction":
        for vocal_kw in VOCALIZING_STRIP_PATTERNS:
            if vocal_kw in out:
                out = out.replace(vocal_kw, "双唇自然闭合安静倾听")
                rewrites.append(f"A30 guard stripped vocal verb '{vocal_kw}'")
        if not any(k in out for k in ("闭唇", "双唇自然闭合", "lips closed", "mouth closed")):
            out = f"{out}，双唇自然闭合安静倾听" if out else "双唇自然闭合安静倾听，目光微动"
            rewrites.append("Injected A30 observable closed-lip listening cue")

    return _clean_clause(out), rewrites


def build_camera_sentence(
    shot_size: str = "",
    camera_movement: str = "",
    lighting: str = "",
    scene_preset: str = "drama",
) -> Tuple[str, List[str]]:
    """Enforce '景别 / 运镜 / 光线 三选二' rule (and omit for talking_head when blank)."""
    s_size = _clean_clause(shot_size)
    c_move = _clean_clause(camera_movement)
    light = _clean_clause(lighting)
    notes: List[str] = []

    dims: List[Tuple[str, str]] = []
    if s_size and s_size.lower() not in ("none", "无"):
        dims.append(("shot_size", s_size))
    if c_move and c_move.lower() not in ("none", "无"):
        dims.append(("camera_movement", c_move))
    if light and light.lower() not in ("none", "无"):
        dims.append(("lighting", light))

    if scene_preset == "talking_head" and not dims:
        notes.append("talking_head preset: omitted optional camera sentence to lock focus on speaker")
        return "", notes

    if len(dims) == 3:
        # Strictly pick 2 of 3 per H3 empirical guide based on preset priority
        if scene_preset in ("commercial", "mv_broll"):
            kept = [d for d in dims if d[0] in ("shot_size", "camera_movement")]
            dropped = "lighting"
        else:
            # Default: keep camera_movement + lighting (or shot_size + lighting if movement is static)
            kept = [dims[0], dims[2]] if "static" in c_move.lower() or "固定" in c_move else [dims[0], dims[1]]
            # Wait: classic example in guide is "近景缓慢推近，侧逆光" -> shot_size/movement merged or 2 items.
            # Let's keep shot_size + camera_movement OR camera_movement + lighting cleanly:
            kept = [dims[1], dims[2]] if scene_preset == "listening_reaction" else [dims[0], dims[1]]
            dropped = [d[0] for d in dims if d not in kept][0]
        notes.append(f"Enforced 2-of-3 camera rule (dropped '{dropped}' to prevent instruction dilution)")
        dims = kept

    return "，".join(val for _, val in dims), notes


def trim_negative_constraints(negative_text: str, max_items: int = 2) -> Tuple[str, int]:
    """Cap negative constraints to <=2 highest-priority items (never a laundry list)."""
    raw = (negative_text or "").strip()
    if not raw:
        return "", 0
    parts = [p.strip() for p in re.split(r"[，,、；;\n]+", raw) if p.strip()]
    kept = parts[:max(0, int(max_items))]
    dropped_count = max(0, len(parts) - len(kept))
    return "，".join(kept), dropped_count


def lint_h3_contradictions(prompt_text: str) -> List[Dict[str, Any]]:
    """Detect mutually destructive instruction pairs in Chinese or English."""
    text_lower = (prompt_text or "").lower()
    issues: List[Dict[str, Any]] = []

    for rule_id, group_a, group_b, message in CONTRADICTION_RULES:
        hit_a = [kw for kw in group_a if kw.lower() in text_lower]
        hit_b = [kw for kw in group_b if kw.lower() in text_lower]
        if hit_a and hit_b:
            issues.append({
                "rule_id": rule_id,
                "conflict_a": hit_a,
                "conflict_b": hit_b,
                "message": message,
            })
    return issues


def resolve_contradictions_in_text(text: str, issues: List[Dict[str, Any]], scene_preset: str = "drama") -> Tuple[str, List[str]]:
    """Deterministically drop one conflicting term per contradiction rule."""
    out = text
    actions: List[str] = []
    for issue in issues:
        rid = issue["rule_id"]
        # Choose which side to remove
        if rid == "silhouette_vs_clear_face":
            # Keep facial clarity unless style explicitly targets pure shadow play; remove silhouette term
            to_drop = issue["conflict_a"]
        elif rid == "closed_lips_vs_vocalizing":
            to_drop = issue["conflict_b"] if scene_preset == "listening_reaction" else issue["conflict_a"]
        else:
            to_drop = issue["conflict_b"]

        for kw in to_drop:
            pattern = re.compile(re.escape(kw), flags=re.IGNORECASE)
            out = pattern.sub("", out)
            actions.append(f"Auto-resolved '{rid}' by removing conflicting term '{kw}'")

    out = re.sub(r"[，,]{2,}", "，", out)
    out = re.sub(r"。{2,}", "。", out).strip("，, ")
    return out, actions


def score_h3_prompt_obedience(
    prompt_text: str,
    sentence_count: int,
    contradiction_count: int,
    negative_dropped: int = 0,
) -> Dict[str, Any]:
    """Score prompt against the empirical H3 obedience curve (1-sentence ~50c -> 72, 4-sentence ~55..200c -> 91+, ~400c -> 78, 800c -> 60)."""
    clean_body = (prompt_text or "").strip()
    char_len = len(clean_body)

    if char_len == 0:
        return {"estimated_obedience": 0, "length_zone": "empty", "char_count": 0, "sentence_count": 0, "slowdown_risk_pct": 0}

    # Count CJK chars vs ASCII chars so compact Chinese 4-sentence prompts (~55-180 chars, matching the official 57-char exemplar)
    # and English 4-sentence prompts (~120-260 chars) both land squarely in the 4-sentence sweet spot.
    cjk_count = sum(1 for ch in clean_body if "\u4e00" <= ch <= "\u9fff")
    ascii_equiv_units = cjk_count * 2.2 + (char_len - cjk_count) * 1.0

    if (char_len < 36 or ascii_equiv_units < 75.0) or (sentence_count <= 1 and ascii_equiv_units < 125.0):
        # 1-sentence / ~50-char vague zone -> ~72 obedience
        base = 68.0 + min(7.0, (ascii_equiv_units / 110.0) * 6.0)
        zone = "too_short"
    elif char_len <= 220 and ascii_equiv_units <= 440.0:
        # 4-sentence sweet spot (~55-185 Chinese chars or ~110-320 English chars) -> ~91-95 obedience
        dist_from_opt = abs(ascii_equiv_units - 240.0)
        base = 93.0 - min(3.5, (dist_from_opt / 200.0) * 3.0)
        zone = "sweet_spot"
    elif char_len <= 520:
        # ~400-char long narrative zone -> ~78 obedience
        ratio = min(1.0, max(0.0, (char_len - 220.0) / 300.0))
        base = 86.0 - ratio * 11.0
        zone = "diluted"
    else:
        # 800+ char overloaded zone -> <=60 obedience (+40% slower inference)
        ratio = min(1.0, (char_len - 520.0) / 320.0)
        base = 73.0 - ratio * 16.0
        zone = "overloaded"

    # Structural bonus/penalty
    if 3 <= sentence_count <= 4 and zone == "sweet_spot":
        base += 1.5
    elif sentence_count > 6:
        base -= min(8.0, (sentence_count - 6) * 2.0)

    # Contradiction penalty (mutually destructive pairs tank obedience!)
    base -= contradiction_count * 16.0

    score = int(round(max(20.0, min(98.0, base))))
    return {
        "estimated_obedience": score,
        "length_zone": zone,
        "char_count": char_len,
        "sentence_count": sentence_count,
        "slowdown_risk_pct": 40 if zone == "overloaded" else (15 if zone == "diluted" else 0),
    }


def compile_h3_four_sentence_prompt(
    subject_anchor: str,
    action_and_scene: str,
    shot_size: str = "近景",
    camera_movement: str = "Dolly in 缓慢推近",
    lighting: str = "侧逆光",
    style_closure: str = "胶片颗粒，自然景深，情绪克制",
    scene_preset: str = "drama",
    setting_bible_json: str = "",
    character_id: str = "",
    scene_id: str = "",
    style_id: str = "",
    negative_constraints: str = "",
    contradiction_policy: str = "auto_resolve",
    product_entry_second: float = 0.0,
    ref_image_tag: str = "",
    ref_audio_tag: str = "",
) -> Dict[str, Any]:
    """Compile a structured 4-sentence MiniMax H3 shot prompt with contradiction guard and obedience telemetry."""
    notes: List[str] = []

    s1, s2_raw, s4, anchor_notes = resolve_setting_anchors(
        subject_anchor=subject_anchor,
        action_and_scene=action_and_scene,
        style_closure=style_closure,
        setting_bible_json=setting_bible_json,
        character_id=character_id,
        scene_id=scene_id,
        style_id=style_id,
    )
    notes.extend(anchor_notes)

    s2, emotion_rewrites = rewrite_abstract_emotions(s2_raw, scene_preset=scene_preset)
    notes.extend(emotion_rewrites)

    if scene_preset == "commercial" and product_entry_second > 0.0:
        entry_clause = f"第{product_entry_second:.1f}秒核心产品清晰入画"
        if "秒" not in s2 and "s" not in s2.lower():
            s2 = f"{s2}，{entry_clause}" if s2 else entry_clause
            notes.append(f"Injected commercial exact product entry timestamp ({product_entry_second:.1f}s)")

    s3, cam_notes = build_camera_sentence(
        shot_size=shot_size,
        camera_movement=camera_movement,
        lighting=lighting,
        scene_preset=scene_preset,
    )
    notes.extend(cam_notes)

    # Assemble the 4 core sentences (1 sentence = 1 job)
    sentences = [seg for seg in (s1, s2, s3, s4) if seg]
    core_prompt = "。".join(sentences) + ("。" if sentences else "")

    # Optional compact reference tags (<Picture 1>, <Audio 1>) without 800-char bloat
    ref_parts = []
    if ref_image_tag and ref_image_tag.strip():
        ref_parts.append(ref_image_tag.strip())
    if ref_audio_tag and ref_audio_tag.strip():
        ref_parts.append(ref_audio_tag.strip())
    ref_prefix = (" ".join(ref_parts) + " ") if ref_parts else ""

    compiled = f"{ref_prefix}{core_prompt}".strip()

    neg_trimmed, neg_dropped = trim_negative_constraints(negative_constraints, max_items=2)
    if neg_dropped > 0:
        notes.append(f"Trimmed {neg_dropped} excess negative items (kept top <=2 critical constraints)")
    if neg_trimmed:
        compiled_with_neg = f"{compiled} 禁止：{neg_trimmed}。"
    else:
        compiled_with_neg = compiled

    # Contradiction lint & optional auto-resolution
    detected_issues = lint_h3_contradictions(compiled_with_neg)
    resolved_actions: List[str] = []
    final_prompt = compiled_with_neg
    if detected_issues and contradiction_policy == "auto_resolve":
        final_prompt, resolved_actions = resolve_contradictions_in_text(
            final_prompt, detected_issues, scene_preset=scene_preset
        )
        notes.extend(resolved_actions)
        remaining_issues = lint_h3_contradictions(final_prompt)
    else:
        remaining_issues = detected_issues

    obedience_info = score_h3_prompt_obedience(
        prompt_text=core_prompt,
        sentence_count=len(sentences),
        contradiction_count=len(remaining_issues),
        negative_dropped=neg_dropped,
    )

    return {
        "compiled_prompt": final_prompt,
        "core_four_sentence_prompt": core_prompt,
        "negative_prompt": neg_trimmed,
        "sentences": {
            "s1_subject_anchor": s1,
            "s2_action_scene": s2,
            "s3_camera_language": s3,
            "s4_style_closure": s4,
        },
        "scene_preset": scene_preset,
        "estimated_obedience": obedience_info["estimated_obedience"],
        "length_zone": obedience_info["length_zone"],
        "char_count": obedience_info["char_count"],
        "slowdown_risk_pct": obedience_info["slowdown_risk_pct"],
        "has_contradiction": len(remaining_issues) > 0,
        "detected_contradictions": detected_issues,
        "remaining_contradictions": remaining_issues,
        "notes": notes,
    }
