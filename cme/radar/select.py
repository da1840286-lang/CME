"""Automatic source selection for unattended runs.

Transfer prefers candidates with a commercial pass, but permits any playable, non-hard-excluded ad
as a pattern source. Strict commercial checks remain available as ranking evidence.
"""
from __future__ import annotations

AUTO_ALLOWED_CLASSES = ("DIRECT", "EXPANDED")


def block_reason(candidate: dict, allowed_classes=AUTO_ALLOWED_CLASSES, min_final_score=None) -> str:
    """Empty string when the candidate passes the legacy strict source-commercial gate."""
    if not candidate.get("video_url"):
        return "no video URL"

    check = candidate.get("source_check") or {}
    classification = str(check.get("classification") or "").upper()
    if check.get("excluded") or classification == "HARD_EXCLUDE":
        return "hard-excluded by the legacy source check"
    if allowed_classes is not None and classification not in allowed_classes:
        return f"source relevance {classification or 'MISSING'} is not one of {', '.join(allowed_classes)}"

    score = candidate.get("source_score") or {}
    if not isinstance(score, dict):
        score = {}
    action = str(score.get("action") or "").upper()
    tags_raw = score.get("tags") or []
    if isinstance(tags_raw, str):
        tags = {tags_raw}
    elif isinstance(tags_raw, (list, tuple, set)):
        tags = {str(tag) for tag in tags_raw}
    else:
        tags = set()

    components = score.get("components") or {}
    components = components if isinstance(components, dict) else {}
    money_gate = components.get("money_signal_gate") or {}
    money_gate = money_gate if isinstance(money_gate, dict) else {}
    duration = components.get("duration") or {}
    duration = duration if isinstance(duration, dict) else {}

    source_value_type = str(score.get("source_value_type") or "").upper()
    try:
        mechanism_score = float(score.get("mechanism_source_score") or 0)
    except (TypeError, ValueError):
        mechanism_score = 0.0
    mechanism_pass = (
        source_value_type in {"MECHANISM_SOURCE", "BOTH"} and mechanism_score >= 70.0
    ) or "long_source_allowed_for_mechanism" in tags

    # A scorer's explicit rejection is terminal. Mechanism-source is a separate path only
    # when the scorer itself did not reject the candidate.
    if action == "REJECT":
        return "legacy scorer action is REJECT"

    money_hard_block = money_gate.get("hard_block") is True or "money_signal_hard_block" in tags
    if money_hard_block and not mechanism_pass:
        return "money-signal hard block; no mechanism-source exception"

    duration_hard_block = duration.get("hard_block") is True or "duration_over_60_hard_block" in tags
    if duration_hard_block and not mechanism_pass:
        return "duration hard block; no mechanism-source exception"

    money_pass = money_gate.get("pass") is True or "money_signal_pass" in tags
    if not money_pass and not mechanism_pass:
        return "no explicit money-gate pass or mechanism-source score >= 70"

    final_score = score.get("final_score")
    if isinstance(final_score, (int, float)) and not isinstance(final_score, bool):
        if final_score <= 0 and not mechanism_pass:
            return f"final_score {final_score!r} is zero or below"
        if min_final_score is not None and final_score < min_final_score:
            return f"final_score {final_score!r} is below the required {min_final_score}"
    elif min_final_score is not None:
        return f"final_score {final_score!r} is below the required {min_final_score}"

    return ""


def select_candidate(candidates, allowed_classes=AUTO_ALLOWED_CLASSES, min_final_score=None):
    """Return the first candidate passing the legacy strict source-commercial gate."""
    skipped = []
    for item in candidates or []:
        reason = block_reason(item, allowed_classes, min_final_score)
        if not reason:
            return item, skipped
        skipped.append(f"{item.get('id')}: {reason}")
    return None, skipped



def pattern_source_reason(candidate: dict) -> str:
    """Return a blocker only when a source cannot safely be used to extract a pattern.

    Deliberately does not check scorer action, money signals, score floors, duration,
    or relevance labels. Those fields remain evidence/ranking metadata; they do not prove
    whether a new target-native creative can be made from the source's pattern.
    """
    if not isinstance(candidate, dict):
        return "candidate record is missing or invalid"
    if not candidate.get("video_url"):
        return "no video URL"

    check = candidate.get("source_check") or {}
    if not isinstance(check, dict):
        check = {}
    classification = str(check.get("classification") or "").upper()
    if check.get("excluded") or classification == "HARD_EXCLUDE":
        return "hard-excluded by the legacy source check"

    return ""


def select_pattern_candidate(candidates):
    """Choose the first playable, non-hard-excluded source in the existing FIND order."""
    skipped = []
    for item in candidates or []:
        reason = pattern_source_reason(item)
        if not reason:
            return item, skipped
        candidate_id = item.get("id") if isinstance(item, dict) else "<invalid>"
        skipped.append(f"{candidate_id}: {reason}")
    return None, skipped


STRONG_SOURCE_BONUS = 5.0       # soft preference for sources that also pass the commercial check
RELEVANCE_TIEBREAK = {"DIRECT": 2, "EXPANDED": 1}


def _score(item) -> float:
    score = (item.get("source_score") or {}) if isinstance(item, dict) else {}
    try:
        return float(score.get("final_score") or 0.0)
    except (TypeError, ValueError):
        return 0.0


def rank_transfer_candidates(candidates):
    """Usable sources ordered by score, with a soft bonus (not a gate) for commercially approved ones.

    A pattern can come from any playable, non-excluded ad, but a higher-scoring source is the better bet. Within
    the same adjusted score a source closer to the target category goes first; remaining ties keep FIND order.
    """
    usable, skipped = [], []
    for position, item in enumerate(candidates or []):
        reason = pattern_source_reason(item)
        if reason:
            candidate_id = item.get("id") if isinstance(item, dict) else "<invalid>"
            skipped.append(f"{candidate_id}: {reason}")
            continue
        adjusted = _score(item) + (0.0 if block_reason(item) else STRONG_SOURCE_BONUS)
        relevance = RELEVANCE_TIEBREAK.get(str((item.get("source_check") or {}).get("classification", "")).upper(), 0)
        usable.append((-adjusted, -relevance, position, item))
    usable.sort(key=lambda row: row[:3])
    return [row[3] for row in usable], skipped


def select_transfer_candidate(candidates):
    """Select a pattern source with a soft commercial preference, not a hard score gate."""
    ordered, skipped = rank_transfer_candidates(candidates)
    return (ordered[0] if ordered else None), skipped



def analysis_review_reason(candidate: dict, allowed_classes=None) -> str:
    """Backward-compatible alias for the former analysis-only gate.

    allowed_classes is accepted for older callers but intentionally does not gate
    pattern extraction. Relevance labels and commercial scores remain visible as evidence.
    """
    return pattern_source_reason(candidate)


def select_analysis_candidate(candidates, allowed_classes=None):
    """Backward-compatible alias; selects pattern sources, not commercially approved ads."""
    return select_pattern_candidate(candidates)
