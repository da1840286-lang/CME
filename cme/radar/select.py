"""Automatic source selection for unattended runs.

Keeps the legacy ranking order. It gates the handoff to analyst/generator calls:
video, confirmed target relevance, and an explicit commercial or mechanism-source pass.
A high view count or DIRECT label alone is not a commercial pass.
"""
from __future__ import annotations

AUTO_ALLOWED_CLASSES = ("DIRECT", "EXPANDED")


def block_reason(candidate: dict, allowed_classes=AUTO_ALLOWED_CLASSES, min_final_score=None) -> str:
    """Empty string when the candidate may be analysed automatically, otherwise the reason it may not."""
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
    """Return (candidate or None, [skip notes]) in legacy ranking order."""
    skipped = []
    for item in candidates or []:
        reason = block_reason(item, allowed_classes, min_final_score)
        if not reason:
            return item, skipped
        skipped.append(f"{item.get('id')}: {reason}")
    return None, skipped



def analysis_review_reason(
    candidate: dict,
    allowed_classes=AUTO_ALLOWED_CLASSES,
    *,
    min_final_score: float = 45.0,
    min_performance_score: float = 55.0,
    min_commercial_score: float = 40.0,
) -> str:
    """Gate for a *sampled-video analysis only*, never for image generation.

    Existing production-approved candidates can also be analysed. The additional review
    route is intentionally narrow: the sole blocker must be the scorer's missing money
    gate; there must be a usable video, confirmed target relevance, no hard duration
    block, and moderate independent pre-video scores. The analysis result is evidence to
    inspect, not proof of conversion or profitability.
    """
    reason = block_reason(candidate, allowed_classes=allowed_classes)
    if not reason:
        return ""
    if reason != "legacy scorer action is REJECT":
        return reason

    score = candidate.get("source_score") or {}
    if not isinstance(score, dict):
        return "source score is missing or invalid"
    if str(score.get("action") or "").upper() != "REJECT":
        return "rejected source is not an explicit scorer REJECT"

    components = score.get("components") or {}
    components = components if isinstance(components, dict) else {}
    money_gate = components.get("money_signal_gate")
    if not isinstance(money_gate, dict) or money_gate.get("pass") is not False:
        return "review path requires an explicit failed money-signal gate"

    duration = components.get("duration") or {}
    duration = duration if isinstance(duration, dict) else {}
    tags_raw = score.get("tags") or []
    tags = {tags_raw} if isinstance(tags_raw, str) else {str(tag) for tag in tags_raw} if isinstance(tags_raw, (list, tuple, set)) else set()
    if duration.get("hard_block") is True or "duration_over_60_hard_block" in tags:
        return "duration hard block; source cannot enter review path"

    source_value_type = str(score.get("source_value_type") or "").upper()
    try:
        mechanism_score = float(score.get("mechanism_source_score") or 0)
    except (TypeError, ValueError):
        mechanism_score = 0.0
    mechanism_pass = source_value_type in {"MECHANISM_SOURCE", "BOTH"} and mechanism_score >= 70.0
    if mechanism_pass:
        return "scorer marked this mechanism source REJECT; manual inspection is required"

    def numeric(name: str):
        value = score.get(name)
        return float(value) if isinstance(value, (int, float)) and not isinstance(value, bool) else None

    final_score = numeric("final_score")
    if final_score is None or final_score < min_final_score:
        return f"final_score {final_score!r} is below analysis-review floor {min_final_score}"
    performance_score = numeric("performance_score")
    if performance_score is None or performance_score < min_performance_score:
        return f"performance_score {performance_score!r} is below analysis-review floor {min_performance_score}"
    commercial_score = numeric("commercial_score")
    if commercial_score is None or commercial_score < min_commercial_score:
        return f"commercial_score {commercial_score!r} is below analysis-review floor {min_commercial_score}"

    return ""


def select_analysis_candidate(candidates, allowed_classes=AUTO_ALLOWED_CLASSES):
    """Choose the first source eligible for bounded video analysis, preserving ranked order."""
    skipped = []
    for item in candidates or []:
        reason = analysis_review_reason(item, allowed_classes=allowed_classes)
        if not reason:
            return item, skipped
        skipped.append(f"{item.get('id')}: {reason}")
    return None, skipped
