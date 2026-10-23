"""Automatic source selection for unattended runs.

Keeps the legacy ranking order (no new ranking rule). It only decides which ranked candidates are allowed to
reach the paid analyst/generator calls: a video must exist, the legacy source check must not exclude the ad,
and its relevance class must be confirmed. UNVERIFIED and LOW-relevance sources are skipped.
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
    if min_final_score is not None:
        score = (candidate.get("source_score") or {}).get("final_score")
        if not isinstance(score, (int, float)) or score < min_final_score:
            return f"final_score {score!r} is below the required {min_final_score}"
    return ""


def select_candidate(candidates, allowed_classes=AUTO_ALLOWED_CLASSES, min_final_score=None):
    """Return (candidate or None, [skip notes]) taking the first candidate in the given (legacy) order that passes."""
    skipped = []
    for item in candidates or []:
        reason = block_reason(item, allowed_classes, min_final_score)
        if not reason:
            return item, skipped
        skipped.append(f"{item.get('id')}: {reason}")
    return None, skipped
