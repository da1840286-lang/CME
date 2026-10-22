"""Tiered QA. Code checks first. A model judge can add evidence but never grants PASS alone."""
from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from PIL import Image, ImageChops, ImageStat
from .atoms import norm


@dataclass(frozen=True)
class CheckResult:
    check_id: str
    status: str  # PASS / FAIL / NOT_RUN
    required: bool = True
    source: str = "code"  # code / judge
    detail: str = ""


def file_check(check_id, path, expected_size=None, required=True) -> CheckResult:
    p = Path(path)
    if not p.exists() or p.stat().st_size == 0:
        return CheckResult(check_id, "FAIL", required, "code", "file missing or empty")
    try:
        with Image.open(p) as im:
            size = im.size
    except Exception as e:  # unreadable image is a real failure, reported with the reason
        return CheckResult(check_id, "FAIL", required, "code", f"unreadable image: {e}")
    if expected_size and tuple(size) != tuple(expected_size):
        return CheckResult(check_id, "FAIL", required, "code", f"size {size} != {expected_size}")
    return CheckResult(check_id, "PASS", required, "code", f"size {size}")


def anchor_similarity(check_id, prev_path, cur_path, region, min_similarity=0.85, required=True):
    """Compare a normalized region (x0, y0, x1, y1) of two frames; guards continuity anchors."""
    try:
        with Image.open(prev_path) as a, Image.open(cur_path) as b:
            b = b.convert("L")
            a = a.convert("L").resize(b.size)
            w, h = b.size
            box = (int(region[0] * w), int(region[1] * h), int(region[2] * w), int(region[3] * h))
            ca, cb = a.crop(box).resize((32, 32)), b.crop(box).resize((32, 32))
            diff = ImageStat.Stat(ImageChops.difference(ca, cb)).mean[0] / 255.0
    except Exception as e:
        return CheckResult(check_id, "FAIL", required, "code", f"cannot compare: {e}")
    sim = 1.0 - diff
    status = "PASS" if sim >= min_similarity else "FAIL"
    return CheckResult(check_id, status, required, "code", f"similarity {sim:.3f} (min {min_similarity})")


def ocr_text_check(check_id, path, text_allowed, ocr=None, max_chars=3, required=True):
    """No-text rule. Without an OCR engine the check is NOT_RUN, never a silent PASS."""
    if ocr is None:
        try:
            import pytesseract  # optional dependency
            ocr = lambda p: pytesseract.image_to_string(Image.open(p))  # noqa: E731
        except Exception:
            return CheckResult(check_id, "NOT_RUN", required, "code", "no OCR engine available")
    try:
        found = "".join(ch for ch in ocr(path) if ch.isalnum())
    except Exception as e:
        return CheckResult(check_id, "NOT_RUN", required, "code", f"OCR failed: {e}")
    if text_allowed or len(found) <= max_chars:
        return CheckResult(check_id, "PASS", required, "code", f"{len(found)} text chars")
    return CheckResult(check_id, "FAIL", required, "code", f"{len(found)} text chars found, none allowed")


def judge_checks(path, qa_atoms, judge_fn, required=False):
    """judge_fn(path, atom_text) -> bool. Evidence only: source='judge'."""
    out = []
    for a in qa_atoms:
        try:
            ok = bool(judge_fn(path, a.text))
            out.append(CheckResult(a.id, "PASS" if ok else "FAIL", required, "judge", ""))
        except Exception as e:
            out.append(CheckResult(a.id, "NOT_RUN", required, "judge", f"judge error: {e}"))
    return out


def judge_batch_checks(path, atoms, judge_fn, required=False):
    """One judge request per frame; malformed or evidence-free rows are NOT_RUN."""
    try:
        answers = judge_fn(path, atoms)
    except Exception as e:
        return [CheckResult(a.id, "NOT_RUN", required, "judge", f"judge unavailable: {e}") for a in atoms]
    if not isinstance(answers, dict):
        return [CheckResult(a.id, "NOT_RUN", required, "judge", "malformed judge response") for a in atoms]
    out = []
    for a in atoms:
        r = answers.get(a.id)
        if not isinstance(r, dict):
            out.append(CheckResult(a.id, "NOT_RUN", required, "judge", "no valid answer for this requirement"))
            continue
        evidence = r.get("evidence")
        if type(r.get("pass")) is not bool or not isinstance(evidence, str) or not evidence.strip():
            out.append(CheckResult(a.id, "NOT_RUN", required, "judge", "malformed answer or missing evidence"))
            continue
        out.append(CheckResult(a.id, "PASS" if r["pass"] else "FAIL", required, "judge", evidence.strip()[:120]))
    return out


def make_qa_checks(atoms, shot, judge_fn=None, ocr=None, expected_size=None, ocr_required=True,
                   text_allowed=False, anchors=(), verified_claims=()):
    """Build QA for one shot. Offer claims are contract-checked, not inferred from pixels."""
    claims = [a for a in atoms if a.shot == shot and a.channel == "qa" and a.kind == "claim"]
    claim_set = {norm(c) for c in verified_claims}
    # A claim atom is an offer-truth record. Visual proof, when needed, is a separate image atom.
    pool = [a for a in atoms if a.shot == shot and a.channel in ("image", "qa")
            and not (a.channel == "qa" and a.kind == "claim")]

    def run(path):
        res = [file_check("file", path, expected_size),
               ocr_text_check("no_text", path, text_allowed, ocr=ocr, required=ocr_required)]
        for cid, prev, region, minimum in anchors:
            res.append(anchor_similarity(cid, prev, path, region, minimum))
        for a in claims:
            valid = norm(a.text) in claim_set
            res.append(CheckResult(
                a.id, "PASS" if valid else "FAIL", a.priority == "MUST", "code",
                "matches verified offer claim; not inferred from pixels" if valid
                else "claim is absent from offer.verified_claims"))
        if judge_fn is not None and pool:
            res.extend(judge_batch_checks(path, pool, judge_fn))
        return res

    return run
