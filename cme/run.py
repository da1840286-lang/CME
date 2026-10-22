"""Produce one shot: gates -> compile -> bind receipt -> send exact text -> QA -> verdict.

Every attempt leaves a record, including blocked ones and provider failures.
"""
from __future__ import annotations
import json
from pathlib import Path
from .compiler import BudgetError, compile_image_prompt
from .atoms import check_claims
from .atoms import check_claims
from .receipt import assert_exact_payload, attach_qa, build_receipt, overall
from .state import accept_shot, record_attempt
from .vertical import GateError, check_vertical, offer_gate


def _write(out: Path, name: str, data: dict) -> None:
    (out / "receipts").mkdir(parents=True, exist_ok=True)
    (out / "receipts" / f"{name}.json").write_text(json.dumps(data, ensure_ascii=False, indent=1), encoding="utf-8")


class AttemptLimit(Exception):
    pass


def _blocked(state, out, shot, attempt, status, detail):
    record_attempt(state, shot, "", status, detail)
    _write(out, f"blocked_{state.run_id}_s{shot:02d}_a{attempt:02d}",
           {"run_id": state.run_id, "shot": shot, "attempt": attempt, "overall": status, "error": detail,
            "commercial_status": "NOT_PRODUCED"})


def _commercial_status(provider_info: dict) -> str:
    """Anything produced by a free or never-confirmed adapter is a test artifact."""
    if provider_info and provider_info.get("tier") == "paid" and provider_info.get("verified"):
        return "CANDIDATE"
    return "TEST_ONLY"


def produce_shot(*, offer, atoms, shot, contract_id, state, out_dir, image_provider, qa_checks,
                 attempt=1, reserve=0, refs=(), vertical=None, max_attempts=0):
    """image_provider(prompt, refs) -> png bytes. qa_checks(path) -> list[CheckResult]."""
    out = Path(out_dir)
    state.bind(offer.offer_id, contract_id)
    if max_attempts and sum(1 for x in state.attempts if x.get("shot") == shot and x.get("receipt_id")) >= max_attempts:
        _blocked(state, out, shot, attempt, "BLOCKED_ATTEMPTS", f"{max_attempts} attempts already used for shot {shot}")
        raise AttemptLimit(f"attempt limit {max_attempts} reached for shot {shot} in this offer/contract")
    # Defence in depth: all entry paths, including hand-supplied atoms, get the same contract checks.
    claim_errors = check_claims(atoms, offer)
    if claim_errors:
        reasons = [f"{aid}: {msg}" for aid, msgs in claim_errors.items() for msg in msgs]
        _blocked(state, out, shot, attempt, "BLOCKED_CLAIMS", "; ".join(reasons))
        raise GateError(reasons)
    if not any(a.shot == shot and a.channel == "image" and a.priority == "MUST" for a in atoms):
        reason = f"shot {shot} has no MUST image atom"
        _blocked(state, out, shot, attempt, "BLOCKED_GATE", reason)
        raise GateError([reason])
    ad_requirements = ()
    if vertical is not None:                       # gates run before any paid call
        gate = offer_gate(offer, vertical)
        reasons = list(gate.blocked) + [f"{k}: {v[0]}" for k, v in check_vertical(atoms, offer, vertical).items()]
        if reasons:
            _blocked(state, out, shot, attempt, "BLOCKED_GATE", "; ".join(reasons))
            raise GateError(reasons)
        ad_requirements = gate.requirements
    try:
        compiled = compile_image_prompt(atoms, shot, reserve=reserve)
    except BudgetError as e:
        _blocked(state, out, shot, attempt, "BLOCKED_BUDGET", str(e))
        raise
    receipt = build_receipt(contract_id, state.run_id, shot, attempt, atoms, compiled.text)
    assert_exact_payload(receipt, compiled.text)                   # nothing appended after binding
    image_path = out / "shots" / f"shot_{shot:02d}_a{attempt:02d}.png"
    image_path.parent.mkdir(parents=True, exist_ok=True)
    base = {"lost": compiled.lost_line(), "ad_requirements": list(ad_requirements),
            "provider": {}, "commercial_status": "TEST_ONLY"}
    try:
        image_path.write_bytes(image_provider(receipt.sent_prompt, list(refs) + list(offer.photos)))
        base["provider"] = dict(getattr(image_provider, "info", {}) or {})
        base["commercial_status"] = _commercial_status(base["provider"])
        receipt = attach_qa(receipt, atoms, qa_checks(image_path))
    except Exception as e:                          # keep the diagnostics, then let the caller see the error
        base["provider"] = dict(getattr(image_provider, "info", {}) or base.get("provider", {}))
        status = f"ERROR_{type(e).__name__}"
        record_attempt(state, shot, receipt.receipt_id, status, str(e))
        _write(out, receipt.receipt_id, {**receipt.to_dict(), **base, "overall": status, "error": str(e)})
        raise
    verdict = overall(receipt)
    record_attempt(state, shot, receipt.receipt_id, verdict)
    if verdict == "PASS":
        accept_shot(state, shot, image_path, receipt.receipt_id)
    _write(out, receipt.receipt_id, {**receipt.to_dict(), **base, "image_path": str(image_path)})
    return receipt, compiled, verdict
