"""Brain over API: structured output with a schema; retry only the atoms that failed.

`call` is injected (provider-specific code lives elsewhere): call(request: dict) -> dict.
"""
from __future__ import annotations
from .atoms import ATOM_SCHEMA, check_claims, parse_atoms
from .vertical import check_vertical


class BrainError(Exception):
    def __init__(self, errors):
        self.errors = errors
        super().__init__(f"Brain output still invalid after retries: {errors}")


class PatchError(Exception):
    pass


def build_request(offer, brief: str, shots: int, failed=None) -> dict:
    req = {"schema": ATOM_SCHEMA, "offer": offer.to_dict(), "brief": brief, "shots": shots,
           "rules": ["Return atoms only, no finished prompts.",
                     "Every kind='claim' atom must copy one verified claim verbatim and use channel='qa'; it records offer truth, not pixels.",
                     "Never put kind='claim' atoms in channel='image'. If visible proof is needed, create a separate image atom of kind product/action/scene that describes an observable cue, such as a close-up of the zip closure visibly shut.",
                     "Do not treat abstract claims such as reusable or dishwasher-safe as visually verifiable from a generic still. Ask for a visible label only when the brief explicitly requires on-image text.",
                     "Never invent text, pseudo-lettering, logos, brand names or labels. Reproduce them only when present in supplied product assets or explicitly specified by the offer/brief.",
                     "Never use a forbidden claim."]}
    if failed:
        req["only_fix"] = failed  # [{id, errors, previous_text}]
    return req


def request_atoms(offer, brief: str, shots: int, call, max_retries: int = 2, vertical=None) -> list:
    """Retry only what failed. An id's error clears only when that id arrives valid in a later round."""
    by_id: dict = {}
    pending: dict = {}
    failed = None
    for _ in range(max_retries + 1):
        resp = call(build_request(offer, brief, shots, failed))
        raw = resp.get("atoms", [])
        parsed = parse_atoms(raw)
        round_errs = {k: list(v) for k, v in parsed.errors.items()}
        for atom in parsed.atoms:
            if atom.shot > shots:
                round_errs.setdefault(atom.id, []).append(f"shot {atom.shot} is outside requested range 1..{shots}")
        if resp.get("_error"):
            round_errs["#root"] = [str(resp["_error"])]
        for aid, errs in check_claims(parsed.atoms, offer).items():
            round_errs.setdefault(aid, []).extend(errs)
        if vertical is not None:
            for aid, errs in check_vertical(parsed.atoms, offer, vertical).items():
                round_errs.setdefault(aid, []).extend(errs)
        for a in parsed.atoms:
            if a.id in round_errs:
                by_id.pop(a.id, None)
            else:
                by_id[a.id] = a
                pending.pop(a.id, None)
        # '#...' keys describe the previous response, not an id: they are recomputed every round
        pending = {k: v for k, v in pending.items() if not k.startswith("#")}
        pending.update(round_errs)
        for k in range(1, shots + 1):
            if not any(a.shot == k and a.channel == "image" and a.priority == "MUST" for a in by_id.values()):
                pending[f"#shot_{k}"] = [f"shot {k} has no MUST image atom"]
        if not pending:
            return list(by_id.values())
        prev_text = {i["id"]: i.get("text", "") for i in raw if isinstance(i, dict) and i.get("id")}
        failed = [{"id": k, "errors": v, "previous_text": prev_text.get(k, "")} for k, v in pending.items()]
    raise BrainError(pending)


def apply_patch(atoms, changes):
    """Patch by id. Brain returns only changed atoms; deleting a MUST needs a stated reason."""
    cur = {a.id: a for a in atoms}
    lost = []
    for ch in changes:
        aid, action = ch.get("id"), ch.get("action")
        if action == "delete":
            old = cur.get(aid)
            if old is None:
                raise PatchError(f"cannot delete unknown atom {aid}")
            if old.priority == "MUST" and not str(ch.get("reason", "")).strip():
                raise PatchError(f"deleting MUST atom {aid} requires a reason")
            lost.append(f"{aid} ({old.priority}): {ch.get('reason', 'no reason given')}")
            del cur[aid]
        elif action == "update":
            parsed = parse_atoms([ch.get("atom", {})])
            if parsed.errors or not parsed.atoms or parsed.atoms[0].id != aid:
                raise PatchError(f"invalid update for {aid}: {parsed.errors}")
            new = parsed.atoms[0]
            old = cur.get(aid)
            if old and old.priority == "MUST" and new.priority != "MUST":
                if not str(ch.get("reason", "")).strip():
                    raise PatchError(f"downgrading MUST atom {aid} requires a reason")
                lost.append(f"{aid} (MUST→{new.priority}): {ch['reason']}")
            cur[aid] = new
        else:
            raise PatchError(f"unknown action {action}")
    report = "потеряно: " + ("; ".join(lost) if lost else "нет")
    return list(cur.values()), report
