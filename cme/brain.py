"""Brain over API: structured output with a schema; retry only the atoms that failed.

`call` is injected (provider-specific code lives elsewhere): call(request: dict) -> dict.
"""
from __future__ import annotations
from .atoms import ATOM_SCHEMA, check_claims, check_language, parse_atoms
from .vertical import check_vertical


class BrainError(Exception):
    def __init__(self, errors):
        self.errors = errors
        super().__init__(f"Brain output still invalid after retries: {errors}")


class PatchError(Exception):
    pass


# Narrow bridge from the older CME transfer contract into the atom authoring request.
# This is a payload adapter, not a new pipeline or atom schema.
def _pick_fields(value, keys):
    if not isinstance(value, dict):
        return {}
    return {key: value[key] for key in keys if key in value and value[key] not in (None, "", [], {})}


def _shot_fields(shots, keys, id_keys=("shot_id", "id")):
    if not isinstance(shots, list):
        return []
    out = []
    for index, shot in enumerate(shots[:8], start=1):
        if not isinstance(shot, dict):
            continue
        row = _pick_fields(shot, keys)
        for key in id_keys:
            if key in shot:
                row["shot_id"] = shot[key]
                break
        row.setdefault("shot_id", index)
        if row:
            out.append(row)
    return out


def build_source_transfer_context(contract: dict) -> dict:
    """Reduce an old CME contract or a saved Radar analysis to useful transfer context.

    Source URLs, raw media paths, and the original source shot prompts are deliberately
    omitted. The current OfferCard remains authoritative for target-product facts.
    """
    if not isinstance(contract, dict):
        raise ValueError("source contract must be a JSON object")

    analysis = contract.get("result") or contract.get("data") or contract
    if isinstance(analysis, dict) and ("mechanism_hypothesis" in analysis or "transferability" in analysis) and not contract.get("mechanism"):
        def statement(key):
            val = analysis.get(key)
            if isinstance(val, dict):
                return val.get("text") or val.get("value")
            return val
        return {
            "format": "radar_analysis_hypothesis",
            "source_evidence_scope": "manual analysis of sampled frames only; not full-video verification or commercial proof",
            "hook": statement("hook"),
            "buyer_problem": statement("buyer_problem"),
            "product_intervention": statement("product_intervention"),
            "result_demo": statement("result_demo"),
            "cta": statement("cta"),
            "mechanism_hypothesis": statement("mechanism_hypothesis"),
            "transferability_hypothesis": statement("transferability"),
            "limits": _pick_fields(analysis.get("limits"), ("text", "missing")),
            "evidence": analysis.get("evidence", [])[:8] if isinstance(analysis.get("evidence"), list) else [],
        }

    mechanism = contract.get("mechanism")
    representation = contract.get("mechanism_representation")
    transfer = contract.get("transfer")
    plan = contract.get("rebuild_plan")
    creative = contract.get("image_creative_contract")
    if not isinstance(creative, dict) and isinstance(plan, dict):
        nested = plan.get("image_creative_contract")
        creative = nested if isinstance(nested, dict) else {}
    creative = creative if isinstance(creative, dict) else {}
    plan = plan if isinstance(plan, dict) else {}
    mechanism = mechanism if isinstance(mechanism, dict) else {}
    representation = representation if isinstance(representation, dict) else {}
    transfer = transfer if isinstance(transfer, dict) else {}
    if not any((mechanism, representation, transfer, plan, creative)):
        raise ValueError(
            "unsupported source contract: expected old CME mechanism/transfer/rebuild_plan/image_creative_contract "
            "or a saved Radar analysis with mechanism_hypothesis and transferability"
        )

    review = contract.get("source_review") if isinstance(contract.get("source_review"), dict) else {}
    verification = review.get("source_verification") if isinstance(review.get("source_verification"), dict) else {}
    verify_status = str(verification.get("status") or "").strip().upper()
    if verification and verification.get("required", True) and verify_status != "PASS":
        raise ValueError(
            "source contract is not accepted: required source_verification.status must be PASS "
            f"(received {verify_status or 'MISSING'})"
        )
    source_review = _pick_fields(verification, ("status", "scope", "required", "evidence"))
    # Source-context and mechanism fields explain the commercial idea; raw source URLs and
    # source video/image prompts are not sent to the Brain.
    mechanism_fields = (
        "name", "canonical_slug", "mechanism_type", "human_desire", "problem", "barrier", "tension",
        "psychological_trigger", "belief_shift", "attention_mechanism", "conversion_mechanism", "proof",
        "transferability", "transfer_notes", "source_context", "contraindications", "required_conditions",
        "mechanism_signature", "mechanism_family", "family_confidence",
    )
    representation_fields = (
        "trigger", "user_problem", "hidden_constraint", "intervention", "belief_shift", "proof",
        "desired_action", "abstraction_statement", "causal_chain", "invariant_elements",
        "source_surface_to_discard", "transfer_conditions", "failure_conditions", "abstraction_confidence",
        "purity_score", "required_conditions", "contraindications", "causal_invariants",
        "abstraction_status", "completeness_score",
    )
    transfer_fields = (
        "transfer_distance", "rationale", "preserve_rules", "change_rules", "contraindications",
        "target_context", "assessment", "reasoning_confidence", "confidence",
    )
    plan_fields = ("target_offer", "total_duration_seconds", "aspect_ratio", "strategy", "mechanism_contract",
                   "visual_strategy", "creative_direction", "funnel_bridge", "voiceover", "cta", "surface_variants")
    shot_keys = (
        "start", "end", "purpose", "scene_description", "action", "subject_motion", "camera_motion",
        "movement_direction", "motion_style", "framing", "camera_angle", "subject_placement", "visual_action",
        "proof", "state_before", "state_after", "mechanism_slots", "result_artifact", "visual_delta_from_previous",
    )
    creative_shot_keys = (
        "purpose", "scene_description", "action", "proof", "state_before", "state_after",
        "mechanism_slots", "result_artifact", "visual_delta_from_previous",
    )
    creative_fields = ("schema_version", "selected_angle", "visual_concept", "hook_frame_purpose",
                       "transfer_explanation", "negative_prompt")
    result = {
        "format": "legacy_cme_transfer_contract",
        "source_review": source_review,
        "mechanism": _pick_fields(mechanism, mechanism_fields),
        "mechanism_representation": _pick_fields(representation, representation_fields),
        "transfer": _pick_fields(transfer, transfer_fields),
        "rebuild_plan": _pick_fields(plan, plan_fields),
        "target_shots": _shot_fields(plan.get("shots"), shot_keys, id_keys=("id", "shot_id")),
        "image_creative_contract": _pick_fields(creative, creative_fields),
        "image_creative_shots": _shot_fields(creative.get("shots"), creative_shot_keys),
    }
    # Keep transfer evidence useful but bounded; do not ship unselected alternative ad scripts.
    assess = result.get("transfer", {}).get("assessment")
    if isinstance(assess, dict):
        keep_assessment = (
            "verdict", "compatibility_score", "source_vertical", "target_vertical", "shared_psychological_core",
            "source_specific_components", "target_native_reframing", "mismatch_reasons", "evidence_for_transfer",
            "required_target_conditions", "counterfactual_failures", "replacement_requirements", "causal_mapping",
            "confidence", "rationale",
        )
        result["transfer"]["assessment"] = _pick_fields(assess, keep_assessment)
    if isinstance(transfer.get("angles"), list):
        result["transfer"]["available_angle_count"] = len(transfer["angles"])
        # The selected angle is represented by rebuild_plan/image_creative_contract, so unused
        # angle scripts are intentionally not copied into the atom authoring prompt.
    return result


# ---- mechanism coverage -------------------------------------------------------------------------
# The atom schema is unchanged. A mechanism link is carried by an ordinary MUST atom whose id starts with
# `mech_<link>_`; code can then verify that every required link reached the contract, and the Judge checks
# those atoms on the frame like any other MUST atom.
MECHANISM_LINKS = ("problem", "intervention", "result")
LINK_TEXT_LIMIT = 220


def _link_text(value) -> str:
    if isinstance(value, dict):
        value = value.get("text") or value.get("value") or " ".join(str(v) for v in value.values() if isinstance(v, str))
    elif isinstance(value, (list, tuple)):
        value = "; ".join(str(v) for v in value if isinstance(v, (str, int, float)))
    return " ".join(str(value or "").split())[:LINK_TEXT_LIMIT]


def derive_mechanism_links(source_transfer) -> list:
    """Pick the causal links the target frames must keep: problem -> intervention -> visible result.

    Derived by code from the already-reduced source_transfer context, never by the Brain. Missing links are
    simply not required (unknown stays unknown).
    """
    if not isinstance(source_transfer, dict):
        return []
    if source_transfer.get("format") == "radar_analysis_hypothesis":
        sources = {"problem": source_transfer.get("buyer_problem"),
                   "intervention": source_transfer.get("product_intervention"),
                   "result": source_transfer.get("result_demo")}
    else:
        rep = source_transfer.get("mechanism_representation") or {}
        mech = source_transfer.get("mechanism") or {}
        sources = {"problem": rep.get("user_problem") or rep.get("trigger") or mech.get("problem"),
                   "intervention": rep.get("intervention"),
                   "result": rep.get("proof") or mech.get("proof")}
    links = []
    for name in MECHANISM_LINKS:
        text = _link_text(sources.get(name))
        if text:
            links.append({"link": name, "atom_id_prefix": f"mech_{name}_", "source_statement": text})
    return links


def _atom_view(atom):
    get = atom.get if isinstance(atom, dict) else (lambda k, d=None: getattr(atom, k, d))
    return get("id") or "", get("priority"), get("channel"), get("kind")


def missing_mechanism_links(atoms, links) -> list:
    """Link names that no MUST atom (image or qa channel, not a claim) carries."""
    missing = []
    for link in links or []:
        prefix = link["atom_id_prefix"] if isinstance(link, dict) else f"mech_{link}_"
        name = link["link"] if isinstance(link, dict) else link
        ok = False
        for atom in atoms:
            aid, prio, ch, kind = _atom_view(atom)
            if str(aid).startswith(prefix) and prio == "MUST" and ch in ("image", "qa") and kind != "claim":
                ok = True
                break
        if not ok:
            missing.append(name)
    return missing


def reserved_system_atom_errors(atoms) -> dict:
    """Brain must not author a no-text criterion; QA injects the canonical system rule."""
    errors = {}
    for atom in atoms:
        aid, _priority, _channel, _kind = _atom_view(atom)
        if str(aid) == "qa_no_text" or str(aid).startswith("qa_no_text_"):
            errors[str(aid)] = ["SYSTEM_OWNED_QA: do not author qa_no_text atoms; the canonical no-text check is injected by code"]
    return errors


def visual_structure_errors(atoms, shots: int, links) -> dict:
    """Deterministic visual-contract checks; this is structural validation, not semantic vision."""
    rows = []
    for atom in atoms:
        get = atom.get if isinstance(atom, dict) else (lambda key, default=None: getattr(atom, key, default))
        rows.append({"id": str(get("id") or ""), "kind": get("kind"), "priority": get("priority"), "channel": get("channel"), "shot": get("shot")})
    names = {str(link.get("link") if isinstance(link, dict) else link) for link in (links or [])}
    errors = {}
    for row in rows:
        if row["id"].startswith("comp_") and not (
            row["kind"] == "scene" and row["channel"] == "image" and row["priority"] == "MUST"
            and type(row["shot"]) is int and 1 <= row["shot"] <= shots
        ):
            errors[row["id"]] = ["COMPOSITION_ATOM_SHAPE: atoms with id prefix comp_ must be kind=scene, channel=image, priority=MUST, and assigned to a requested shot"]
    if shots == 1 and {"problem", "result"}.issubset(names):
        comp = [r for r in rows if r["id"].startswith("comp_") and r["kind"] == "scene" and r["channel"] == "image" and r["priority"] == "MUST" and r["shot"] == 1]
        if not comp:
            errors["#composition"] = ["COMPOSITION_REQUIRED: shots=1 with problem+result mechanism links requires a MUST image scene atom for shot 1 whose id starts with comp_; describe an achievable layout and visible before/result regions"]
    if shots >= 2:
        expected_shot = {"problem": 1, "intervention": 1, "result": 2}
        for name in sorted(names.intersection(expected_shot)):
            expected = expected_shot[name]
            prefix = f"mech_{name}_"
            matching = [r for r in rows if r["id"].startswith(prefix) and r["kind"] != "claim" and r["priority"] == "MUST" and r["channel"] == "image"]
            if not any(r["shot"] == expected for r in matching):
                errors[f"#mechanism_shot_{name}"] = [f"MECHANISM_SHOT_ORDER: required MUST image link '{name}' must be assigned to shot {expected}"]
            for row in matching:
                if row["shot"] != expected:
                    errors[row["id"]] = [f"MECHANISM_SHOT_ORDER: atom '{row['id']}' carries link '{name}' but is in shot {row['shot']}; assign it to shot {expected}"]
    return errors


def build_request(offer, brief: str, shots: int, failed=None, source_transfer=None) -> dict:
    req = {"schema": ATOM_SCHEMA, "offer": offer.to_dict(), "brief": brief, "shots": shots,
           "rules": ["Return atoms only, no finished prompts.",
                     "Every kind='claim' atom must copy one verified claim verbatim and use channel='qa'; it records offer truth, not pixels.",
                     "Never put kind='claim' atoms in channel='image'. If visible proof is needed, create a separate image atom of kind product/action/scene that describes an observable cue, such as a close-up of the zip closure visibly shut.",
                     "Do not treat abstract care or durability claims as visually verifiable from a generic still. Ask for a visible label only when the brief explicitly requires on-image text.",
                     "Every kind='qa_check' atom must use channel='qa'; it is a verification criterion and must not be included in the image-generation prompt.",
                     "Do not emit an atom whose id is 'qa_no_text' or starts with 'qa_no_text_'. The canonical no-added-text requirement is injected and checked by code, so Brain must not author a duplicate QA criterion.",
                     "Never invent text, pseudo-lettering, logos, brand names or labels. Reproduce them only when present in supplied product assets or explicitly specified by the offer/brief.",
                     "Never use a forbidden claim.",
                     "When target-product reference photos are supplied, preserve the actual target product's form factor, silhouette, proportions, and distinctive visible parts. Keep the target product body recognizable in the frame; do not substitute a generic lookalike or change a compact handheld product into an upright/stick device.",
                     "Write every atom text in English, even when the input context is in another language. Translate the source statements; never copy non-English words into atoms."]}
    if source_transfer:
        req["source_transfer"] = source_transfer
        links = derive_mechanism_links(source_transfer)
        if links:
            req["mechanism_links"] = links
            req["rules"].append(
                "mechanism_links lists the causal links that must survive the transfer. For EACH link create at least one "
                "MUST atom with channel='image' (kind product/action/scene/composition, never claim) whose id starts with its "
                "atom_id_prefix (for example mech_intervention_clamp) and whose text shows that link in the TARGET product's own "
                "scene, with visible, checkable content. Do not copy the source product, brand, creator or setting.")
            link_names = {link["link"] for link in links}
            if shots == 1 and {"problem", "result"}.issubset(link_names):
                req["rules"].append(
                    "ONE-SHOT COMPOSITION CONTRACT: because both problem and result must be visible in one frame, create an explicit additional MUST image atom with kind='scene', shot=1, and id prefix 'comp_'. Describe a concrete legible layout and locate the problem/result regions; do not copy an existing mech atom into comp_.")
            elif shots >= 2:
                req["rules"].append(
                    "MULTI-SHOT ROUTING CONTRACT: mech_problem_* and mech_intervention_* MUST image atoms go to shot 1; mech_result_* MUST image atoms go to shot 2.")
        req["rules"].extend([
            "When source_transfer is supplied, preserve its causal mechanism and target-native rebuild sequence as structured atom requirements.",
            "The current OfferCard and supplied target-product photos are the ONLY authority for target-product facts. Source-transfer context is authority only for the causal mechanism, not target hardware or specifications. Never copy source-only attachments, tools, features, colors, construction, or product form into target atoms unless visible in target-product references or explicitly confirmed by OfferCard/brief. When no target-native feature is verified, retain the generic target product shown in the reference and transfer the demo logic only.",
            "Transfer the problem/intervention/proof logic, not the source ad's exact scene, creator, brand, UI, props, wording, or composition. Respect source_specific_components, replacement_requirements, must_change, contraindications, and limits.",
            "For each requested shot, keep the target-native action, proof, state_before/state_after, and result/delta where present. Express required visual directions as stable MUST atoms; do not silently omit them.",
            "Do not pack several unverified accessories into one MUST atom. Keep the main target product body visible and show only the attachment performing the action; omit accessories not visible in the target reference or confirmed by OfferCard/brief.",
            "When a MUST image atom depicts an intervention result or before/after delta, define a visibly checkable contrast in the same frame: specify where the untreated/dirty side and treated/clean side are located (for example, left/right); keep the intervention tool/action at their visible boundary; make the cleaned result a substantial, unobstructed debris-free strip that is not hidden under the tool. Avoid relying only on phrases like 'completely clean' without describing the visible boundary and changed area.",
            "The source's commercial metrics or reasoning-confidence values are not proof that the mechanism is profitable for the target offer.",
        ])
    if failed:
        req["only_fix"] = failed  # [{id, errors, previous_text}]
    return req


def request_atoms(offer, brief: str, shots: int, call, max_retries: int = 2, vertical=None, source_transfer=None) -> list:
    """Retry only what failed. An id's error clears only when that id arrives valid in a later round."""
    by_id: dict = {}
    pending: dict = {}
    failed = None
    required_links = derive_mechanism_links(source_transfer)
    for _ in range(max_retries + 1):
        resp = call(build_request(offer, brief, shots, failed, source_transfer=source_transfer))
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
        for aid, errs in check_language(parsed.atoms).items():
            round_errs.setdefault(aid, []).extend(errs)
        reserved_no_text = reserved_system_atom_errors(parsed.atoms)
        for aid, errs in reserved_no_text.items():
            round_errs.setdefault("#system_owned_qa", []).extend(f"{aid}: {error}" for error in errs)
        if vertical is not None:
            for aid, errs in check_vertical(parsed.atoms, offer, vertical).items():
                round_errs.setdefault(aid, []).extend(errs)
        reserved_no_text_ids = set(reserved_no_text)
        for a in parsed.atoms:
            if a.id in round_errs or a.id in reserved_no_text_ids:
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
        for name in missing_mechanism_links(by_id.values(), required_links):
            pending[f"#mechanism_{name}"] = [f"mechanism link '{name}' has no MUST image atom with id prefix mech_{name}_"]
        structure_errors = visual_structure_errors(by_id.values(), shots, required_links)
        pending.update(structure_errors)
        for aid in list(structure_errors):
            if not aid.startswith("#"):
                by_id.pop(aid, None)
        for name in missing_mechanism_links(by_id.values(), required_links):
            pending[f"#mechanism_{name}"] = [f"mechanism link '{name}' has no MUST image atom with id prefix mech_{name}_"]
        structure_errors = visual_structure_errors(by_id.values(), shots, required_links)
        pending.update(structure_errors)
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
