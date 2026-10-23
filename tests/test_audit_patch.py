"""Regression tests for the audit patch: English atoms, honest PASS, mechanism coverage, strict auto-selection."""
import json
import tempfile
import unittest
from pathlib import Path

from cme.adapters.openrouter import make_openrouter_brain
from cme.atoms import Atom, check_language
from cme.brain import (BrainError, build_request, build_source_transfer_context, derive_mechanism_links,
                       missing_mechanism_links, request_atoms)
from cme.offer import load_offer
from cme.qa import CheckResult, file_check, frame_content_check
from cme.radar.select import block_reason, select_candidate
from cme.receipt import attach_qa, build_receipt, overall
from cme.smoke import contract_id_for, main as smoke_main, offer_fingerprint, validate_atoms_manifest
from tests.helpers import png_bytes

ROOT = Path(__file__).resolve().parent.parent
EX = ROOT / "examples" / "offer_example.json"


def atom(i, text="A clean kitchen counter", shot=1, kind="scene", prio="MUST", ch="image"):
    return {"id": i, "kind": kind, "text": text, "priority": prio, "channel": ch, "shot": shot}


RADAR = {"result": {
    "hook": {"text": "Visible mess", "kind": "OBSERVED"},
    "buyer_problem": {"text": "Crumbs trapped between car seats", "kind": "OBSERVED"},
    "product_intervention": {"text": "Handheld vacuum lifts crumbs from the seat gap", "kind": "OBSERVED"},
    "result_demo": {"text": "Seat gap visibly clean afterwards", "kind": "OBSERVED"},
    "mechanism_hypothesis": {"text": "Show mess, remove it, show clean", "kind": "INFERRED"},
    "transferability": {"text": "Works for other cleaning tools", "kind": "INFERRED"}}}
FULL = [atom("mech_problem_seat", "Crumbs packed into a seat gap"),
        atom("mech_intervention_nozzle", "Compact vacuum nozzle pulling crumbs from the gap"),
        atom("mech_result_clean", "The same seat gap now visibly clean")]


class LanguageTests(unittest.TestCase):
    def test_non_latin_letters_are_rejected_accents_are_fine(self):
        bad = check_language([Atom("ru", "scene", "Чистый салон автомобиля", "MUST", "image", 1)])
        self.assertIn("NON_ENGLISH_ATOM", bad["ru"][0])
        self.assertEqual(check_language([Atom("ok", "scene", "A café-style counter, 3 m wide", "MUST", "image", 1)]), {})

    def test_brain_retries_a_russian_atom_and_rules_demand_english(self):
        offer = load_offer(EX)
        self.assertTrue(any("English" in r for r in build_request(offer, "b", 1)["rules"]))
        calls = []

        def call(req):
            calls.append(req)
            return {"atoms": [atom("s1_scene", "Чистый салон")]} if len(calls) == 1 else {"atoms": [atom("s1_scene")]}
        self.assertEqual(len(request_atoms(offer, "b", 1, call)), 1)
        self.assertIn("NON_ENGLISH_ATOM", json.dumps(calls[1]["only_fix"], ensure_ascii=False))
        with self.assertRaises(BrainError):
            request_atoms(offer, "b", 1, lambda r: {"atoms": [atom("s1_scene", "Чистый салон")]}, max_retries=0)


class HonestPassTests(unittest.TestCase):
    def setUp(self):
        self.atoms = [Atom("a", "scene", "A bright kitchen", "MUST", "image", 1)]
        self.r = build_receipt("c1", "r1", 1, 1, self.atoms, "A bright kitchen")

    def test_blank_frame_fails_textured_frame_passes(self):
        with tempfile.TemporaryDirectory() as d:
            blank, ok = Path(d) / "b.png", Path(d) / "o.png"
            blank.write_bytes(png_bytes(blank=True))
            ok.write_bytes(png_bytes())
            self.assertEqual(frame_content_check("frame_content", blank).status, "FAIL")
            self.assertEqual(frame_content_check("frame_content", ok).status, "PASS")
            self.assertTrue(frame_content_check("frame_content", ok).pixels)

    def test_file_integrity_plus_judge_is_not_a_pass(self):
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "f.png"
            p.write_bytes(png_bytes())
            res = [file_check("file", p), CheckResult("a", "PASS", True, "judge", "ok")]
            self.assertEqual(overall(attach_qa(self.r, self.atoms, res)), "UNVERIFIED")

    def test_pixel_check_plus_judge_passes_and_blank_frame_cannot(self):
        res = [CheckResult("frame_content", "PASS", True, "code", "", True), CheckResult("a", "PASS", True, "judge", "ok")]
        self.assertEqual(overall(attach_qa(self.r, self.atoms, res)), "PASS")
        res[0] = CheckResult("frame_content", "FAIL", True, "code", "blank", True)
        self.assertEqual(overall(attach_qa(self.r, self.atoms, res)), "FAILED_QA")

    def test_claim_match_is_not_pixel_evidence(self):
        atoms = [Atom("a", "scene", "A bright kitchen", "MUST", "image", 1),
                 Atom("c", "claim", "Cordless car vacuum", "MUST", "qa", 1)]
        r = build_receipt("c1", "r1", 1, 1, atoms, "A bright kitchen")
        res = [CheckResult("c", "PASS", True, "code", "matches verified offer claim"),
               CheckResult("a", "PASS", True, "judge", "ok")]
        self.assertEqual(overall(attach_qa(r, atoms, res)), "UNVERIFIED")


class MechanismCoverageTests(unittest.TestCase):
    def ctx(self, drop=()):
        data = json.loads(json.dumps(RADAR))
        for k in drop:
            del data["result"][k]
        return build_source_transfer_context(data)

    def test_links_are_derived_by_code_and_missing_ones_stay_unrequired(self):
        self.assertEqual([l["link"] for l in derive_mechanism_links(self.ctx())], ["problem", "intervention", "result"])
        self.assertEqual([l["link"] for l in derive_mechanism_links(self.ctx(drop=("result_demo",)))], ["problem", "intervention"])

    def test_legacy_contract_links(self):
        ctx = build_source_transfer_context({"mechanism_representation": {
            "user_problem": "Dust on the shelf", "intervention": "Wipe with the tool", "proof": ["Shelf visibly clean"]}})
        self.assertEqual([l["link"] for l in derive_mechanism_links(ctx)], ["problem", "intervention", "result"])

    def test_request_carries_links_and_openrouter_forwards_them(self):
        req = build_request(load_offer(EX), "b", 1, source_transfer=self.ctx())
        self.assertEqual(len(req["mechanism_links"]), 3)
        self.assertTrue(any("atom_id_prefix" in r for r in req["rules"]))
        seen = []

        def send(method, url, headers, payload):
            seen.append(payload)
            return 200, json.dumps({"model": payload["model"], "choices": [{"message": {"content": json.dumps({"atoms": FULL})}}]}).encode()
        make_openrouter_brain({"model_id": "v/t", "response_format_mode": "json_object"}, "k", send)(req)
        self.assertEqual(len(json.loads(seen[0]["messages"][1]["content"])["mechanism_links"]), 3)

    def test_brain_must_cover_every_link_and_only_the_missing_ones_are_retried(self):
        offer, ctx, calls = load_offer(EX), self.ctx(), []

        def call(req):
            calls.append(req)
            return {"atoms": FULL[:1] + [atom("s1_scene")]} if len(calls) == 1 else {"atoms": FULL[1:]}
        atoms = request_atoms(offer, "b", 1, call, source_transfer=ctx)
        self.assertEqual(missing_mechanism_links(atoms, ["problem", "intervention", "result"]), [])
        fix_ids = {f["id"] for f in calls[1]["only_fix"]}
        self.assertEqual(fix_ids, {"#mechanism_intervention", "#mechanism_result"})

    def test_brain_fails_closed_when_a_link_is_never_carried(self):
        with self.assertRaises(BrainError) as cm:
            request_atoms(load_offer(EX), "b", 1, lambda r: {"atoms": [atom("s1_scene")]}, max_retries=1,
                          source_transfer=self.ctx())
        self.assertIn("#mechanism_problem", cm.exception.errors)

    def test_claim_or_motion_atoms_do_not_count_as_a_carrier(self):
        fake = [atom("mech_problem_x", "Cordless car vacuum", kind="claim", ch="qa"),
                atom("mech_intervention_x", "slow push in", ch="motion"),
                atom("mech_result_x", "clean seat", prio="SHOULD")]
        self.assertEqual(missing_mechanism_links(fake, ["problem", "intervention", "result"]),
                         ["problem", "intervention", "result"])

    def test_manifest_records_links_and_frame_stage_rejects_a_contract_that_lost_one(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "settings.json").write_text((ROOT / "settings.example.json").read_text(encoding="utf-8"), encoding="utf-8")
            (root / "src.json").write_text(json.dumps(RADAR), encoding="utf-8")

            def transport(method, url, headers, payload):
                body = {"model": payload["model"], "choices": [{"message": {"content": json.dumps({"atoms": FULL})}}]}
                return 200, json.dumps(body).encode()
            out = root / "out"
            code = smoke_main(["brain", "--settings", str(root / "settings.json"), "--out", str(out),
                               "--source-contract", str(root / "src.json")],
                              {"transport": transport, "openrouter_key": "test-key"})
            self.assertEqual(code, 0)
            manifest = json.loads((out / "atoms_manifest.json").read_text(encoding="utf-8"))
            self.assertEqual(manifest["mechanism_links"], ["problem", "intervention", "result"])
            offer = load_offer(EX)
            data = json.loads((out / "atoms.json").read_text(encoding="utf-8"))
            validate_atoms_manifest(out / "atoms.json", out / "atoms_manifest.json", data, offer, shot=1)
            # Re-bind a contract that silently dropped the result link: hashes are consistent, coverage is not.
            data = [a for a in data if a["id"] != "mech_result_clean"]
            (out / "atoms.json").write_text(json.dumps(data, indent=1), encoding="utf-8")
            fingerprint, photos = offer_fingerprint(offer)
            import hashlib
            manifest.update({"atoms_sha256": hashlib.sha256((out / "atoms.json").read_bytes()).hexdigest(),
                             "contract_id": contract_id_for(data, offer)})
            (out / "atoms_manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
            with self.assertRaises(BrainError) as cm:
                validate_atoms_manifest(out / "atoms.json", out / "atoms_manifest.json", data, offer, shot=1)
            self.assertIn("#mechanism", cm.exception.errors)


class AutoSelectionTests(unittest.TestCase):
    def cand(self, cid, cls="DIRECT", video="https://v", excluded=None, score=None):
        return {"id": cid, "video_url": video,
                "source_check": {"classification": cls, "excluded": excluded or []},
                "source_score": {"final_score": score, "action": "DEEP",
                                 "tags": ["money_signal_pass"],
                                 "components": {"money_signal_gate": {"pass": True, "hard_block": False}}}}

    def test_legacy_order_is_kept_and_unconfirmed_sources_are_skipped(self):
        items = [self.cand("a", "UNVERIFIED"), self.cand("b", "LOW"), self.cand("c", video=None),
                 self.cand("d", "HARD_EXCLUDE"), self.cand("e", "EXPANDED"), self.cand("f", "DIRECT")]
        chosen, skipped = select_candidate(items)
        self.assertEqual(chosen["id"], "e")   # first valid in legacy order, not re-ranked
        self.assertEqual(len(skipped), 4)

    def test_nothing_eligible_returns_none_with_reasons(self):
        chosen, skipped = select_candidate([self.cand("a", "UNVERIFIED")])
        self.assertIsNone(chosen)
        self.assertIn("UNVERIFIED", skipped[0])

    def test_excluded_hits_and_optional_score_floor(self):
        self.assertIn("hard-excluded", block_reason(self.cand("a", excluded=["weapon"])))
        self.assertEqual(block_reason(self.cand("a", score=0.8), min_final_score=0.5), "")
        self.assertIn("below", block_reason(self.cand("a", score=0.2), min_final_score=0.5))
        self.assertIn("below", block_reason(self.cand("a", score=None), min_final_score=0.5))


if __name__ == "__main__":
    unittest.main()
