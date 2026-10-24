import unittest
from pathlib import Path

from cme.brain import atom_budget_errors
from cme.radar.source_loop import run_source_loop


def atom(atom_id, text, shot=1, kind="scene", priority="MUST", channel="image"):
    return {"id": atom_id, "text": text, "shot": shot, "kind": kind, "priority": priority, "channel": channel}


class FrameBudgetTests(unittest.TestCase):
    def test_redundant_requirements_are_rejected_but_mechanism_is_protected(self):
        items = [
            atom("mech_problem_debris", "car seat and floor mat covered with visible crumbs and dust"),
            atom("scene_problem", "car interior with crumbs and dust on seats and floor mats"),
            atom("action_problem_debris", "visible crumbs and dust scattered on car seat and floor mat"),
            atom("product_vacuum", "handheld cordless vacuum, compact black body"),
        ]
        errors = atom_budget_errors(items, 1)
        self.assertEqual(set(errors), {"scene_problem", "action_problem_debris"})
        self.assertIn("FRAME_REDUNDANT", errors["scene_problem"][0])

    def test_one_layout_per_shot(self):
        items = [
            atom("comp_split", "left half dirty seat, right half same seat clean, sharp boundary", kind="scene"),
            atom("composition_side_view", "side view with vacuum body fully visible", kind="composition"),
        ]
        errors = atom_budget_errors(items, 1)
        self.assertEqual(set(errors), {"composition_side_view"})
        self.assertIn("FRAME_LAYOUT_CONFLICT", errors["composition_side_view"][0])

    def test_must_ceiling_is_reported_per_shot(self):
        words = "red blue green amber violet cyan olive pink brown grey teal gold".split()
        items = [atom(f"obj_{i}", f"{words[i]} {words[(i+3)%12]} item number {i} on shelf {words[(i+5)%12]}")
                 for i in range(8)]
        errors = atom_budget_errors(items, 1)
        self.assertIn("#must_budget_1", errors)

    def test_different_shots_have_independent_budgets(self):
        items = [atom(f"s1_{i}", f"alpha{i} beta{i} gamma{i} delta{i} epsilon{i}") for i in range(5)]
        items += [atom(f"s2_{i}", f"zeta{i} eta{i} theta{i} iota{i} kappa{i}", shot=2) for i in range(5)]
        self.assertEqual(atom_budget_errors(items, 2), {})


class SourceLoopTests(unittest.TestCase):
    def run_loop(self, ids, *, statuses=None, reports=None, limit=3, **kwargs):
        statuses, reports = statuses or {}, reports or {}
        order = []

        def analyze(source):
            cid = source["id"]
            order.append(("analyze", cid))
            return {"candidate_id": cid, "status": statuses.get(cid, "OK")}, {"candidate_id": cid}

        def produce(source, payload):
            cid = source["id"]
            order.append(("produce", cid))
            return reports.get(cid, {"failure_class": "DELIVERED_PASS", "exit_code": 0})

        result = run_source_loop([{"id": cid} for cid in ids], limit=limit, analyze=analyze, produce=produce, **kwargs)
        return result, order

    def test_transfer_rejection_moves_to_next_ranked_candidate(self):
        result, order = self.run_loop(
            ["a", "b", "c"],
            reports={"a": {"failure_class": "TRANSFER_REJECTED", "failure_detail": "mechanism mismatch"},
                     "b": {"failure_class": "DELIVERED_PASS"}},
        )
        self.assertEqual(result.selected["id"], "b")
        self.assertEqual([r["candidate_id"] for r in result.rejections], ["a"])
        self.assertEqual(order, [("analyze", "a"), ("produce", "a"), ("analyze", "b"), ("produce", "b")])

    def test_all_candidates_rejected_returns_last_rejection(self):
        rejection = {"failure_class": "TRANSFER_REJECTED", "failure_detail": "not transferable"}
        result, _ = self.run_loop(["a", "b", "c"], reports={"a": rejection, "b": rejection, "c": rejection})
        self.assertEqual(result.selected["id"], "c")
        self.assertEqual(result.report["failure_class"], "TRANSFER_REJECTED")
        self.assertEqual(len(result.rejections), 3)

    def test_manual_choice_is_never_replaced(self):
        result, order = self.run_loop(
            ["a", "b"], reports={"a": {"failure_class": "TRANSFER_REJECTED"}, "b": {"failure_class": "DELIVERED_PASS"}},
            allow_transfer_fallback=False,
        )
        self.assertEqual(result.selected["id"], "a")
        self.assertEqual(order, [("analyze", "a"), ("produce", "a")])

    def test_provider_or_visual_qa_failure_does_not_trigger_transfer_fallback(self):
        for failure in ("INFRASTRUCTURE_ERROR", "VISUAL_QA_FAILED", "QA_UNVERIFIED", "BRAIN_OUTPUT_INVALID"):
            result, order = self.run_loop(
                ["a", "b"], reports={"a": {"failure_class": failure}, "b": {"failure_class": "DELIVERED_PASS"}}
            )
            self.assertEqual(result.selected["id"], "a", failure)
            self.assertEqual(order, [("analyze", "a"), ("produce", "a")], failure)

    def test_retryable_analysis_failure_moves_on_but_internal_error_stops(self):
        result, _ = self.run_loop(["a", "b"], statuses={"a": "VIDEO_UNAVAILABLE"})
        self.assertEqual(result.selected["id"], "b")
        result, order = self.run_loop(["a", "b"], statuses={"a": "INTERNAL_ERROR"})
        self.assertIsNone(result.selected)
        self.assertEqual(order, [("analyze", "a")])

    def test_limit_counts_tried_candidates_and_skips_do_not_consume_it(self):
        result, order = self.run_loop(
            ["skip", "a", "b"], limit=2,
            skip_reason=lambda source: "no video" if source["id"] == "skip" else "",
            reports={"a": {"failure_class": "TRANSFER_REJECTED"}, "b": {"failure_class": "DELIVERED_PASS"}},
        )
        self.assertEqual([item for item in order if item[0] == "produce"], [("produce", "a"), ("produce", "b")])
        self.assertEqual(result.skipped, ["skip: no video"])

    def test_workflow_is_wired_to_testable_source_loop(self):
        workflow = (Path(__file__).resolve().parents[1] / ".github" / "workflows" / "live-smoke.yml").read_text(encoding="utf-8")
        self.assertIn("from cme.radar.source_loop import run_source_loop", workflow)
        self.assertIn("loop_result = run_source_loop(", workflow)
        self.assertIn('allow_transfer_fallback=(selection_mode != "manual_from_prior_find")', workflow)


if __name__ == "__main__":
    unittest.main()
