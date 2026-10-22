import json, tempfile, unittest
from pathlib import Path
from cme.atoms import ATOM_SCHEMA, MAX_TEXT, check_claims, parse_atoms
from cme.compiler import BudgetError, compile_image_prompt, compile_motion_prompt
from cme.offer import OfferError, load_offer
from tests.helpers import A

EX = Path(__file__).resolve().parent.parent / "examples" / "offer_example.json"


class OfferTests(unittest.TestCase):
    def test_example_loads(self):
        o = load_offer(EX)
        self.assertEqual(o.offer_id, "example-001")

    def test_missing_photo_and_claims_rejected(self):
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "o.json"
            p.write_text(json.dumps({"offer_id": "x", "name": "n", "product_type": "t",
                                     "verified_claims": [], "photos": []}))
            with self.assertRaises(OfferError) as cm:
                load_offer(p)
            self.assertIn("verified_claims", str(cm.exception))
            self.assertIn("photo", str(cm.exception))

    def test_long_claim_rejected(self):
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "o.json"
            p.write_text(json.dumps({"offer_id": "x", "name": "n", "product_type": "t",
                                     "verified_claims": ["c" * 121], "photos": ["a.png"]}))
            with self.assertRaises(OfferError):
                load_offer(p, check_files=False)


class AtomTests(unittest.TestCase):
    def good(self, **kw):
        d = {"id": "a_1", "kind": "scene", "text": "A kitchen counter", "priority": "MUST",
             "channel": "image", "shot": 1}
        d.update(kw)
        return d

    def test_schema_carries_length_limit(self):
        item = ATOM_SCHEMA["properties"]["atoms"]["items"]["properties"]["text"]
        self.assertEqual(item["maxLength"], MAX_TEXT)

    def test_parse_collects_per_atom_errors(self):
        raw = [self.good(), self.good(id="b_2", text="x" * 121), self.good(id="c_3", priority="HIGH"),
               self.good(id="a_1")]
        r = parse_atoms(raw)
        self.assertEqual(r.atoms, [])  # a_1 is used twice: every occurrence is rejected
        self.assertEqual(set(r.errors), {"b_2", "c_3", "a_1"})
        self.assertIn("2 times", r.errors["a_1"][0])
        self.assertIn("split or shorten", r.errors["b_2"][0])

    def test_claims_verified_verbatim_only(self):
        o = load_offer(EX)
        ok = parse_atoms([self.good(kind="claim", text="Reusable and dishwasher safe")]).atoms
        bad = parse_atoms([self.good(id="b_2", kind="claim", text="Lasts a lifetime")]).atoms
        forb = parse_atoms([self.good(id="c_3", text="Guaranteed fresh food")]).atoms
        self.assertEqual(check_claims(ok, o), {})
        self.assertIn("UNVERIFIED_CLAIM", check_claims(bad, o)["b_2"][0])
        self.assertIn("FORBIDDEN_CLAIM", check_claims(forb, o)["c_3"][0])


class CompilerTests(unittest.TestCase):
    def test_text_is_only_atoms_no_engine_template(self):
        atoms = [A("a", "first line"), A("b", "second line", "SHOULD")]
        c = compile_image_prompt(atoms, 1)
        self.assertEqual(c.text, "first line\nsecond line")
        self.assertEqual(c.lost_line(), "потеряно: нет")

    def test_drops_may_then_should_never_must(self):
        atoms = [A("m1", "M" * 40), A("s1", "S" * 40, "SHOULD"), A("y1", "Y" * 40, "MAY"), A("m2", "N" * 40)]
        c = compile_image_prompt(atoms, 1, limit=85)
        self.assertEqual(c.included, ("m1", "m2"))
        self.assertEqual(c.dropped, (("y1", "MAY"), ("s1", "SHOULD")))
        self.assertIn("y1 (MAY)", c.lost_line())

    def test_may_dropped_before_should(self):
        atoms = [A("s1", "S" * 40, "SHOULD"), A("y1", "Y" * 40, "MAY")]
        c = compile_image_prompt(atoms, 1, limit=45)
        self.assertEqual(c.included, ("s1",))

    def test_must_overflow_stops_with_longest_first(self):
        atoms = [A("short", "a" * 10), A("long", "b" * 60), A("mid", "c" * 30)]
        with self.assertRaises(BudgetError) as cm:
            compile_image_prompt(atoms, 1, limit=50)
        self.assertEqual(cm.exception.must_ids, ["long", "mid", "short"])

    def test_reserve_counts_against_limit(self):
        atoms = [A("a", "x" * 50)]
        with self.assertRaises(BudgetError):
            compile_image_prompt(atoms, 1, limit=60, reserve=20)

    def test_only_this_shot_and_channel(self):
        atoms = [A("a", "one", shot=1), A("b", "two", shot=2), A("c", "move", ch="motion")]
        self.assertEqual(compile_image_prompt(atoms, 1).text, "one")
        self.assertEqual(compile_motion_prompt(atoms).text, "Shot 1: move")


if __name__ == "__main__":
    unittest.main()
