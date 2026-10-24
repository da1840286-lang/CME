"""Regression tests for campaign-scoped branding across Brain, image prompt and QA."""
import json
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path
from unittest.mock import patch

from PIL import Image, ImageDraw
from cme.brain import build_request
from cme.branding import NO_BRANDING, TARGET_BRAND_ALLOWED, allowed_brand_tokens, prepare_unbranded_reference, qa_branding_rule
from cme.compiler import compile_image_prompt
from cme.offer import load_offer
from cme.qa import CheckResult, SYSTEM_NO_TEXT_ID, make_qa_checks
from cme.run import produce_shot
from cme.state import new_run
from tests.helpers import A, png_bytes

ROOT = Path(__file__).resolve().parent.parent
EX = ROOT / "examples" / "offer_example.json"

class CampaignBrandingPolicyTests(unittest.TestCase):
    def test_strict_mode_never_whitelists_target_name_but_brand_mode_does(self):
        self.assertEqual(allowed_brand_tokens(NO_BRANDING, ("Fanttik",), "Fanttik V8 Mate Vacuum", "cordless vacuum"), ())
        self.assertEqual(set(allowed_brand_tokens(TARGET_BRAND_ALLOWED, (), "Fanttik V8 Mate Cordless Car Vacuum",
                                                  "cordless handheld car vacuum")), {"fanttik", "mate"})
        self.assertEqual(set(allowed_brand_tokens(TARGET_BRAND_ALLOWED, ("ExampleBrand",), "Fanttik Vacuum", "vacuum")),
                         {"examplebrand"})

    def test_qa_rules_separate_strict_and_target_brand_modes(self):
        strict = qa_branding_rule(NO_BRANDING)
        branded = qa_branding_rule(TARGET_BRAND_ALLOWED, ("ExampleBrand",))
        self.assertIn("Product-native markings are NOT exempt", strict)
        self.assertIn("ExampleBrand", branded)
        self.assertIn("source-ad", branded)

    def test_brain_and_generation_prompt_follow_campaign_mode(self):
        offer = load_offer(EX)
        strict = replace(offer, branding_policy=NO_BRANDING)
        brand = replace(offer, branding_policy=TARGET_BRAND_ALLOWED, approved_brand_marks=("ExampleBrand",))
        strict_rules = " ".join(build_request(strict, "brief", 1)["rules"])
        brand_rules = " ".join(build_request(brand, "brief", 1)["rules"])
        strict_prompt = compile_image_prompt([A("product_main", "Black handheld vacuum body is visible", kind="product")],
                                             1, branding_policy=NO_BRANDING).text
        brand_prompt = compile_image_prompt([A("product_main", "Black handheld vacuum body is visible", kind="product")],
                                            1, branding_policy=TARGET_BRAND_ALLOWED,
                                            approved_brand_marks=("ExampleBrand",)).text
        self.assertIn("even when present on supplied target-product photos", strict_rules)
        self.assertIn("ExampleBrand", brand_rules)
        self.assertIn("SYSTEM-OWNED BRANDING POLICY", strict_prompt)
        self.assertIn("clean, unmarked material", strict_prompt)
        self.assertIn("ExampleBrand", brand_prompt)

    def test_detected_brand_text_fails_in_strict_mode(self):
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / "frame.png"
            path.write_bytes(png_bytes())
            seen = {}
            def judge(_path, atoms):
                seen["rule"] = {a.id: a.text for a in atoms}[SYSTEM_NO_TEXT_ID]
                return {a.id: {"pass": True, "evidence": "No readable text or branding"} for a in atoms}
            qa = make_qa_checks([A("product_main", "Black handheld vacuum is visible", kind="product")], 1,
                               judge_fn=judge, ocr=lambda _p: "Fanttik", branding_policy=NO_BRANDING, allowed_text=())
            rows = {x.check_id:x for x in qa(path)}
            self.assertEqual(rows["no_text"].status, "FAIL")
            self.assertTrue(rows["no_text"].required)
            self.assertIn("Product-native markings are NOT exempt", seen["rule"])

    def test_empty_ocr_does_not_bypass_required_visual_judge(self):
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / "frame.png"
            path.write_bytes(png_bytes())
            def judge(_path, atoms):
                return {a.id: {"pass": a.id != SYSTEM_NO_TEXT_ID,
                               "evidence": "branding violation" if a.id == SYSTEM_NO_TEXT_ID else "visible"}
                        for a in atoms}
            qa = make_qa_checks([A("product_main", "Black handheld vacuum is visible", kind="product")], 1,
                               judge_fn=judge, ocr=lambda _p: "", branding_policy=NO_BRANDING)
            rows = {x.check_id:x for x in qa(path)}
            self.assertEqual(rows["no_text"].status, "NOT_RUN")
            self.assertFalse(rows["no_text"].required)
            self.assertTrue(rows[SYSTEM_NO_TEXT_ID].required)
            self.assertEqual(rows[SYSTEM_NO_TEXT_ID].status, "FAIL")

    def test_reference_text_is_masked_but_original_is_preserved(self):
        with tempfile.TemporaryDirectory() as d:
            source = Path(d) / "product.jpg"
            out = Path(d) / "out"
            image = Image.new("RGB", (160, 100), (15,15,15))
            ImageDraw.Draw(image).text((25,35), "Fanttik", fill=(245,245,245))
            image.save(source)
            response = {"text":["Fanttik"],"conf":["96"],"left":["25"],"top":["35"],"width":["65"],"height":["14"]}
            with patch("pytesseract.image_to_data", return_value=response):
                prepared = prepare_unbranded_reference(source, out / "clean.png")
            self.assertEqual(prepared["status"], "TEXT_REGIONS_MASKED")
            with Image.open(source) as original, Image.open(prepared["path"]) as clean:
                self.assertNotEqual(list(original.getdata()), list(clean.getdata()))

    def test_generator_receives_scrubbed_ref_and_receipt_records_policy(self):
        with tempfile.TemporaryDirectory() as d:
            source = Path(d) / "target.png"
            image = Image.new("RGB", (160,100), (15,15,15))
            ImageDraw.Draw(image).text((25,35), "Fanttik", fill=(245,245,245))
            image.save(source)
            offer = replace(load_offer(EX), photos=(str(source),), branding_policy=NO_BRANDING)
            refs, prompts = [], []
            def fake_ocr(_image, *, config, output_type):
                return {"text":["Fanttik"],"conf":["96"],"left":["25"],"top":["35"],"width":["65"],"height":["14"]}
            def provider(prompt, images):
                prompts.append(prompt); refs.extend(images); return png_bytes()
            with patch("pytesseract.image_to_data", side_effect=fake_ocr):
                receipt, _, _ = produce_shot(
                    offer=offer, atoms=[A("product_main","Black handheld vacuum body is visible",kind="product")],
                    shot=1, contract_id="branding-test-contract", state=new_run("branding-test-run"),
                    out_dir=Path(d)/"out", image_provider=provider,
                    qa_checks=lambda _p:[CheckResult("file","PASS",True,"code")])
            self.assertEqual(len(refs), 1)
            self.assertNotEqual(Path(refs[0]).resolve(), source.resolve())
            self.assertTrue(Path(refs[0]).is_file())
            self.assertIn("no_branding", prompts[0])
            data=json.loads((Path(d)/"out"/"receipts"/f"{receipt.receipt_id}.json").read_text())
            self.assertEqual(data["branding_policy"], NO_BRANDING)
            self.assertEqual(data["image_inputs"][0]["role"], "target_product_photo_unbranded_reference")
            self.assertEqual(data["image_inputs"][0]["preparation"]["status"], "TEXT_REGIONS_MASKED")
            self.assertEqual(data["image_inputs"][0]["source_filename"], "target.png")

if __name__ == "__main__":
    unittest.main()
