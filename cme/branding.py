"""Campaign-scoped branding rules shared by Brain, generation and QA."""
from __future__ import annotations
import re
from pathlib import Path

NO_BRANDING = "no_branding"
TARGET_BRAND_ALLOWED = "target_brand_allowed"
BRANDING_POLICIES = (NO_BRANDING, TARGET_BRAND_ALLOWED)

def _marks(values):
    if not isinstance(values, (list, tuple)):
        return ()
    return tuple(str(v).strip() for v in values if isinstance(v, str) and v.strip())

def _tokens(text):
    return {t.lower() for t in re.findall(r"[^\W_]+", str(text or "")) if len(t) >= 3}

def allowed_brand_tokens(policy, approved_brand_marks=(), offer_name="", product_type=""):
    """OCR allowlist for the campaign mode; strict mode never whitelists the target name."""
    if policy == NO_BRANDING:
        return ()
    if policy != TARGET_BRAND_ALLOWED:
        raise ValueError(f"unsupported branding_policy {policy!r}")
    explicit = _marks(approved_brand_marks)
    tokens = set().union(*(_tokens(m) for m in explicit)) if explicit else (_tokens(offer_name) - _tokens(product_type))
    return tuple(sorted(tokens))

def branding_prompt_instruction(policy, approved_brand_marks=()):
    """System-owned instruction included in the exact receipt-bound provider prompt."""
    if policy == NO_BRANDING:
        return (
            "SYSTEM-OWNED BRANDING POLICY (no_branding; non-negotiable): Show NO logos, brand names, "
            "wordmarks, model names, labels, branding emblems, readable lettering or pseudo-text anywhere, "
            "including marks printed, painted, embossed or engraved on the product itself. This overrides "
            "every source/target reference and conflicting atom. Use product references only for silhouette, "
            "proportions, colors, materials, controls and non-text construction. Render marked surfaces as "
            "clean, unmarked material; do not reproduce or replace markings with symbols."
        )
    if policy == TARGET_BRAND_ALLOWED:
        marks = _marks(approved_brand_marks)
        allow = ("Explicitly approved target-brand marks: " + ", ".join(marks) + ". ") if marks else (
            "Target-product branding may appear only if already visible on the original target reference and "
            "consistent with the OfferCard product name/model. "
        )
        return ("SYSTEM-OWNED BRANDING POLICY (target_brand_allowed; non-negotiable): " + allow +
                "Never invent text, pseudo-lettering, logos, brand names or labels beyond the explicitly approved target marks. "
                "Never copy source-ad, competitor or unrelated branding. No captions, subtitles, UI, watermarks or promotional text.")
    raise ValueError(f"unsupported branding_policy {policy!r}")

def brain_branding_rule(policy, approved_brand_marks=()):
    marks = _marks(approved_brand_marks)
    if policy == NO_BRANDING:
        return (
            "CAMPAIGN BRANDING POLICY: no_branding. Never invent text, pseudo-lettering, logos, brand names or labels. "
            "Never reproduce those marks, even when present on supplied target-product photos. Preserve the target "
            "product's non-text shape, proportions, colors, materials and construction; require clean, unmarked "
            "product surfaces. This overrides reference-photo markings."
        )
    if policy == TARGET_BRAND_ALLOWED:
        allow = ("Only these explicitly approved target-brand marks may be shown: " + ", ".join(marks) + ". ") if marks else (
            "Only target-product markings already visible in the target reference and consistent with the "
            "OfferCard name/model may be shown. "
        )
        return ("CAMPAIGN BRANDING POLICY: target_brand_allowed. " + allow +
                "Never invent text, pseudo-lettering, logos, brand names or labels beyond the approved target marks. "
                "Never copy source-ad/competitor branding. Do not add text overlays, captions, subtitles, UI, "
                "watermarks or promotional wording.")
    raise ValueError(f"unsupported branding_policy {policy!r}")

def qa_branding_rule(policy, approved_brand_marks=()):
    marks = _marks(approved_brand_marks)
    if policy == NO_BRANDING:
        return (
            "STRICT NO-BRANDING CHECK: FAIL if any logo, brand name, wordmark, model name, product label, "
            "branding emblem, readable letters/numbers or pseudo-text is visible anywhere, including lettering "
            "physically printed, painted, embossed or engraved on the product. Product-native markings are NOT "
            "exempt. The original reference may contain branding; its presence is not permission. Judge actual "
            "pixels. Product identity is judged from non-text shape, proportions, colors, materials, controls and construction."
        )
    if policy == TARGET_BRAND_ALLOWED:
        allow = ("The only approved target-brand text/marks are: " + ", ".join(marks) + ". ") if marks else (
            "Only branding already visible on the original target-product reference and consistent with the "
            "OfferCard name/model may be retained. "
        )
        return ("TARGET-BRAND-ONLY CHECK: No added captions, subtitles, UI, watermarks or promotional text. " +
                allow + "Target markings must appear on the actual product and match its reference. FAIL on "
                "source-ad, competitor, unrelated or invented branding, and unapproved readable text.")
    raise ValueError(f"unsupported branding_policy {policy!r}")

def prepare_unbranded_reference(source: str | Path, destination: str | Path):
    """Mask OCR-detected lettering in the image-provider input; retain original for identity QA."""
    source, destination = Path(source), Path(destination)
    try:
        import pytesseract
        from PIL import Image, ImageFilter
    except Exception as exc:
        return {"path": str(source), "status": "OCR_UNAVAILABLE", "masked_regions": 0, "error_type": type(exc).__name__}
    try:
        with Image.open(source) as opened:
            image = opened.convert("RGB")
            width, height = image.size
            boxes = []
            for psm in (3, 11, 12):
                data = pytesseract.image_to_data(image, config=f"--psm {psm}", output_type=pytesseract.Output.DICT)
                fields = [data.get(k, []) for k in ("text", "conf", "left", "top", "width", "height")]
                for i, word in enumerate(fields[0]):
                    if not re.search(r"[A-Za-z0-9]{2,}", str(word or "")):
                        continue
                    try:
                        conf, x, y, w, h = float(fields[1][i]), int(fields[2][i]), int(fields[3][i]), int(fields[4][i]), int(fields[5][i])
                    except (IndexError, TypeError, ValueError):
                        continue
                    if conf < 55 or w < 1 or h < 1:
                        continue
                    pad = max(3, int(max(w, h) * 0.4))
                    box = (max(0, x-pad), max(0, y-pad), min(width, x+w+pad), min(height, y+h+pad))
                    if box[2] > box[0] and box[3] > box[1]:
                        boxes.append(box)
            merged = []
            for box in sorted(boxes, key=lambda b: (b[1], b[0])):
                for j, prior in enumerate(merged):
                    if not (box[2] < prior[0] or prior[2] < box[0] or box[3] < prior[1] or prior[3] < box[1]):
                        merged[j] = (min(box[0], prior[0]), min(box[1], prior[1]), max(box[2], prior[2]), max(box[3], prior[3]))
                        break
                else:
                    merged.append(box)
            destination.parent.mkdir(parents=True, exist_ok=True)
            if not merged:
                image.save(destination, format="PNG")
                return {"path": str(destination), "status": "NO_CONFIDENT_TEXT_FOUND", "masked_regions": 0}
            largest = max(max(b[2]-b[0], b[3]-b[1]) for b in merged)
            radius = max(4, min(28, int(largest * 0.30)))
            blurred = image.filter(ImageFilter.GaussianBlur(radius=radius))
            for box in merged:
                image.paste(blurred.crop(box), box)
            image.save(destination, format="PNG")
            return {"path": str(destination), "status": "TEXT_REGIONS_MASKED", "masked_regions": len(merged)}
    except Exception as exc:
        kind = "OCR_UNAVAILABLE" if "tesseract" in str(exc).lower() else "OCR_FAILED"
        return {"path": str(source), "status": kind, "masked_regions": 0, "error_type": type(exc).__name__}
