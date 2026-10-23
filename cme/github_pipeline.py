"""GitHub-only CME pipeline up to the Layer 2 frame QA boundary.

Scope: FIND -> sampled-frame source analysis -> source-to-target atom transfer ->
image generation -> strict frame QA. Video generation / Layer 3 is intentionally out of scope.
"""
from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path
from urllib.request import Request, urlopen
from urllib.parse import urlsplit

from PIL import Image

from .offer import load_offer
from .qa import validate_product_references
from .radar import find as find_mod
from .radar.server import ROOT, RadarApp

OUT = Path("out").resolve()
WORK = Path("cme-live-work").resolve()


def _write_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")


def _download_photo(url: str, dest: Path) -> None:
    parsed = urlsplit(url)
    if parsed.scheme != "https" or not parsed.hostname:
        raise RuntimeError("Target product photo must be a direct HTTPS image URL.")
    req = Request(url, headers={"User-Agent": "Mozilla/5.0", "Accept": "image/*"})
    try:
        with urlopen(req, timeout=30) as response:
            content_type = response.headers.get_content_type()
            data = response.read(15 * 1024 * 1024 + 1)
    except Exception as exc:
        raise RuntimeError(f"Cannot download target product photo: {type(exc).__name__}: {exc}") from exc
    if len(data) > 15 * 1024 * 1024:
        raise RuntimeError("Target product photo exceeds 15 MB.")
    if not content_type.startswith("image/"):
        raise RuntimeError(f"Photo URL returned {content_type}, not an image.")
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_bytes(data)
    try:
        with Image.open(dest) as image:
            image.verify()
    except Exception as exc:
        dest.unlink(missing_ok=True)
        raise RuntimeError(f"Downloaded target photo is not a valid image: {exc}") from exc


def _prepare_offer() -> Path:
    photo_url = os.environ.get("TARGET_PHOTO_URL", "").strip()
    if not photo_url:
        raise RuntimeError("TARGET_PHOTO_URL is required for image production.")
    photo = OUT / "target_product_reference.jpg"
    _download_photo(photo_url, photo)
    offer = {
        "offer_id": os.environ.get("OFFER_ID", "fanttik-v8-mate-us-001").strip(),
        "name": os.environ.get("OFFER_NAME", "Fanttik V8 Mate Cordless Car Vacuum").strip(),
        "product_type": os.environ.get("PRODUCT_TYPE", "cordless handheld car vacuum").strip(),
        "verified_claims": [x.strip() for x in os.environ.get("VERIFIED_CLAIMS", "Cordless car vacuum").split(",") if x.strip()],
        "forbidden_claims": ["guaranteed", "medical", "cures"],
        "photos": ["target_product_reference.jpg"],
        "platform": os.environ.get("TARGET_PLATFORM", "TikTok").strip(),
        "geo": os.environ.get("FIND_COUNTRY", "US").strip().upper(),
        "product_category": "Physical Products",
        "confirmations": [],
    }
    if not offer["name"] or not offer["product_type"] or not offer["verified_claims"]:
        raise RuntimeError("Offer name, product type and at least one verified claim are required.")
    path = OUT / "live_offer.json"
    _write_json(path, offer)
    loaded = load_offer(path)
    issues = validate_product_references(loaded.photos)
    if issues:
        raise RuntimeError("Target-photo preflight blocked before provider calls: " + "; ".join(issues))
    return path


def _select_candidate(candidates: list[dict]) -> dict:
    # Preserve the legacy ranking order; automatically choose its highest-ranked usable video source.
    eligible = [c for c in candidates if c.get("video_url") and c.get("source_check", {}).get("excluded") is not True]
    if not eligible:
        raise RuntimeError("FIND returned no eligible TikTok candidate with a usable video URL. No Brain/image calls were made.")
    return eligible[0]


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    result = {
        "status": "RUNNING",
        "scope": "pre-Layer-3 only: FIND -> ANALYZE -> TRANSFER/ATOMS -> IMAGE -> FRAME_QA",
        "started_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "repository": os.getenv("GITHUB_REPOSITORY"),
        "commit": os.getenv("GITHUB_SHA"),
        "workflow_run_id": os.getenv("GITHUB_RUN_ID"),
        "vertical": "Physical Products",
        "find": {}, "selected_source": {}, "analysis": {}, "production": {}, "errors": [],
    }
    app = None
    try:
        for key in ("SOCIALCRAWL_API_KEY", "OPENROUTER_API_KEY", "HF_TOKEN"):
            if not os.environ.get(key):
                raise RuntimeError(f"Required Actions secret {key} is missing.")
        settings = ROOT / "settings.json"
        settings.write_text((ROOT / "settings.example.json").read_text(encoding="utf-8"), encoding="utf-8")
        offer_path = _prepare_offer()
        shots = int(os.environ.get("SHOTS", "1"))
        if not 1 <= shots <= 6:
            raise RuntimeError("SHOTS must be between 1 and 6.")
        find_params = {
            "country": os.environ.get("FIND_COUNTRY", "US"),
            "period": int(os.environ.get("FIND_PERIOD", "30")),
            "max_requests": int(os.environ.get("FIND_REQUESTS", "4")),
            "max_ads": int(os.environ.get("FIND_MAX_ADS", "3")),
            "keywords": [x.strip() for x in os.environ.get("FIND_KEYWORDS", "car vacuum").split(",") if x.strip()],
            "order_bys": ["for_you"],
        }
        discovery = find_mod.run_find(find_params)
        candidates = discovery.get("candidates", [])
        result["find"] = {
            "status": "OK", "request": discovery.get("request"), "stats": discovery.get("stats"),
            "candidate_count": len(candidates),
            "candidates": [
                {"id": c.get("id"), "brand": c.get("brand") or c.get("brand_name"), "title": c.get("title"),
                 "source_score": c.get("source_score"), "video_available": bool(c.get("video_url"),
                 ), "source_url": c.get("source_url"),
                 "source_check": c.get("source_check")}
                for c in candidates
            ],
        }
        if not candidates:
            raise RuntimeError("FIND completed without candidates; no Brain/image calls were made.")
        app = RadarApp(
            WORK / "radar", settings_path=settings, catalog_path=ROOT / "adapters.json",
            offer_path=offer_path, vertical_path=ROOT / "verticals" / "physical_goods_us.json",
            production_shots=shots,
        )
        app.store.ingest(candidates, "github-" + str(os.getenv("GITHUB_RUN_ID", "manual")), "found")
        selected = _select_candidate(candidates)
        cid = selected["id"]
        app.store.decide(cid, "shortlist", "Automatically selected the highest-ranked eligible video source for the one-click GitHub run.")
        result["selected_source"] = {
            "candidate_id": cid, "brand": selected.get("brand") or selected.get("brand_name"),
            "title": selected.get("title"), "source_url": selected.get("source_url"),
            "source_score": selected.get("source_score"),
            "evidence_scope": "legacy source-ranking evidence plus sampled frames; mechanism and transfer remain hypotheses",
        }
        analysis = app.do_analyze(cid)
        result["analysis"] = analysis
        if analysis.get("status") != "OK":
            raise RuntimeError("Source analysis did not return OK: " + str(analysis.get("status")) + " — " + str(analysis.get("detail", "")))
        production = app.do_produce(cid)
        result["production"] = production
        if production.get("status") != "OK" or production.get("exit_code") != 0:
            raise RuntimeError(
                "Frame production/QA did not PASS: status=" + str(production.get("status")) +
                ", exit_code=" + str(production.get("exit_code")) + ". No later stage was run."
            )
        verdicts = production.get("verdicts", [])
        if len(verdicts) < shots or any(str(v.get("overall", "")).upper() != "PASS" for v in verdicts):
            raise RuntimeError("Production did not provide a PASS receipt for every requested frame.")
        images = production.get("images", [])
        if len(images) < shots:
            raise RuntimeError(f"Expected {shots} generated frames; production report lists {len(images)}.")
        # Preserve honest provider/commercial status. QA PASS alone does not imply commercial readiness.
        commercial = [v.get("commercial_status") for v in verdicts]
        result["status"] = "READY_FOR_REVIEW" if all(x == "CANDIDATE" for x in commercial) else "TEST_ONLY"
        result["frame_qa"] = {
            "status": "PASS", "requested_shots": shots, "passed_receipts": len(verdicts),
            "images": images, "verdicts": verdicts,
            "commercial_status": "CANDIDATE" if result["status"] == "READY_FOR_REVIEW" else "TEST_ONLY",
        }
    except Exception as exc:
        result["status"] = "FAILED"
        result["errors"].append({"type": type(exc).__name__, "message": str(exc)})
        print("CME pre-Layer-3 pipeline failed:", type(exc).__name__, str(exc), file=sys.stderr)
    finally:
        if app is not None:
            try:
                app.store.close()
            except Exception:
                pass
        result["finished_at"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        _write_json(OUT / "live_result.json", result)
        summary_path = os.environ.get("GITHUB_STEP_SUMMARY", "").strip()
        if summary_path:
            with Path(summary_path).open("a", encoding="utf-8") as f:
                f.write("# CME Control Center — pre-Layer-3\n\n")
                f.write(f"- **Status:** {result['status']}\n")
                f.write(f"- **Run:** {result.get('workflow_run_id')}\n")
                f.write("- **Scope:** FIND → ANALYZE → TRANSFER/ATOMS → IMAGE → FRAME QA. Layer 3 is out of scope.\n")
                if result.get("selected_source"):
                    source = result["selected_source"]
                    f.write(f"- **Selected source:** {source.get('brand') or 'unknown brand'} — {source.get('title') or 'untitled'}\n")
                    if source.get("source_url"):
                        f.write(f"- **Source ad:** {source['source_url']}\n")
                if result.get("frame_qa"):
                    f.write(f"- **Frame QA:** PASS ({result['frame_qa']['passed_receipts']} receipts)\n")
                    f.write(f"- **Commercial status:** {result['frame_qa']['commercial_status']}\n")
                if result["errors"]:
                    f.write("\n## Blockers\n")
                    for error in result["errors"]:
                        f.write(f"- {error['type']}: {error['message']}\n")
                f.write("\nDownload **cme-control-center-output** for the report, generated frames, atom manifests, receipts, and call logs.\n")
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0 if result["status"] in {"READY_FOR_REVIEW", "TEST_ONLY"} else 1


if __name__ == "__main__":
    raise SystemExit(main())
