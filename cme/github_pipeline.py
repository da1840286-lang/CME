"""GitHub-only end-to-end CME run: FIND -> source analysis -> atoms/images/QA -> Magic Hour video."""
from __future__ import annotations

import hashlib
import json
import os
import shutil
import subprocess
import sys
import time
import traceback
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.parse import urlsplit
from urllib.request import Request, urlopen

from PIL import Image

from .offer import load_offer
from .qa import validate_product_references
from .radar import find as find_mod
from .radar.server import ROOT, RadarApp

OUT = Path("out").resolve()
WORK = Path("cme-live-work").resolve()


def _json(path: Path, value: dict) -> None:
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
    except (OSError, HTTPError, URLError) as exc:
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


def _make_offer() -> Path:
    OUT.mkdir(parents=True, exist_ok=True)
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
    _json(path, offer)
    loaded = load_offer(path)
    issues = validate_product_references(loaded.photos)
    if issues:
        raise RuntimeError("Target-photo preflight blocked before provider calls: " + "; ".join(issues))
    return path


def _choose_candidate(candidates: list[dict]) -> dict:
    eligible = [c for c in candidates if c.get("video_url") and c.get("source_check", {}).get("excluded") is not True]
    if not eligible:
        raise RuntimeError("FIND returned no eligible TikTok candidate with a usable video URL. No Brain/image/video calls were made.")
    # discover_and_rank already returns source-score order. Keep that order: do not invent a second ranking.
    return eligible[0]


def _make_video(image_paths: list[Path], video_dir: Path) -> dict:
    if not os.environ.get("MAGIC_HOUR_API_KEY"):
        raise RuntimeError("MAGIC_HOUR_API_KEY is missing. Add it to repository Actions secrets before video production.")
    try:
        from magic_hour import Client
    except ImportError as exc:
        raise RuntimeError("Magic Hour SDK is not installed in the runner.") from exc

    video_dir.mkdir(parents=True, exist_ok=True)
    client = Client(token=os.environ["MAGIC_HOUR_API_KEY"])
    clips: list[Path] = []
    jobs: list[dict] = []
    for index, image_path in enumerate(image_paths, start=1):
        clip_dir = video_dir / f"clip_{index:02d}"
        clip_dir.mkdir(parents=True, exist_ok=True)
        prompt = (
            "Create a vertical performance-marketing product-ad clip from this approved frame. "
            "Use restrained, physically plausible camera motion and subtle product movement. "
            "Keep the exact vacuum design, proportions, colors and visible details unchanged. "
            "No new text, logos, hands, accessories or product claims. Add relevant natural sound "
            "and a subtle clean sound bed; do not add spoken claims or music that masks the product."
        )
        result = client.v1.image_to_video.generate(
            assets={"image_file_path": str(image_path)},
            style={"prompt": prompt},
            end_seconds=5.0,
            name=f"CME {os.environ.get('GITHUB_RUN_ID', 'run')} shot {index}",
            model=os.environ.get("MAGIC_HOUR_MODEL", "ltx-2.5"),
            resolution=os.environ.get("MAGIC_HOUR_RESOLUTION", "480p"),
            audio=True,
            wait_for_completion=True,
            download_outputs=True,
            download_directory=str(clip_dir),
        )
        status = str(getattr(result, "status", "")).lower()
        downloaded = getattr(result, "downloaded_paths", None) or getattr(result, "downloadedPaths", None) or []
        if isinstance(downloaded, (str, Path)):
            downloaded = [downloaded]
        files = [Path(p) for p in downloaded if Path(p).is_file() and Path(p).stat().st_size > 0]
        if status != "complete" or not files:
            err = getattr(result, "error", None)
            detail = getattr(err, "message", str(err)) if err else "no downloaded output"
            raise RuntimeError(f"Magic Hour clip {index} did not complete successfully (status={status or 'unknown'}): {detail}")
        clip = next((p for p in files if p.suffix.lower() == ".mp4"), files[0])
        clips.append(clip)
        jobs.append({"shot": index, "status": status, "file": str(clip.relative_to(OUT)), 
                     "project_id": getattr(result, "id", None), "credits_charged": getattr(result, "credits_charged", None)})
    if len(clips) != len(image_paths):
        raise RuntimeError("Video clip count does not match the number of QA-approved frames.")

    concat_file = video_dir / "concat.txt"
    concat_file.write_text("".join("file '" + str(p.resolve()).replace("'", "'\\''") + "'\n" for p in clips), encoding="utf-8")
    final_video = video_dir / "cme_final_ad.mp4"
    command = [
        "ffmpeg", "-y", "-v", "error", "-f", "concat", "-safe", "0", "-i", str(concat_file),
        "-map", "0:v:0", "-map", "0:a:0", "-c:v", "libx264", "-preset", "medium",
        "-pix_fmt", "yuv420p", "-c:a", "aac", "-b:a", "128k", "-movflags", "+faststart", str(final_video),
    ]
    result = subprocess.run(command, capture_output=True, text=True, timeout=180)
    if result.returncode != 0 or not final_video.is_file() or final_video.stat().st_size == 0:
        raise RuntimeError("FFmpeg assembly failed: " + (result.stderr or "")[-1200:])
    probe = subprocess.run(
        ["ffprobe", "-v", "error", "-show_streams", "-show_format", "-of", "json", str(final_video)],
        capture_output=True, text=True, timeout=30,
    )
    if probe.returncode != 0:
        raise RuntimeError("Final video QA could not inspect the MP4: " + (probe.stderr or "")[-500:])
    data = json.loads(probe.stdout)
    streams = data.get("streams", [])
    video_stream = next((s for s in streams if s.get("codec_type") == "video"), None)
    audio_stream = next((s for s in streams if s.get("codec_type") == "audio"), None)
    duration = float((data.get("format") or {}).get("duration") or 0)
    if not video_stream or not audio_stream or duration <= 0:
        raise RuntimeError("Final video QA failed: expected non-empty video and audio streams. Silent or incomplete video is not accepted.")
    if int(video_stream.get("height") or 0) <= int(video_stream.get("width") or 0):
        raise RuntimeError("Final video QA failed: output is not vertical.")
    return {
        "status": "PASS", "path": str(final_video.relative_to(OUT)), "duration_seconds": round(duration, 2),
        "width": video_stream.get("width"), "height": video_stream.get("height"),
        "video_codec": video_stream.get("codec_name"), "audio_codec": audio_stream.get("codec_name"),
        "bytes": final_video.stat().st_size, "clips": jobs,
    }


def main() -> int:
    started = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    result = {
        "status": "RUNNING", "started_at": started, "repository": os.getenv("GITHUB_REPOSITORY"),
        "commit": os.getenv("GITHUB_SHA"), "workflow_run_id": os.getenv("GITHUB_RUN_ID"),
        "vertical": "Physical Products", "find": {}, "selected_source": {}, "analysis": {},
        "production": {}, "video": {}, "errors": [],
    }
    app = None
    try:
        for key in ("SOCIALCRAWL_API_KEY", "OPENROUTER_API_KEY", "HF_TOKEN", "MAGIC_HOUR_API_KEY"):
            if not os.environ.get(key):
                raise RuntimeError(f"Required Actions secret {key} is missing.")
        settings = ROOT / "settings.json"
        settings.write_text((ROOT / "settings.example.json").read_text(encoding="utf-8"), encoding="utf-8")
        offer_path = _make_offer()
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
        result["find"] = {"status": "OK", "request": discovery.get("request"), "stats": discovery.get("stats"),
                          "candidate_count": len(candidates),
                          "candidates": [{"id": c.get("id"), "brand": c.get("brand") or c.get("brand_name"),
                                          "title": c.get("title"), "source_score": c.get("source_score"),
                                          "video_available": bool(c.get("video_url")), "source_url": c.get("source_url")}
                                         for c in candidates]}
        if not candidates:
            raise RuntimeError("FIND completed without candidates; no provider calls after discovery were made.")
        app = RadarApp(
            WORK / "radar", settings_path=settings, catalog_path=ROOT / "adapters.json",
            offer_path=offer_path, vertical_path=ROOT / "verticals" / "physical_goods_us.json",
            production_shots=shots,
        )
        app.store.ingest(candidates, "github-" + str(os.getenv("GITHUB_RUN_ID", "manual")), "found")
        selected = _choose_candidate(candidates)
        cid = selected["id"]
        app.store.decide(cid, "shortlist", "Automatically selected highest-ranked eligible video candidate for one-click GitHub production.")
        result["selected_source"] = {"candidate_id": cid, "brand": selected.get("brand") or selected.get("brand_name"),
                                     "title": selected.get("title"), "source_url": selected.get("source_url"),
                                     "source_score": selected.get("source_score"),
                                     "evidence_scope": "ranked source metadata plus six sampled video frames; mechanism/transfer remain hypotheses"}
        analysis = app.do_analyze(cid)
        result["analysis"] = analysis
        if analysis.get("status") != "OK":
            raise RuntimeError("Source analysis did not return OK: " + str(analysis.get("status")) + " — " + str(analysis.get("detail", "")))
        production = app.do_produce(cid)
        result["production"] = production
        if production.get("status") != "OK" or production.get("exit_code") != 0:
            raise RuntimeError("Image production/QA did not PASS: " + str(production.get("status")) + "; no video was generated.")
        production_dir = WORK / "radar" / "production" / str(production.get("run_id"))
        approved = []
        for receipt_path in sorted((production_dir / "receipts").glob("*.json")):
            receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
            if str(receipt.get("overall", "")).upper() != "PASS":
                raise RuntimeError("At least one frame is not QA PASS; video generation blocked.")
            image_path = Path(receipt.get("image_path") or "")
            if not image_path.is_absolute():
                image_path = production_dir / image_path
            if not image_path.is_file():
                # Some receipts store only the filename while images live in shots/.
                image_path = production_dir / "shots" / image_path.name
            if not image_path.is_file():
                raise RuntimeError("QA-approved image path is missing from its receipt: " + str(receipt.get("image_path")))
            approved.append(image_path)
        if len(approved) != shots:
            raise RuntimeError(f"Expected {shots} QA-approved frames, found {len(approved)}.")
        video = _make_video(approved, OUT / "video")
        result["video"] = video
        # A successful test run is still TEST_ONLY unless the existing provider receipts explicitly qualify otherwise.
        commercial = [v.get("commercial_status") for v in production.get("verdicts", [])]
        result["status"] = "READY_FOR_REVIEW" if all(x == "CANDIDATE" for x in commercial) else "TEST_ONLY"
    except Exception as exc:
        result["status"] = "FAILED"
        result["errors"].append({"type": type(exc).__name__, "message": str(exc)})
        print("CME pipeline failed:", type(exc).__name__, str(exc), file=sys.stderr)
    finally:
        if app is not None:
            try:
                app.store.close()
            except Exception:
                pass
        result["finished_at"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        _json(OUT / "live_result.json", result)
        summary_raw = os.environ.get("GITHUB_STEP_SUMMARY", "").strip()
        if summary_raw:
            summary = Path(summary_raw)
            with summary.open("a", encoding="utf-8") as f:
                f.write("# CME Control Center\n\n")
                f.write(f"- **Status:** {result['status']}\n")
                f.write(f"- **Run:** {result.get('workflow_run_id')}\n")
                if result.get("selected_source"):
                    s = result["selected_source"]
                    f.write(f"- **Selected source:** {s.get('brand') or 'unknown brand'} — {s.get('title') or 'untitled'}\n")
                    if s.get("source_url"):
                        f.write(f"- **Source ad:** {s['source_url']}\n")
                if result.get("video", {}).get("path"):
                    f.write(f"- **Video artifact:** `{result['video']['path']}` ({result['video'].get('duration_seconds')} s, audio verified)\n")
                if result["errors"]:
                    f.write("\n## Blockers\n")
                    for e in result["errors"]:
                        f.write(f"- {e['type']}: {e['message']}\n")
                f.write("\nDownload **cme-control-center-output** from Artifacts for the full report, frames, receipts, and MP4.\n")
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0 if result["status"] in {"READY_FOR_REVIEW", "TEST_ONLY"} else 1


if __name__ == "__main__":
    raise SystemExit(main())
