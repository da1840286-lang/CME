"""Creative Radar acceptance tests. All data here is SYNTHETIC and lives only in temporary folders."""
import hashlib, http.client, json, os, shutil, subprocess, tempfile, threading, unittest
from pathlib import Path
from unittest import mock

import cme.radar.find as F
from cme.radar import analyze as A
from cme.radar.legacy import discover as legacy_discover
from cme.radar.legacy.discovery import tiktok_socialcrawl as T
from cme.radar.server import RadarApp, make_server
from cme.radar.store import Store
from tests.radar_fixtures import FakeResp, synthetic_post, synthetic_response

ROOT = Path(__file__).resolve().parent.parent
HAS_FFMPEG = bool(shutil.which("ffmpeg") and shutil.which("ffprobe"))


class Env(unittest.TestCase):
    """No external keys, no home-directory writes: offline tests need nothing from the outside."""

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        patcher = mock.patch.dict(os.environ, {"HOME": self.tmp.name}, clear=False)
        patcher.start()
        self.addCleanup(patcher.stop)
        for k in ("SOCIALCRAWL_API_KEY", "OPENROUTER_API_KEY", "HF_TOKEN"):
            os.environ.pop(k, None)
        self.data = Path(self.tmp.name) / "radar_data"

    def fake_find(self, items, calls=None):
        def fake_get(url, params=None, headers=None, timeout=None):
            if calls is not None:
                calls.append((url, dict(params or {})))
            return FakeResp(synthetic_response(items))
        return mock.patch.object(T.requests, "get", fake_get)


class VendoredCodeIsTheRealOne(Env):
    def test_vendored_files_match_manifest_hashes(self):
        man = json.loads((ROOT / "cme/radar/legacy/MANIFEST.json").read_text())
        self.assertTrue(man["files"])
        for rel, digest in man["files"].items():
            self.assertEqual(hashlib.sha256((ROOT / "cme/radar" / rel).read_bytes()).hexdigest(), digest, f"{rel} was edited")

    def test_find_uses_the_old_functions_not_a_stub(self):
        self.assertIs(F.discover_and_rank, legacy_discover.discover_and_rank)
        self.assertIs(F.SocialCrawlTikTokTopAds, T.SocialCrawlTikTokTopAds)
        self.assertEqual(F.LEGACY_ERROR, "")

    def test_run_find_calls_the_real_client_and_ranker(self):
        calls = []
        with self.fake_find([synthetic_post("1001")], calls):
            res = F.run_find({"country": "us", "max_requests": 2}, client=T.SocialCrawlTikTokTopAds(api_key="TESTKEY"))
        self.assertTrue(calls and "socialcrawl" in calls[0][0] and len(calls) <= 2)
        self.assertEqual(calls[0][1].get("country"), "US")
        sc = res["candidates"][0]["source_score"]
        self.assertIsNotNone(sc["final_score"])  # produced by the old arbitrage.score_tiktok_candidate
        self.assertEqual(res["request"]["vertical"], "Physical Products")


class CandidateRecord(Env):
    def run(self, *items):
        with self.fake_find(list(items)):
            return F.run_find({"country": "US", "max_requests": 1}, client=T.SocialCrawlTikTokTopAds(api_key="K"),
                              now="2026-10-21T00:00:00Z")["candidates"]

    def test_provenance_url_and_real_metrics_are_kept(self):
        c = self.run(synthetic_post("1001", views=120000, ctr=2.4))[0]
        self.assertEqual(c["metrics"]["views"]["value"], 120000)
        self.assertEqual(c["metrics"]["views"]["source"], "socialcrawl:tiktok_ads_top")
        self.assertEqual(c["metrics"]["views"]["collected_at"], "2026-10-21T00:00:00Z")
        self.assertEqual(c["metrics"]["ctr"]["value"], 2.4)
        self.assertTrue(c["source_url"].startswith("https://example.invalid/ad/1001"))
        self.assertEqual(c["country"], "US")
        self.assertEqual(c["origin"], "found")
        self.assertIn("legacy_record", c)

    def test_unknown_stays_unknown_not_zero(self):
        c = self.run(synthetic_post("1002", views=None, ctr=None, cvr=None))[0]
        for name in ("views", "ctr", "cvr"):
            self.assertNotIn(name, c["metrics"])

    def test_missing_video_is_none_not_invented(self):
        post = synthetic_post("1003")
        post["post"]["content"]["media_urls"] = []
        c = self.run(post)[0]
        self.assertIsNone(c["video_url"])


class DedupeAndFilters(Env):
    def cand(self, ad, **kw):
        with self.fake_find([synthetic_post(ad, **kw)]):
            return F.run_find({"country": kw.pop("country", "US"), "max_requests": 1},
                              client=T.SocialCrawlTikTokTopAds(api_key="K"), now="2026-10-21T00:00:00Z")["candidates"][0]

    def test_same_ad_twice_is_one_card_with_history_and_latest_metrics(self):
        s = Store(self.data / "r.db")
        s.ingest([self.cand("2001", views=1000)], "run1", "found")
        out = s.ingest([self.cand("2001", views=5000)], "run2", "found")
        self.assertEqual(out, {"new": 0, "updated": 1})
        cards = s.list(state="all")
        self.assertEqual(len(cards), 1)
        self.assertEqual(cards[0]["metrics"]["views"]["value"], 5000)
        hist = s.history(cards[0]["id"])
        self.assertEqual([h["metrics"]["views"]["value"] for h in hist], [1000, 5000])

    def test_duplicates_inside_one_batch_and_across_identity_keys(self):
        s = Store(self.data / "r.db")
        a, b = self.cand("2002"), self.cand("2002")
        self.assertEqual(s.ingest([a, b], "run1", "found"), {"new": 1, "updated": 1})
        c = dict(self.cand("2002"), id="tiktok:ad:other", keys=["tiktok:ad:other"] + self.cand("2002")["keys"][:1])
        s.ingest([c], "run2", "found")
        self.assertEqual(len(s.list(state="all")), 1)

    def test_filters(self):
        s = Store(self.data / "r.db")
        us = self.cand("3001", brand="alphabrand")
        de = dict(self.cand("3002", brand="betabrand"), country="DE", period_days=7, origin="imported")
        s.ingest([us], "r", "found")
        s.ingest([de], "r", "imported")
        self.assertEqual(len(s.list(state="all")), 2)
        self.assertEqual([c["ad_id"] for c in s.list(state="all", country="de")], ["3002"])
        self.assertEqual([c["ad_id"] for c in s.list(state="all", origin="imported")], ["3002"])
        self.assertEqual([c["ad_id"] for c in s.list(state="all", period="7")], ["3002"])
        self.assertEqual([c["ad_id"] for c in s.list(state="all", q="ALPHA")], ["3001"])
        self.assertEqual(s.list(state="all", platform="youtube"), [])


class OperatorDecisions(Env):
    def setUp(self):
        super().setUp()
        with self.fake_find([synthetic_post("4001"), synthetic_post("4002", brand="otherbrand")]):
            self.cands = F.run_find({"country": "US", "max_requests": 1}, client=T.SocialCrawlTikTokTopAds(api_key="K"))["candidates"]

    def blob(self, s, cid):
        return s.db.execute("SELECT data FROM candidates WHERE id=?", (cid,)).fetchone()["data"]

    def test_decisions_persist_across_restart_and_leave_the_original_untouched(self):
        s = Store(self.data / "r.db")
        s.ingest(self.cands, "run", "found")
        a, b = [c["id"] for c in self.cands]
        before = self.blob(s, a)
        s.decide(a, "shortlist", "strong demo")
        s.decide(b, "exclude", "not a physical product")
        self.assertEqual(self.blob(s, a), before)  # operator choice never edits the ad record
        s.close()
        s2 = Store(self.data / "r.db")
        self.assertEqual([c["id"] for c in s2.list(state="shortlist")], [a])
        self.assertEqual(s2.get(b)["operator"]["reason"], "not a physical product")
        self.assertEqual(s2.list(state="feed"), [])
        self.assertEqual(s2.counts()["saved"], 1)

    def test_reasons_are_required_and_skip_leaves_the_feed(self):
        s = Store(self.data / "r.db")
        s.ingest(self.cands, "run", "found")
        a = self.cands[0]["id"]
        with self.assertRaises(ValueError):
            s.decide(a, "exclude", "")
        with self.assertRaises(ValueError):
            s.decide(a, "shortlist", "  ")
        s.decide(a, "skip")
        self.assertNotIn(a, [c["id"] for c in s.list(state="feed")])
        s.decide(a, "restore")
        self.assertIn(a, [c["id"] for c in s.list(state="feed")])
        self.assertEqual(len(s.decision_log(a)), 2)

    def test_viewed_counter(self):
        s = Store(self.data / "r.db")
        s.ingest(self.cands, "run", "found")
        s.mark_viewed(self.cands[0]["id"])
        self.assertEqual(s.counts()["viewed"], 1)


class ImportIsLabelled(Env):
    def test_import_marks_origin_and_rejects_other_formats(self):
        with self.fake_find([synthetic_post("5001")]):
            found = F.run_find({"country": "US", "max_requests": 1}, client=T.SocialCrawlTikTokTopAds(api_key="K"))["candidates"][0]
        saved = Path(self.tmp.name) / "tiktok_discovery.json"
        saved.write_text(json.dumps({"request": {"country": "US", "period": 30}, "stats": {},
                                     "results": [{"record": found["legacy_record"], "decision": found["source_score"]}]}))
        res = F.import_discovery_file(saved)
        self.assertEqual(res["candidates"][0]["origin"], "imported")
        self.assertTrue(res["candidates"][0]["metrics"]["views"]["source"].startswith("import:"))
        bad = Path(self.tmp.name) / "bad.json"
        bad.write_text(json.dumps([1, 2]))
        with self.assertRaises(ValueError):
            F.import_discovery_file(bad)


class FindStates(Env):
    def test_not_configured_without_key(self):
        self.assertEqual(F.find_status()["state"], "NOT_CONFIGURED")
        app = RadarApp(self.data, settings_path=ROOT / "settings.example.json", catalog_path=ROOT / "adapters.json")
        res = app.do_find({"country": "US"})
        self.assertEqual(res["status"], "NOT_CONFIGURED")
        self.assertIn("SOCIALCRAWL_API_KEY", res["detail"])
        self.assertEqual(app.store.counts()["found"], 0)

    def test_api_failure_is_reported_not_replaced(self):
        def boom(*a, **k):
            raise OSError("network down")
        with mock.patch.object(T.requests, "get", boom):
            app = RadarApp(self.data, find_client=T.SocialCrawlTikTokTopAds(api_key="K"))
            res = app.do_find({"country": "US", "max_requests": 1})
        self.assertIn(res["status"], ("UNAVAILABLE",))
        self.assertEqual(app.store.counts()["found"], 0)

    def test_bad_parameters_are_rejected(self):
        app = RadarApp(self.data, find_client=T.SocialCrawlTikTokTopAds(api_key="K"))
        self.assertEqual(app.do_find({"period": 14})["status"], "BAD_REQUEST")
        self.assertEqual(app.do_find({"max_requests": 99})["status"], "BAD_REQUEST")


def fake_runner(frames_dir_times=None):
    """Stands in for ffprobe/ffmpeg: duration 12s; every ffmpeg call writes a tiny file."""
    class R:
        returncode, stdout, stderr = 0, "", ""
    def run(cmd):
        r = R()
        if cmd[0] == "ffprobe":
            r.stdout = "12.0\n"
        else:
            Path(cmd[-1]).write_bytes(b"\xff\xd8fakejpeg" + cmd[cmd.index("-ss") + 1].encode())
        return r
    return run


class FakeVideo:
    status_code = 200
    def __init__(self, data=b"fakevideo" * 50):
        self.data = data
    def iter_content(self, chunk_size=0):
        yield self.data


def analyst_transport(calls, content=None, status=200):
    good = {k: {"text": f"текст {k}", "kind": "OBSERVED"} for k in ("hook", "buyer_problem", "product_intervention", "result_demo", "cta")}
    good["mechanism_hypothesis"] = {"text": "гипотеза", "kind": "OBSERVED"}  # model overclaims: code must force INFERRED
    good["transferability"] = {"text": "перенос", "kind": "INFERRED"}
    good["evidence"] = [{"t_sec": 3.0, "frame": 1, "observation": "виден товар"}, {"t_sec": 8.0, "frame": 3, "observation": "выдумано"}]
    good["limits"] = {"text": "кадры, не всё видео", "missing": ["звук"]}
    good["ratings"] = {"problem_clarity": 4, "product_demo": 9, "proof_visibility": None, "sequence_quality": 3}
    def send(method, url, headers, payload):
        calls.append(payload)
        text = content if content is not None else json.dumps(good)
        return status, json.dumps({"model": payload["model"], "choices": [{"message": {"content": text}}]}).encode()
    return send


class ManualAnalysis(Env):
    def make_app(self, calls, **kw):
        s = json.loads((ROOT / "settings.example.json").read_text())
        sp = Path(self.tmp.name) / "settings.json"
        sp.write_text(json.dumps(s))
        ctx = {"transport": analyst_transport(calls, **kw), "openrouter_key": "K"}
        app = RadarApp(self.data, settings_path=sp, catalog_path=ROOT / "adapters.json", ctx=ctx, getter=lambda *a, **k: FakeVideo(),
                       runner=fake_runner(), tools={"state": "READY", "detail": "", "how_to_fix": ""})
        with self.fake_find([synthetic_post("6001")]):
            res = F.run_find({"country": "US", "max_requests": 1}, client=T.SocialCrawlTikTokTopAds(api_key="K"))
        app.store.ingest(res["candidates"], "run", "found")
        return app, res["candidates"][0]["id"]

    def test_nothing_calls_the_model_until_the_operator_asks(self):
        calls = []
        app, cid = self.make_app(calls)
        app.state(), app.feed({}), app.candidate(cid), app.store.mark_viewed(cid), app.store.decide(cid, "shortlist", "x")
        self.assertEqual(calls, [])
        app.do_analyze(cid)
        self.assertEqual(len(calls), 1)

    def test_analysis_keeps_source_link_frames_and_is_honest(self):
        calls = []
        app, cid = self.make_app(calls)
        res = app.do_analyze(cid)
        self.assertEqual(res["status"], "OK")
        a = app.store.analyses(cid)[0]
        self.assertEqual(a["status"], "OK")
        self.assertEqual(a["model_requested"], "google/gemma-4-26b-a4b-it:free")
        self.assertEqual(a["model_responded"], "google/gemma-4-26b-a4b-it:free")
        m = a["material"]
        self.assertEqual([f["t_sec"] for f in m["frames"]], [1.0, 3.0, 5.0, 7.0, 9.0, 11.0])
        self.assertFalse(m["video_fully_viewed"])
        self.assertTrue(m["source_url"].startswith("https://example.invalid/ad/6001"))
        self.assertEqual(m["ad_id"], "6001")
        r = a["result"]
        self.assertEqual(r["mechanism_hypothesis"]["kind"], "INFERRED")           # hypothesis never becomes an observation
        self.assertEqual([e["t_sec"] for e in r["evidence"]], [3.0])             # invented timestamp dropped
        self.assertTrue(any("8.0" in w for w in r["warnings"]))
        self.assertIsNone(r["ratings"]["product_demo"])                           # out-of-range rating is not trusted
        self.assertEqual(len([c for c in calls[0]["messages"][1]["content"] if c["type"] == "image_url"]), 6)

    def test_missing_key_is_not_configured_and_makes_no_call(self):
        calls = []
        app, cid = self.make_app(calls)
        app.ctx = {}
        res = app.do_analyze(cid)
        self.assertEqual(res["status"], "NOT_CONFIGURED")
        self.assertIn("OPENROUTER_API_KEY", res["detail"])
        self.assertEqual(calls, [])
        self.assertEqual(app.store.analyses(cid), [])

    def test_api_error_is_recorded_without_inventing_an_analysis(self):
        calls = []
        app, cid = self.make_app(calls, status=500)
        res = app.do_analyze(cid)
        self.assertEqual(res["status"], "API_ERROR")
        a = app.store.analyses(cid)[0]
        self.assertEqual((a["status"], a["result"]), ("API_ERROR", None))
        self.assertTrue(a["material"]["frames"])  # what was sent is still on record

    def test_garbage_output_is_invalid_not_accepted(self):
        calls = []
        app, cid = self.make_app(calls, content="I think this ad is great!")
        self.assertEqual(app.do_analyze(cid)["status"], "INVALID_OUTPUT")
        a = app.store.analyses(cid)[0]
        self.assertIsNone(a["result"].get("hook"))
        self.assertIn("raw_excerpt", a["result"])

    def test_video_unavailable_is_reported(self):
        calls = []
        app, cid = self.make_app(calls)
        class Gone(FakeVideo):
            status_code = 403
        app.getter = lambda *a, **k: Gone()
        res = app.do_analyze(cid)
        self.assertEqual(res["status"], "VIDEO_UNAVAILABLE")
        self.assertEqual(calls, [])

    def test_no_video_url_and_missing_ffmpeg(self):
        calls = []
        app, cid = self.make_app(calls)
        app.tools = {"state": "NOT_CONFIGURED", "detail": "ffmpeg/ffprobe not found", "how_to_fix": "install ffmpeg"}
        self.assertEqual(app.do_analyze(cid)["status"], "NOT_CONFIGURED")
        self.assertEqual(calls, [])

    @unittest.skipUnless(HAS_FFMPEG, "ffmpeg not installed")
    def test_real_ffmpeg_extracts_frames_with_timestamps(self):
        video = Path(self.tmp.name) / "t.mp4"
        subprocess.run(["ffmpeg", "-y", "-v", "error", "-f", "lavfi", "-i", "testsrc=duration=6:size=320x568:rate=10",
                        "-pix_fmt", "yuv420p", str(video)], check=True)
        frames = A.extract_frames(video, Path(self.tmp.name) / "fr", count=3)
        self.assertEqual([f["t_sec"] for f in frames], [1.0, 3.0, 5.0])
        self.assertTrue(all(Path(f["path"]).stat().st_size > 500 and len(f["sha256"]) == 64 for f in frames))


class Server(ManualAnalysis):
    def start(self, calls):
        app, cid = self.make_app(calls)
        srv = make_server(app, port=0)
        threading.Thread(target=srv.serve_forever, daemon=True).start()
        self.addCleanup(srv.shutdown)
        return app, cid, srv.server_address[1]

    def req(self, port, method, path, body=None, token=None, headers=None):
        conn = http.client.HTTPConnection("127.0.0.1", port, timeout=10)
        h = {"Content-Type": "application/json", **(headers or {})}
        if token:
            h["X-Radar-Token"] = token
        conn.request(method, path, json.dumps(body) if body is not None else None, h)
        r = conn.getresponse()
        raw = r.read()
        conn.close()
        return r.status, (json.loads(raw) if raw[:1] in (b"{", b"[") else raw.decode())

    def test_get_endpoints_never_trigger_analysis_and_page_is_served(self):
        calls = []
        app, cid, port = self.start(calls)
        code, page = self.req(port, "GET", "/")
        self.assertEqual(code, 200)
        self.assertIn("Physical Products", page)
        self.assertIn(app.token, page)
        self.assertIn("Видео недоступно", page)           # honest placeholder for missing video
        self.assertEqual(self.req(port, "GET", "/api/state")[0], 200)
        code, feed = self.req(port, "GET", "/api/feed?state=feed")
        self.assertEqual([c["id"] for c in feed["items"]], [cid])
        self.assertNotIn("legacy_record", feed["items"][0])
        self.assertEqual(self.req(port, "GET", f"/api/candidate?id={cid}")[0], 200)
        self.assertEqual(calls, [])

    def test_post_needs_token_origin_and_host(self):
        calls = []
        app, cid, port = self.start(calls)
        self.assertEqual(self.req(port, "POST", "/api/analyze", {"id": cid})[0], 403)
        self.assertEqual(self.req(port, "POST", "/api/analyze", {"id": cid}, token="wrong")[0], 403)
        self.assertEqual(self.req(port, "POST", "/api/analyze", {"id": cid}, token=app.token,
                                  headers={"Origin": "http://evil.example"})[0], 403)
        self.assertEqual(self.req(port, "GET", "/api/state", headers={"Host": "evil.example"})[0], 403)
        self.assertEqual(calls, [])
        code, res = self.req(port, "POST", "/api/analyze", {"id": cid}, token=app.token)
        self.assertEqual((code, res["status"]), (200, "OK"))
        self.assertEqual(len(calls), 1)

    def test_decisions_over_http(self):
        app, cid, port = self.start([])
        code, res = self.req(port, "POST", "/api/decide", {"id": cid, "action": "exclude", "reason": ""}, token=app.token)
        self.assertEqual(code, 400)
        self.assertEqual(self.req(port, "POST", "/api/decide", {"id": cid, "action": "shortlist", "reason": "good"}, token=app.token)[0], 200)
        _, sl = self.req(port, "GET", "/api/feed?state=shortlist")
        self.assertEqual(sl["items"][0]["operator"]["reason"], "good")
        _, state = self.req(port, "GET", "/api/state")
        self.assertEqual(state["counts"]["saved"], 1)

    def test_server_refuses_non_local_binding(self):
        app, _cid = self.make_app([])
        with self.assertRaises(ValueError):
            make_server(app, host="0.0.0.0")


class CoreStaysCompatible(unittest.TestCase):
    def test_settings_positional_order_and_new_field_are_backward_compatible(self):
        from cme.registry import Settings, load_settings
        s = Settings("hf", "paid", True)
        self.assertEqual((s.image_adapter, s.fallback_adapter, s.allow_free, s.analyst_adapter), ("hf", "paid", True, ""))
        self.assertEqual(load_settings(ROOT / "settings.example.json").analyst_adapter, "openrouter_free_analyst")

    def test_radar_does_not_touch_core_contract_modules(self):
        import cme.atoms, cme.receipt, cme.compiler
        src = "\n".join(Path(m.__file__).read_text() for m in (cme.atoms, cme.receipt, cme.compiler))
        self.assertNotIn("radar", src)


if __name__ == "__main__":
    unittest.main()
