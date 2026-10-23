import unittest

from cme.github_pipeline import _select_candidate


class GitHubPipelineSelectionTests(unittest.TestCase):
    def test_selects_first_ranked_candidate_with_video(self):
        candidates = [
            {"id": "no-video", "video_url": None, "source_check": {"excluded": False}},
            {"id": "excluded", "video_url": "https://example.test/ad.mp4", "source_check": {"excluded": True}},
            {"id": "top-video", "video_url": "https://example.test/top.mp4", "source_check": {"excluded": False}},
            {"id": "lower-video", "video_url": "https://example.test/lower.mp4", "source_check": {"excluded": False}},
        ]
        self.assertEqual(_select_candidate(candidates)["id"], "top-video")

    def test_missing_exclusion_metadata_does_not_drop_candidate(self):
        candidate = {"id": "video", "video_url": "https://example.test/ad.mp4"}
        self.assertIs(_select_candidate([candidate]), candidate)

    def test_fails_closed_when_no_video_candidate_exists(self):
        with self.assertRaisesRegex(RuntimeError, "no eligible TikTok candidate"):
            _select_candidate([{"id": "no-video", "video_url": None}])


if __name__ == "__main__":
    unittest.main()
