import importlib
import importlib.util
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
from urllib.error import HTTPError


class GemmaReviewTests(unittest.TestCase):
    def setUp(self):
        self.assertIsNotNone(importlib.util.find_spec("src.gemma_review"))
        self.gemma = importlib.import_module("src.gemma_review")
        self.review = importlib.import_module("src.llm_review")
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.output = Path(self.temp.name) / "responses.jsonl"
        self.settings = {
            "temperature": 0, "maxOutputTokens": 32,
            "thinkingConfig": {"thinkingLevel": "minimal"},
        }
        self.features = {
            "dur": 1, "stos": 0, "dtos": None, "tot_pkts": 2,
            "tot_bytes": 100, "src_bytes": 50, "proto": "tcp",
            "dir": "->", "state": "con",
        }
        self.request = {
            "flow_id": "s03_r1", "scenario": 3, "source_files": ["validation.csv"],
            "prompt_id": "unit-test-prompt",
            "messages": [
                {"role": "system", "content": "Reply Normal or Botnet."},
                {"role": "user", "content": json.dumps(self.features)},
            ],
        }
        self.api_response = {
            "candidates": [{
                "content": {"parts": [
                    {"text": "An internal reasoning summary", "thought": True},
                    {"text": "Botnet"},
                ]},
                "finishReason": "STOP",
            }],
            "usageMetadata": {
                "promptTokenCount": 70, "candidatesTokenCount": 2,
                "thoughtsTokenCount": 8, "totalTokenCount": 80,
            },
            "modelVersion": "unit-test-version", "responseId": "unit-test-response",
        }

    def test_payload_contains_only_instruction_features_and_explicit_settings(self):
        payload = self.gemma.make_payload(self.request, self.settings)
        self.assertEqual(set(payload), {"contents", "systemInstruction", "generationConfig"})
        self.assertEqual(payload["generationConfig"], self.settings)
        self.assertEqual(
            json.loads(payload["contents"][0]["parts"][0]["text"]), self.features
        )
        encoded = json.dumps(payload)
        for excluded in ("s03_r1", "scenario", "flow_id", "source_files"):
            self.assertNotIn(excluded, encoded)
        modified = json.loads(json.dumps(self.request))
        modified["messages"][1]["content"] = json.dumps({**self.features, "true_label": 1})
        with self.assertRaisesRegex(ValueError, "features"):
            self.gemma.make_payload(modified, self.settings)

    def test_parse_counts_thinking_tokens_without_using_thought_text_as_label(self):
        result = self.gemma.parse_response(self.api_response)
        self.assertEqual(result["response"], "Botnet")
        self.assertEqual(result["input_tokens"], 70)
        self.assertEqual(result["output_tokens"], 10)
        self.assertEqual(result["response_metadata"]["model_version"], "unit-test-version")
        self.assertIsNone(result["error"])
        missing_usage = self.gemma.parse_response({"candidates": []})
        self.assertIsNone(missing_usage["input_tokens"])
        self.assertIsNone(missing_usage["output_tokens"])
        truncated = json.loads(json.dumps(self.api_response))
        truncated["candidates"][0]["finishReason"] = "MAX_TOKENS"
        self.assertEqual(self.gemma.parse_response(truncated)["error"], "invalid_response")

    def test_api_key_is_in_header_not_in_url_or_payload(self):
        with patch("src.gemma_review.urlopen") as opened:
            opened.return_value.__enter__.return_value.read.return_value = json.dumps(
                self.api_response
            ).encode()
            self.gemma.call_gemma(self.request, "unit-test-secret", self.settings)
        sent = opened.call_args.args[0]
        self.assertIn("gemma-4-31b-it:generateContent", sent.full_url)
        self.assertNotIn("unit-test-secret", sent.full_url)
        self.assertNotIn(b"unit-test-secret", sent.data)
        self.assertIn("unit-test-secret", dict(sent.header_items()).values())

    def test_runner_reuses_response_for_another_flow_and_resumes_without_calls(self):
        another = {**self.request, "flow_id": "s03_r2"}
        parsed = self.gemma.parse_response(self.api_response)
        with patch("src.gemma_review.call_gemma", return_value=parsed) as call:
            stats = self.gemma.run_reviews(
                [self.request, another], self.output, api_key="unit-test-secret",
                generation_settings=self.settings, max_new_requests=1,
            )
            self.assertEqual(call.call_count, 1)
            self.assertEqual(stats["cache_hits"], 1)
            again = self.gemma.run_reviews(
                [self.request, another], self.output, api_key="unit-test-secret",
                generation_settings=self.settings, max_new_requests=1,
            )
            self.assertEqual(call.call_count, 1)
            self.assertEqual(again["already_recorded"], 2)
        first, reused = [json.loads(line) for line in self.output.read_text().splitlines()]
        self.assertEqual(reused["flow_id"], "s03_r2")
        self.assertTrue(reused["cache_hit"])
        self.assertFalse(first["cache_hit"])
        self.assertEqual(reused["output_tokens"], 10)
        self.assertEqual(reused["response_metadata"], first["response_metadata"])

    def test_runner_stops_after_quota_error_and_keeps_failure_record(self):
        changed = json.loads(json.dumps(self.request))
        changed["flow_id"] = "s03_r2"
        changed["messages"][1]["content"] = json.dumps({**self.features, "dur": 2})
        failure = HTTPError("https://example.test", 429, "quota", None, None)
        with patch("src.gemma_review.call_gemma", side_effect=failure) as call:
            stats = self.gemma.run_reviews(
                [self.request, changed], self.output, api_key="unit-test-secret",
                generation_settings=self.settings, max_new_requests=2,
            )
        self.assertEqual(call.call_count, 1)
        self.assertEqual(stats["failed_calls"], 1)
        saved = json.loads(self.output.read_text())
        self.assertEqual(saved["error"], "http_429")
        self.assertIsNone(saved["prediction"])
        self.assertNotIn("unit-test-secret", self.output.read_text())

    def test_runner_retries_503_and_records_the_recovered_prediction(self):
        failure = HTTPError("https://example.test", 503, "unavailable", None, None)
        parsed = self.gemma.parse_response(self.api_response)
        with (
            patch("src.gemma_review.call_gemma", side_effect=[failure, parsed]) as call,
            patch("src.gemma_review.time.sleep") as sleep,
        ):
            stats = self.gemma.run_reviews(
                [self.request], self.output, api_key="unit-test-secret",
                generation_settings=self.settings, max_new_requests=2,
            )
        self.assertEqual(call.call_count, 2)
        self.assertEqual(stats["new_calls"], 2)
        self.assertEqual(stats["failed_calls"], 1)
        self.assertIsNone(stats["stop_reason"])
        self.assertGreater(sleep.call_args.args[0], 2)
        first, final = map(json.loads, self.output.read_text().splitlines())
        self.assertEqual(first["error"], "http_503")
        self.assertEqual(final["flow_id"], first["flow_id"])
        self.assertEqual(final["prediction"], 1)
        self.assertEqual(final["input_tokens"], 70)

    def test_persistent_503_stops_at_the_retry_or_call_limit(self):
        for budget, expected in ((2, 2), (10, 3)):
            with self.subTest(budget=budget):
                output = self.output.with_name(f"budget-{budget}.jsonl")
                failure = HTTPError("https://example.test", 503, "unavailable", None, None)
                with (
                    patch("src.gemma_review.call_gemma", side_effect=failure) as call,
                    patch("src.gemma_review.time.sleep"),
                ):
                    stats = self.gemma.run_reviews(
                        [self.request], output, api_key="unit-test-secret",
                        generation_settings=self.settings, max_new_requests=budget,
                    )
                self.assertEqual(call.call_count, expected)
                self.assertEqual(stats["new_calls"], expected)
                self.assertEqual(stats["stop_reason"], "http_503")
                self.assertEqual(len(output.read_text().splitlines()), expected)

    def test_call_limit_does_not_silently_process_the_rest(self):
        changed = json.loads(json.dumps(self.request))
        changed["flow_id"] = "s03_r2"
        changed["messages"][1]["content"] = json.dumps({**self.features, "dur": 2})
        parsed = self.gemma.parse_response(self.api_response)
        with patch("src.gemma_review.call_gemma", return_value=parsed) as call:
            stats = self.gemma.run_reviews(
                [self.request, changed], self.output, api_key="unit-test-secret",
                generation_settings=self.settings, max_new_requests=1,
            )
        self.assertEqual(call.call_count, 1)
        self.assertEqual(stats["new_calls"], 1)
        self.assertEqual(len(self.output.read_text().splitlines()), 1)

    def test_key_is_loaded_from_dotenv_without_changing_environment(self):
        self.assertTrue(hasattr(self.gemma, "get_api_key"))
        env_file = Path(self.temp.name) / ".env"
        env_file.write_text('# Local settings\nGEMINI_API_KEY="unit-test-file-key"\n')
        with patch.dict(os.environ, {}, clear=True):
            self.assertEqual(self.gemma.get_api_key(env_file), "unit-test-file-key")
            self.assertNotIn("GEMINI_API_KEY", os.environ)

    def test_environment_key_takes_priority_over_dotenv(self):
        self.assertTrue(hasattr(self.gemma, "get_api_key"))
        env_file = Path(self.temp.name) / ".env"
        env_file.write_text("GEMINI_API_KEY=unit-test-file-key\n")
        with patch.dict(os.environ, {"GEMINI_API_KEY": "unit-test-env-key"}, clear=True):
            self.assertEqual(self.gemma.get_api_key(env_file), "unit-test-env-key")

    def test_empty_or_absent_dotenv_does_not_produce_a_key(self):
        self.assertTrue(hasattr(self.gemma, "get_api_key"))
        env_file = Path(self.temp.name) / ".env"
        with patch.dict(os.environ, {}, clear=True):
            self.assertEqual(self.gemma.get_api_key(env_file), "")
            env_file.write_text("GEMINI_API_KEY=\n")
            self.assertEqual(self.gemma.get_api_key(env_file), "")


if __name__ == "__main__":
    unittest.main()
