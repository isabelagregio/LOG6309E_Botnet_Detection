import csv
import importlib
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest


FEATURES = [
    "dur", "stos", "dtos", "tot_pkts", "tot_bytes", "src_bytes",
    "proto", "dir", "state",
]


class LLMReviewTests(unittest.TestCase):
    def setUp(self):
        self.assertIsNotNone(
            importlib.util.find_spec("src.llm_review"),
            "The offline LLM preparation module is missing",
        )
        self.review = importlib.import_module("src.llm_review")
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.folder = Path(self.temp.name)
        self.prompt = self.folder / "prompt.txt"
        self.prompt.write_text("Draft instruction: reply Normal or Botnet.")

    def write_csv(self, name, rows):
        path = self.folder / name
        with path.open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
        return path

    def flow(self, flow_id="s11_r1", **changes):
        row = {
            "flow_id": flow_id, "scenario": "11", "dur": "1.25",
            "stos": "0", "dtos": "", "tot_pkts": "3",
            "tot_bytes": "180", "src_bytes": "100", "proto": "tcp",
            "dir": "->", "state": "CON",
            "true_label": "GROUND_TRUTH_SENTINEL",
            "ml_prediction": "ML_PREDICTION_SENTINEL",
            "ml_probability": "ML_PROBABILITY_SENTINEL",
            "uncertainty": "UNCERTAINTY_SENTINEL",
        }
        return {**row, **changes}

    def test_only_flow_features_enter_messages(self):
        source = self.write_csv("review.csv", [self.flow()])
        request = self.review.prepare_requests([source], self.prompt)[0]
        features = json.loads(request["messages"][1]["content"])
        self.assertEqual(set(features), set(FEATURES))
        self.assertEqual(features["dur"], 1.25)
        self.assertIsNone(features["dtos"])
        self.assertEqual(request["flow_id"], "s11_r1")
        messages = json.dumps(request["messages"])
        for excluded in (
            "GROUND_TRUTH_SENTINEL", "ML_PREDICTION_SENTINEL",
            "ML_PROBABILITY_SENTINEL", "UNCERTAINTY_SENTINEL", "s11_r1",
            "flow_id", "scenario",
        ):
            self.assertNotIn(excluded, messages)

    def test_nested_selections_share_one_request_with_all_memberships(self):
        first = self.write_csv("review_uncertainty_k1.csv", [self.flow()])
        second = self.write_csv(
            "review_random_k2.csv", [self.flow(), self.flow("s11_r2")]
        )
        requests = self.review.prepare_requests([first, second], self.prompt)
        self.assertEqual(len(requests), 2)
        shared = next(r for r in requests if r["flow_id"] == "s11_r1")
        self.assertEqual(set(shared["source_files"]), {str(first), str(second)})

    def test_conflicting_data_for_same_flow_are_rejected(self):
        first = self.write_csv("first.csv", [self.flow()])
        second = self.write_csv("second.csv", [self.flow(dur="9.5")])
        with self.assertRaisesRegex(ValueError, "Conflicting"):
            self.review.prepare_requests([first, second], self.prompt)

    def test_missing_required_feature_is_not_silently_dropped(self):
        row = self.flow()
        del row["tot_pkts"]
        source = self.write_csv("missing.csv", [row])
        with self.assertRaisesRegex(ValueError, "tot_pkts"):
            self.review.prepare_requests([source], self.prompt)

    def test_results_preserve_actual_response_and_unknown_token_counts(self):
        source = self.write_csv("review.csv", [self.flow()])
        request = self.review.prepare_requests([source], self.prompt)[0]
        output = self.folder / "responses.jsonl"
        self.review.save_response(
            output, request, model="unit-test-model", response=" Botnet\n",
            input_tokens=None, output_tokens=None, response_time_seconds=0.25,
        )
        saved = json.loads(output.read_text())
        self.assertEqual(saved["flow_id"], "s11_r1")
        self.assertEqual(saved["prediction"], 1)
        self.assertEqual(saved["response"], " Botnet\n")
        self.assertIsNone(saved["input_tokens"])
        self.assertIsNone(saved["output_tokens"])
        self.assertIsNone(saved["error"])
        self.assertEqual(saved["prompt_id"], request["prompt_id"])

    def test_invalid_and_failed_responses_are_not_guessed(self):
        source = self.write_csv("review.csv", [self.flow()])
        request = self.review.prepare_requests([source], self.prompt)[0]
        output = self.folder / "responses.jsonl"
        self.review.save_response(
            output, request, model="unit-test-model",
            response="Normal or Botnet: I cannot decide.",
            input_tokens=30, output_tokens=10, response_time_seconds=0.1,
        )
        self.review.save_response(
            output, request, model="unit-test-model", response=None,
            input_tokens=None, output_tokens=None, response_time_seconds=0.2,
            error="provider_unavailable",
        )
        invalid, failed = [json.loads(line) for line in output.read_text().splitlines()]
        self.assertIsNone(invalid["prediction"])
        self.assertEqual(invalid["error"], "invalid_response")
        self.assertIsNone(failed["prediction"])
        self.assertEqual(failed["error"], "provider_unavailable")

    def test_cache_key_tracks_input_and_configuration_but_not_selection(self):
        self.assertTrue(hasattr(self.review, "make_cache_key"))
        source = self.write_csv("review.csv", [self.flow()])
        request = self.review.prepare_requests([source], self.prompt)[0]
        config = {
            "backend": "unit-test-backend", "model": "unit-test-model",
            "generation_settings": {"temperature": 0, "max_tokens": 4},
        }
        key = self.review.make_cache_key(request, **config)
        other_selection = {**request, "flow_id": "another-id", "source_files": []}
        self.assertEqual(key, self.review.make_cache_key(other_selection, **config))
        for field, value in (
            ("backend", "another-backend"), ("model", "another-model"),
            ("generation_settings", {"temperature": 1, "max_tokens": 4}),
        ):
            with self.subTest(field=field):
                self.assertNotEqual(
                    key, self.review.make_cache_key(request, **{**config, field: value})
                )
        changed_messages = json.loads(json.dumps(request))
        changed_messages["messages"][1]["content"] = '{"dur": 999}'
        self.assertNotEqual(key, self.review.make_cache_key(changed_messages, **config))
        changed_prompt = json.loads(json.dumps(request))
        changed_prompt["messages"][0]["content"] = "Another instruction"
        self.assertNotEqual(key, self.review.make_cache_key(changed_prompt, **config))

    def test_cache_preserves_tokens_and_invalid_outputs_but_excludes_call_failures(self):
        self.assertTrue(hasattr(self.review, "load_cached_responses"))
        source = self.write_csv("review.csv", [self.flow()])
        request = self.review.prepare_requests([source], self.prompt)[0]
        output = self.folder / "responses.jsonl"
        self.assertEqual(self.review.load_cached_responses(output), {})
        config = {
            "backend": "unit-test-backend", "model": "unit-test-model",
            "generation_settings": {"temperature": 0, "max_tokens": 4},
        }
        self.review.save_response(
            output, request, **config, response=None, input_tokens=None,
            output_tokens=None, response_time_seconds=0.1, error="connection_failed",
        )
        self.assertEqual(self.review.load_cached_responses(output), {})
        self.review.save_response(
            output, request, **config, response="I cannot decide.", input_tokens=31,
            output_tokens=5, response_time_seconds=0.2,
        )
        cache = self.review.load_cached_responses(output)
        key = self.review.make_cache_key(request, **config)
        self.assertIsNone(cache[key]["prediction"])
        self.assertEqual(cache[key]["error"], "invalid_response")
        self.assertEqual(cache[key]["input_tokens"], 31)
        self.assertEqual(cache[key]["output_tokens"], 5)
        self.review.save_response(
            output, request, **config, response="Botnet", input_tokens=31,
            output_tokens=2, response_time_seconds=0.3,
        )
        # Do not silently replace the first completed response with a later one.
        self.assertEqual(self.review.load_cached_responses(output)[key], cache[key])


if __name__ == "__main__":
    unittest.main()
