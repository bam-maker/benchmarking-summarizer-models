"""Device choice, transparent fallback, memory residency, and EOS regressions."""

import threading
import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

import torch

import runtime
import summarizer as engine
from hf_summarizer import HFSummarizer


class DeviceTests(unittest.TestCase):
    def test_auto_falls_back_but_explicit_cuda_does_not(self):
        with (
            patch.dict("os.environ", {"SUMMARIZER_DEVICE": "auto"}),
            patch.object(torch.cuda, "is_available", return_value=False),
        ):
            self.assertEqual(runtime.resolve_device("auto"), "cpu")
            self.assertEqual(runtime.resolve_device("cpu"), "cpu")
            with self.assertRaisesRegex(RuntimeError, "unavailable"):
                runtime.resolve_device("cuda")

    def test_model_switch_offloads_inactive_gpu_weights_without_reload(self):
        class Backend:
            def __init__(self, device):
                self.device = device

            def to_device(self, device):
                self.device = device

        old = Backend("cuda")
        selected = Backend("cpu")
        with (
            patch.object(engine, "_models", {"LSTM": old, "BART": selected}),
            patch.object(engine, "resolve_device", return_value="cuda"),
            patch.object(torch.cuda, "empty_cache"),
        ):
            self.assertIs(engine.get_model("BART", "cuda"), selected)
            self.assertEqual(old.device, "cpu")
            self.assertEqual(selected.device, "cuda")

    def test_auto_oom_retry_is_labeled_cpu_and_timing_is_separate(self):
        backend = SimpleNamespace(
            generate=Mock(
                side_effect=[
                    torch.cuda.OutOfMemoryError("test"),
                    ("a summary.", 3, True, 0.25),
                ]
            ),
            to_device=Mock(),
        )
        with (
            patch.object(engine, "get_model", return_value=backend),
            patch.object(engine, "resolve_device", return_value="cuda"),
            patch.object(engine, "synchronize"),
            patch.object(torch.cuda, "empty_cache"),
        ):
            result = engine.summarize("A real article.", "LSTM", "Short", device="auto")
        self.assertEqual(result.device, "cpu")
        self.assertEqual(result.seconds, 0.25)
        self.assertGreaterEqual(result.load_seconds, 0)
        self.assertIn("CPU fallback", result.notices)
        backend.to_device.assert_called_once_with("cpu")

    def test_explicit_cuda_never_retries_on_cpu(self):
        backend = SimpleNamespace(
            generate=Mock(side_effect=torch.cuda.OutOfMemoryError("test"))
        )
        with (
            patch.object(engine, "get_model", return_value=backend),
            patch.object(engine, "resolve_device", return_value="cuda"),
            patch.object(engine, "synchronize"),
            self.assertRaises(torch.cuda.OutOfMemoryError),
        ):
            engine.summarize("A real article.", "LSTM", "Short", device="cuda")
        self.assertEqual(backend.generate.call_count, 1)

    def test_auto_cpu_fallback_when_device_activation_runs_out_of_memory(self):
        backend = SimpleNamespace(
            generate=Mock(return_value=("a summary.", 3, True, 0.25))
        )
        with (
            patch.object(
                engine,
                "get_model",
                side_effect=[torch.cuda.OutOfMemoryError("test"), backend],
            ) as loader,
            patch.object(engine, "resolve_device", return_value="cuda"),
            patch.object(engine, "synchronize"),
        ):
            result = engine.summarize("A real article.", "LSTM", "Short", device="auto")
        self.assertEqual(result.device, "cpu")
        self.assertIn("loading ran out of memory", result.notices)
        self.assertEqual(loader.call_args.args, ("LSTM", "cpu"))

    def test_hf_decoder_start_eos_is_not_mistaken_for_termination(self):
        backend = HFSummarizer.__new__(HFSummarizer)
        backend.device = "cpu"
        backend.lock = threading.Lock()
        backend.tokenizer = SimpleNamespace(
            eos_token_id=2, decode=Mock(return_value="unfinished")
        )
        backend.model = SimpleNamespace(
            generate=Mock(return_value=torch.tensor([[2, 7, 8, 9]]))
        )
        prepared = engine.PreparedArticle((4, 5, 6), 3)
        _, count, ended, _ = backend.generate(prepared, engine.Budget(1, 3))
        self.assertFalse(ended)
        self.assertEqual(count, 3)
        self.assertIsNone(
            backend.model.generate.call_args.kwargs["forced_eos_token_id"]
        )


if __name__ == "__main__":
    unittest.main()
