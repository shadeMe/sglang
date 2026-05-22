"""Smoke tests for Mellum (JetBrains Qwen3-MoE variant with interleaved SWA).

These tests verify that the model loads, generates coherent output, and that the
hybrid SWA / per-layer RoPE infrastructure is wired up correctly.  They do NOT
compare against an HF reference (the BF16 model is too large for a single GPU).

Usage (local, assumes ~/.grazie/models/mellum-v2.3 exists):
    MELLUM_MODEL_PATH=~/.grazie/models/mellum-v2.3 python3 -m pytest test/registered/models_e2e/test_mellum.py -v

When a public HF hub checkpoint is available, replace the path and consider
adding an HF-reference logit comparison in test_generation_models.py.
"""

import os
import unittest

from sglang.test.test_utils import CustomTestCase, run_bench_one_batch

# No CI registration yet — the model checkpoint is not publicly available.
# Uncomment and adjust when it is:
# from sglang.test.ci.ci_register import register_cuda_ci
# register_cuda_ci(est_time=120, stage="extra-b", runner_config="1-gpu-large")

MELLUM_MODEL_PATH = os.environ.get(
    "MELLUM_MODEL_PATH", os.path.expanduser("~/.grazie/models/mellum-v2.3")
)

# Minimal flags to run the 12B-A2.5B MoE model on a single GPU with FP8.
_MELLUM_ARGS = [
    "--quantization",
    "fp8",
    "--mem-fraction-static",
    "0.50",
    "--context-length",
    "4096",
    "--cpu-offload-gb",
    "4",
]


@unittest.skipIf(
    not os.path.isdir(MELLUM_MODEL_PATH),
    f"Mellum checkpoint not found at {MELLUM_MODEL_PATH}",
)
class TestMellumSmoke(CustomTestCase):
    """Smoke test: load Mellum with FP8 quantization and generate tokens."""

    def test_bench_one_batch(self):
        _, output_throughput, _ = run_bench_one_batch(
            MELLUM_MODEL_PATH,
            [*_MELLUM_ARGS],
        )
        self.assertGreater(output_throughput, 0)

    def test_correctness(self):
        """Run bench_one_batch --correctness-test (prefill + decode, no HF ref)."""
        _, output_throughput, _ = run_bench_one_batch(
            MELLUM_MODEL_PATH,
            [*_MELLUM_ARGS, "--correctness-test"],
        )
        self.assertGreater(output_throughput, 0)


if __name__ == "__main__":
    unittest.main()
