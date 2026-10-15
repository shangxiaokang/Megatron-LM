# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Blackwell numerical contract tests against native PyTorch linear gradients."""

import unittest


class BlackwellNumerics(unittest.TestCase):
    def test_ragged_grouped_overrides_preserve_layout_and_gradients(self):
        import torch
        import torch.nn.functional as functional
        import transformer_engine.pytorch as te

        from megatron.core.extensions.transformer_engine import (
            TEGroupedLinear,
            TEQuantizationParams,
        )
        from megatron.core.quantization.quant_config import MatchContext, QuantizationConfig
        from tools.precision_search.model import Policy

        class MinimalGrouped(TEGroupedLinear):
            def __init__(self, precision):
                te.GroupedLinear.__init__(
                    self, 4, 256, 512, bias=False, device="cuda", params_dtype=torch.bfloat16
                )
                self.te_quant_params = TEQuantizationParams.parse_from_config(
                    QuantizationConfig(
                        Policy(precision).export()["configs"][precision],
                        MatchContext("grouped", 0),
                        precision,
                    )
                )
                self.disable_parameter_transpose_cache = True
                self.is_first_microbatch = True
                self.te_return_bias = False

        torch.manual_seed(42)
        splits = [37, 67, 0, 119]
        inputs = torch.randn(sum(splits), 256, device="cuda", dtype=torch.bfloat16)
        weights = [
            torch.randn(512, 256, device="cuda", dtype=torch.bfloat16) * 0.02 for _ in splits
        ]
        dy = torch.randn(sum(splits), 512, device="cuda", dtype=torch.bfloat16)
        reference_x = inputs.clone().requires_grad_()
        reference_w = [weight.clone().requires_grad_() for weight in weights]
        reference = torch.cat(
            [functional.linear(x, w) for x, w in zip(reference_x.split(splits), reference_w)]
        )
        reference.backward(dy)
        for precision, training, bound in (
            ("bf16", True, 0.02),
            ("bf16", False, 0.02),
            ("mxfp8", True, 0.12),
            ("mxfp8", False, 0.12),
            ("nvfp4", True, 0.35),
            ("nvfp4", False, 0.35),
        ):
            with self.subTest(precision=precision, training=training):
                module = MinimalGrouped(precision).train(training)
                with torch.no_grad():
                    for index, weight in enumerate(weights):
                        getattr(module, f"weight{index}").copy_(weight)
                x = inputs.clone().requires_grad_()
                with torch.set_grad_enabled(training):
                    output, bias = module(x, splits)
                self.assertIsNone(bias)
                self.assertEqual(output.shape, reference.shape)
                measurements = [("fprop", output, reference)]
                if training:
                    output.backward(dy)
                    measurements += [
                        ("dgrad", x.grad, reference_x.grad),
                        (
                            "wgrad",
                            torch.stack([getattr(module, f"weight{i}").grad for i in range(4)]),
                            torch.stack([weight.grad for weight in reference_w]),
                        ),
                    ]
                for role, actual, expected in measurements:
                    actual, expected = actual.float(), expected.float()
                    relative = ((actual - expected).norm() / expected.norm()).item()
                    self.assertTrue(torch.isfinite(actual).all())
                    self.assertLess(relative, bound, f"{precision} {role}: {relative}")
                if precision == "bf16":
                    self.assertFalse(hasattr(module, "_local_quantization_padding"))
                else:
                    self.assertTrue(hasattr(module, "_local_quantization_padding"))

    def test_bf16_policy_skips_idle_autocast_and_overrides_active_quantization(self):
        from unittest.mock import patch

        import torch
        import transformer_engine.pytorch as te
        from transformer_engine.common.recipe import MXFP8BlockScaling, NVFP4BlockScaling
        from transformer_engine.pytorch.quantization import FP8GlobalStateManager

        import megatron.core.extensions.transformer_engine as extension
        from tools.precision_search.model import Policy

        config = Policy("bf16").export()["configs"]["bf16"]
        params = extension.TEQuantizationParams(
            training_recipe=extension.TEQuantizationRecipe.parse_from_config(
                config["training_recipe"]
            ),
            evaluation_recipe=extension.TEQuantizationRecipe.parse_from_config(
                config["evaluation_recipe"]
            ),
        )
        linear = te.Linear(256, 512, bias=False, params_dtype=torch.bfloat16, device="cuda")
        x = torch.randn(128, 256, device="cuda", dtype=torch.bfloat16)
        reference = torch.nn.functional.linear(x, linear.weight)
        self.assertFalse(FP8GlobalStateManager.is_fp8_enabled())
        with patch.object(extension, "fp8_autocast", wraps=extension.fp8_autocast) as autocast:
            with extension._get_fp8_autocast_for_quant_params(params, True):
                actual = linear(x)
            autocast.assert_not_called()
        torch.testing.assert_close(actual, reference, rtol=0, atol=0)
        for recipe in (MXFP8BlockScaling(), NVFP4BlockScaling()):
            for training in (True, False):
                with self.subTest(recipe=type(recipe).__name__, training=training):
                    with te.autocast(enabled=True, recipe=recipe):
                        with extension._get_fp8_autocast_for_quant_params(params, training):
                            self.assertFalse(FP8GlobalStateManager.is_fp8_enabled())
                            actual = linear(x)
                        self.assertTrue(FP8GlobalStateManager.is_fp8_enabled())
                    torch.testing.assert_close(actual, reference, rtol=0, atol=0)

    def test_large_parameter_sampling_uses_exact_integer_offsets(self):
        import torch

        from tools.precision_search.worker import Worker

        worker = Worker.__new__(Worker)
        worker.torch = torch
        # The old float32 linspace rounds the final index to size, exceeding
        # the valid range. Real vocabulary projections are much larger.
        tensor = torch.ones(2**24 + 4, device="cuda", dtype=torch.bfloat16)
        tensor[0], tensor[-1] = 2, 3
        values = worker._sample(tensor)
        self.assertEqual(len(values), 256)
        self.assertEqual(values[0], 2)
        self.assertEqual(values[-1], 3)
        self.assertEqual(values[1:-1], [1] * 254)
        self.assertEqual(worker._sample(torch.empty(0, device="cuda")), [])
        self.assertEqual(worker._sample(torch.tensor(7, device="cuda")), [7])

    def test_forward_dgrad_wgrad_against_native_reference(self):
        import torch
        import torch.nn.functional as functional
        import transformer_engine.pytorch as te
        from transformer_engine.common.recipe import MXFP8BlockScaling, NVFP4BlockScaling
        from transformer_engine.pytorch.quantization import is_mxfp8_available, is_nvfp4_available

        self.assertTrue(torch.cuda.is_available(), "GPU acceptance requires a CUDA device")
        self.assertIn(
            "B200", torch.cuda.get_device_name(), "Acceptance must run on the requested B200"
        )
        self.assertTrue(is_mxfp8_available())
        self.assertTrue(is_nvfp4_available())
        torch.manual_seed(1234)
        source_x = torch.randn(128, 256, device="cuda", dtype=torch.bfloat16)
        source_w = torch.randn(512, 256, device="cuda", dtype=torch.bfloat16) * 0.03
        dy = torch.randn(128, 512, device="cuda", dtype=torch.bfloat16)
        x_ref = source_x.clone().requires_grad_()
        w_ref = source_w.clone().requires_grad_()
        y_ref = functional.linear(x_ref, w_ref)
        y_ref.backward(dy)
        results = []
        for name, recipe, bound in (
            ("bf16", None, 0.02),
            ("mxfp8", MXFP8BlockScaling(), 0.12),
            ("nvfp4", NVFP4BlockScaling(), 0.35),
        ):
            with self.subTest(precision=name):
                linear = te.Linear(256, 512, bias=False, params_dtype=torch.bfloat16, device="cuda")
                with torch.no_grad():
                    linear.weight.copy_(source_w)
                x = source_x.clone().requires_grad_()
                with te.autocast(enabled=recipe is not None, recipe=recipe):
                    y = linear(x)
                y.backward(dy)
                errors = {}
                for role, actual, expected in (
                    ("fprop", y, y_ref),
                    ("dgrad", x.grad, x_ref.grad),
                    ("wgrad", linear.weight.grad, w_ref.grad),
                ):
                    actual, expected = actual.detach().float(), expected.detach().float()
                    self.assertTrue(torch.isfinite(actual).all())
                    relative = ((actual - expected).norm() / expected.norm()).item()
                    cosine = functional.cosine_similarity(
                        actual.flatten(), expected.flatten(), dim=0
                    ).item()
                    self.assertLess(relative, bound, f"{name} {role} relative error")
                    self.assertGreater(cosine, 0.9, f"{name} {role} gradient/output direction")
                    errors[role] = {"relative_l2": relative, "cosine": cosine}
                results.append({"precision": name, "errors": errors})
        import json

        print("B200 numerical reference:", json.dumps(results), flush=True)


if __name__ == "__main__":
    unittest.main()
