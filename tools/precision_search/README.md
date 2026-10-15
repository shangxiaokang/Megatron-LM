# Precision search

The default `beam` mode measures BF16, MXFP8 and NVFP4 module assignments and
selects the fastest verified policy. BF16 remains eligible, so a slower
low-precision policy can pass quality gates without being selected.

## Repair a low-precision baseline

`repair` starts from an explicit `{default, overrides}` module policy. Existing
BF16 protections are retained. Every proposed change restores a group of
low-precision modules to BF16; it never lowers an already-protected module or
changes the original low-precision format of another module.

Each ablation restarts from the same checkpoint, optimizer, seed and training
data position. Compare the parent and candidate on matched training/validation
loss sequences. Keep a restoration only when **both** mean losses improve by
at least `--min-loss-improvement * mean(BF16 loss)`. This threshold is a practical
screening criterion, not a statistical significance test. Groups are partitioned
by backend family, operator and depth; the most useful group is refined into
individual modules when budget permits. A joint group benefit does not prove
each member is individually sensitive.

Use an explicit coverage constraint to keep low precision dominant. Coverage is
the fraction of configurable module names assigned MXFP8/NVFP4, **not** parameter
count or GEMM FLOPs. Forced BF16 modules count in its denominator.

```bash
python -m tools.precision_search \
  --output results/repair \
  --strategy repair --base-policy baseline-policy.json \
  --quality-tolerance 0.005 --quality-criterion mean \
  --min-low-precision-fraction 0.8 --require-mixed \
  --min-loss-improvement 0.0001 \
  --repair-steps 8 --short-steps 16 --final-steps 128 \
  --budget-minutes 210 --max-trials 48 \
  -- python -m torch.distributed.run ... pretrain_gpt.py ...
```

Baseline policy format:

```json
{"default": "nvfp4", "overrides": {"decoder.layers.19.mlp.experts.linear_fc1": "bf16"}}
```

The baseline file represents standard module-format assignments. It does not
accept arbitrary TE recipe dictionaries or silently preserve custom RHT,
stochastic-rounding, role/factory or quantizer parameters. GEMM roles within a
module and individual experts within GroupedLinear are not independent search
variables in this controller.

Repair selection prefers fewer added BF16 modules among quality-passing
restorations, then lower validation loss and measured step time. It permits a
mixed policy to be slower than BF16. The original baseline remains eligible if
no eligible measured restoration improves it. BF16 is only a reference in repair
mode and cannot be exported as a successful fallback.

Candidates are locked before independent holdout. Fresh branches compare BF16,
the original low-precision baseline and repaired policies. Final acceptance
requires both training and validation gates, coverage, and a measured holdout
improvement over the baseline for a changed policy. The paired block bound is
reported alongside the mean gate. Missing training loss, incomplete ranks,
nonfinite values or failed holdout produces `not_verified` and no recipe export.
The generic controller may shorten confirmation to the original training horizon;
experiments promising a fixed step count must enforce that count separately.

`protection-ablation.json` records the measured parent/candidate deltas.
`accepted_promotions` describes the greedy exploration path;
`selected_protection_ablation` describes the actual exported policy's provenance.

CPU controller contracts can run without CUDA:

```bash
python -m unittest tools.precision_search.tests.test_search \
  tools.precision_search.tests.test_repair \
  tools.precision_search.tests.test_training_measurements
```

These synthetic contracts validate search logic. Actual model loss, throughput,
static deployment and long-run convergence require separate GPU evidence.
