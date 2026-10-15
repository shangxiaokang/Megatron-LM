# Automatic static precision search

This tool automatically diagnoses quantization sensitivity, runs matched training
ablations, and exports a static per-module BF16/MXFP8/NVFP4 configuration. The
optimizer, learning-rate schedule, training data position, batch size, parallelism,
and parameter-storage policy are fixed. Only module compute precision is searched.

## Run

From the Megatron-LM checkout, wrap the existing training command:

```bash
python -m tools.precision_search --output /scratch/precision-search -- \
  python -m torch.distributed.run --nproc-per-node 8 pretrain_gpt.py \
  <your existing training arguments>
```

No per-stage settings or manual layer selection are required. Optional controls:

- `--budget-minutes`: total wall-clock budget, default 60 minutes. It includes
  startup, sampling, training branches, and validation. Reserve cluster resources
  for this budget plus container startup and cleanup.
- `--max-trials`: maximum number of training branches, default 32.
- `--quality-tolerance`: relative increase of validation loss allowed against the
  matched BF16 reference, default 0.005 (0.5%). This is a user quality constraint,
  not an error threshold inferred from tensor noise.
  Validation uses each candidate's training precision, so the constraint includes
  direct forward quantization error as well as the effect of quantized training
  updates. For a 0.1% limit, set `--quality-tolerance 0.001`.
- `--metric`: validation loss key to minimize. Automatically selects `lm loss`,
  or the first available loss key.
- `--resume`: reuse completed trials from an interrupted search. Command, source,
  working directory, checkpoint metadata, and relevant TE environment must match.
  Provide a fresh wall-clock budget for the resumed invocation.

The original `--train-iters` is retained for scheduler construction. Each worker
ends its branch early rather than shortening the learning-rate schedule. By
default diagnosis uses 8 steps, search branches 16, and final branches 64; the
tool adapts to CLI CUDA Graph warmup and the remaining original training horizon.

All branches use the same initialization seed or the same loaded checkpoint and
optimizer/RNG state. The tool checks sampled initial weights, training position,
seed, and iteration across branches. Keep the source checkpoint immutable during
a search. Save and logging directories are redirected into each trial directory;
the source checkpoint is never overwritten.

Matched initial RNG state does not imply identical random draws throughout the
branches: TE's NVFP4 stochastic rounding advances the default CUDA generator,
which can affect later dropout draws. Ablations measure the effect of deploying
the complete format policy, including its original RNG behavior; they do not
isolate rounding error alone or establish robustness across training seeds.

## Automatic stages

1. A BF16 reference and supported global low-precision baselines sample module
   output, output/input gradient, and first-update vectors. Normalized L2 and cosine
   errors prioritize candidates. These sparse network-level samples include
   upstream error propagation and do not establish local causal sensitivity.
2. Module type/depth groups and individual modules are automatically ablated.
   A budgeted bidirectional beam search explores upgrades and downgrades. Actual
   paired validation loss determines feasibility; actual steady-state training
   timing determines speed. Tensor reconstruction error does not decide quality.
3. Up to two promising policies are retrained against a matched BF16 reference on
   a longer branch, using a separate validation sample range. A conservative
   paired block-mean screening bound gates export. The final timing has no
   diagnostic hooks. A verified BF16 policy is the fallback when faster policies
   fail final validation. Insufficient evidence produces `not_verified` and no
   new configuration.

Timing excludes warmup and synchronizes CUDA around each complete `train_step`,
including the optimizer update. Distributed timing takes the slowest rank at each
step. This is a consistent latency benchmark; throughput with a different degree
of overlap should be checked in the actual deployment configuration.

The search space is finite but not exhaustively enumerated. Results identify the
best verified policy found in the budget, not a certified global optimum. Validation
across short branches does not establish full-training convergence or robustness
across later checkpoints; the report explicitly records the tested horizon.

## Results and deployment

- `best_precision.json`: JSON, which is valid YAML and accepted by the existing
  `--te-precision-config-file` loader. Exact module matchers precede the default.
- `report.json`: selected policy, quality evidence, step timing, memory, all trials,
  and the scope of the result.
- `sensitivity.json` and `ablation-*.json`: sampling and causal ablation evidence.
- `trials/*/training.log` and `rank-*.json`: commands, software/hardware versions,
  rank results, and failure reasons.

Progress and selection reports omit raw diagnostic vectors. The complete vectors
remain in each diagnostic trial's `result.json` and rank artifacts, and sensitivity
ranking uses those original measurements. This keeps reports practical for large
MoE module inventories without losing the underlying evidence.

For models that combine grouped MoE GEMMs and ordinary linear GEMMs, the search
also measures BF16 protection of the grouped backend with MXFP8/NVFP4 elsewhere.
These automatic candidates account for different quantization/padding costs;
diagnostic sensitivity alone cannot predict backend performance. Unsupported
formats fall back to BF16, and these candidates use the same loss and holdout gates.

Add the exported configuration to the original training command:

```bash
<original training command> --te-precision-config-file /scratch/precision-search/best_precision.json
```

The exported recipes inherit original parameter storage. The configurable modules
use their selected training precision during evaluation, so validation includes
direct forward quantization error. This evaluation policy is also in the exported config.
Other operations, including attention and non-configurable fused modules, keep
their original configuration. The report's inventory defines the searched scope.

## Supported scope

The first implementation assigns one recipe to each configurable TELinear,
TELayerNormLinear, or TEGroupedLinear module, covering that module's forward and
backward together. It does not search optimizer settings, storage precision,
attention precision, RHT/scaling parameters, or independent GEMM directions.

The adapter requires BF16 base training, actual CUDA/TE support, ordinary
supervised training, and a single bounded validation set. Sixteen validation
batches are sampled automatically per evaluation, bounding search cost even when
production validation is much longer. Every rank must finish the requested steps;
skipped optimizer updates, incomplete rank artifacts, nonfinite measurements, and
missing validation evidence reject a trial. Unsupported kernel/shape/backend
combinations remain failed trials with the original error logs. A failed global
recipe does not remove all mixed assignments of that format from the search.

CUDA Graphs, activation recompute, MoE, TP/PP/FSDP, and quantized parameter storage
retain their original settings, but require validation in the actual execution
configuration; this tool does not infer support from a successful different setup.
Graph diagnostics sample an existing eager warmup step; without one, CPU sampling
is omitted automatically and training ablations provide the evidence. The existing
mixed-precision backend has additional limitations described in
[TransformerEngineMixedPrecision.md](../../../megatron/core/extensions/TransformerEngineMixedPrecision.md).

## Engineering validation

Dependency-free controller tests:

```bash
python -m unittest tools.precision_search.tests.test_search
```

On the requested B200, test native-reference forward/dgrad/wgrad numerics and run
all stages with a small actual Megatron GPT (mock-token data), followed by an
uninstrumented static-config replay:

```bash
python -m unittest tools.precision_search.tests.test_gpu
python -m examples.precision_search.acceptance /scratch/precision-search-acceptance
```

The acceptance command also creates a checkpoint and checks matched BF16/MXFP8/
NVFP4 branches restored from it. On two B200 GPUs, additionally run:

```bash
python -m examples.precision_search.distributed_acceptance /scratch/precision-search-distributed
```

This exercises DP=2, TP=2, and PP=2 with all three formats, including distributed
optimizer shards, global module inventory, and rank-complete measurements. Each
mode also checks a mixed NVFP4/BF16 policy that protects the second global layer's
FC1; PP repeats that policy as an uninstrumented static training command. Module
precision matchers use global decoder layer indices, including PP/VP offsets.

The mock-data integration test validates engineering behavior; its speedup and
quality measurements are not evidence of convergence on a production dataset.
