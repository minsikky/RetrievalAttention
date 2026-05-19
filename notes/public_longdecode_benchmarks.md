# Public Long-Decode Benchmark Setup

Goal: evaluate dense vs current paged-PQ frontier on public benchmarks where generated tokens matter, not only long-prefill retrieval.

## Benchmark Set

| category | primary benchmark | runner suite | why it is included |
| --- | --- | --- | --- |
| Coding | LiveCodeBench code generation | `livecodebench_codegen` | Public, contamination-aware coding benchmark. Generated solution code gives a real decode workload. |
| Reasoning | AIME 2024 | `aime24` | Short prompts, potentially long reasoning generations, exact integer scoring. |
| Long generation | LongGenBench SGT short/long | `longgenbench_sgt_short`, `longgenbench_sgt_long` | Direct 16K/32K-style long-form generation stress; public benchmark specifically targets long generated outputs. |
| Long generation, auto-score | LongGenBench GSM8K compound | `longgenbench_gsm8k` | Easier automated scoring than SGT because subanswers are numeric. Useful as a fast sanity track. |

## Local Artifacts

The benchmark repositories were cloned under ignored `third_party/benchmarks/`:

| repo | local path | source |
| --- | --- | --- |
| LiveCodeBench | `third_party/benchmarks/LiveCodeBench` | `https://github.com/LiveCodeBench/LiveCodeBench` |
| LongGenBench, long-context QA generation | `third_party/benchmarks/LongGenBench_dominic` | `https://github.com/Dominic789654/LongGenBench` |
| LongGenBench SGT long-form generation | `third_party/benchmarks/LongGenBench_mozhu` | `https://github.com/mozhu621/LongGenBench` |

## Runner

Main runner:

```bash
benchmark/public_longdecode_eval.py
```

Slurm wrapper:

```bash
sbatch benchmark/run_public_longdecode_hf.sh
```

Matrix submitter:

```bash
bash scripts/submit_public_longdecode_matrix.sh
```

The wrapper accepts `BENCHMARK`, `ATTENTION_MODE=dense|pagedpq`, `MAX_EXAMPLES`, `MAX_NEW_TOKENS`, `MIN_NEW_TOKENS`, and the existing paged-PQ frontier knobs.

Default model target:

- `HF_MODEL_PRESET=qwen3_8b` is the default and resolves to cached snapshot `.hf_cache/hub/models--Qwen--Qwen3-8B/snapshots/b968826d9c46dd6066d109eabc6255188de91218`.
- `HF_MODEL_PRESET=llama31_8b` resolves to cached snapshot `.hf_cache/hub/models--meta-llama--Llama-3.1-8B-Instruct/snapshots/0e9e39f249a16976918f6564b8830bc894c89659`.
- Qwen presets auto-enable `.hf_pydeps`, because the repo venv's pinned Transformers does not recognize Qwen3. The Llama preset uses the repo venv by default.
- Qwen3 YaRN default uses `QWEN_YARN_ORIGINAL_MAX_POSITION_EMBEDDINGS=32768`; Qwen3.5 preset uses `262144`.
- The paged-PQ HF attention patch handles Qwen3's `past_key_values` cache argument and `q_norm/k_norm`; otherwise Qwen3 silently falls through to native attention.

Examples:

```bash
HF_MODEL_PRESET=qwen3_8b bash scripts/submit_public_longdecode_matrix.sh
HF_MODEL_PRESET=llama31_8b bash scripts/submit_public_longdecode_matrix.sh
```

## First Validation Plan

1. Smoke: `MAX_EXAMPLES=1` for `aime24`, `livecodebench_codegen`, and `longgenbench_sgt_short`.
2. Sampled validation: `MAX_EXAMPLES=8-16`, dense vs paged-PQ, with `max_new_tokens` large enough to exercise decode.
3. Full validation: LiveCodeBench release slice, AIME full set, LongGenBench short/long subsets.

## Caveats

- LiveCodeBench can be fully scored with official pass@1 when `EVALUATE_CODE=1`; code execution is sandboxed by the benchmark harness but still should be run only on Slurm.
- LongGenBench SGT official quality uses an LLM yes/no judge. The local runner currently records completion rate and cheap substring smoke metrics; use the official judge path before treating SGT scores as paper numbers.
- LongGenBench SGT should use `FORCE_MAX_NEW_TOKENS=1` and nonzero `MIN_NEW_TOKENS` for true long-decode stress, otherwise instruction-following models may stop early.
