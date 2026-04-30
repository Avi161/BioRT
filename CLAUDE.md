# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What to read vs avoid (agents)

- **Prompts.** When inspecting prompt structure or examples, use **`prompts/mock_prompts.json`** only. Do not open, search, or summarize other prompt dataset files (e.g. paths under `backend/prompts/`) unless a human explicitly points you at a specific path for a local run.
- **Red-team JSONL results.** When you need the shape of runner output, use **`results_example/`** only. It follows the same directory layout as live outputs under **`results/`** (see below). Do **not** read, glob, or index **`results/`** — it is gitignored and may contain real jailbreak transcripts and sensitive model responses.
- **Victim model folder: production vs example.** Live matrix output lives under **`results/<victim_model_slug>/...`**. That segment is **not** free-form: in this project it is exactly one of **`claude_sonnet_4_6`**, **`deepseek_v4_flash`**, **`gemini_3_pro`**, **`gpt_5_4`**, or **`kimi_k2_5`** (same spelling and underscores as in the model registry / on-disk trees — do not substitute variants like `gpt-5.4` or `Claude_Sonnet`). **`example_victim`** is **only** for committed fixtures under **`results_example/`** so examples are never confused with a real model’s run directory. Do **not** rename `results_example/`’s folder to one of the five production slugs unless a maintainer explicitly chooses to mirror a real path (the default is **`example_victim`**).
- **Synthetic example folder (exact string).** Under **`results_example/`**, the canonical directory name is **`example_victim`** (all lowercase, underscore). When adding paths or docs for fixtures, keep that exact string — avoid `ExampleVictim`, `example-victim`, etc. If you add a second fixture tree, pick another **visibly fake** slug that is **not** one of the five production names above.

**Layout parity:** `results_example/example_victim/<method>/` parallels `results/<one_of_five_slugs_above>/<method>/`, where `<method>` is one of `base64`, `direct`, `pair`, or `crescendo` (crescendo examples may be added later). Filenames are typically `<method>_<line_count>_<yymmdd>_prompts_short.jsonl` in `results_example/`; production runs often use `<count>_<yymmdd>_prompts_short.jsonl` without the method prefix, but the JSON object schema is the same.

## Project

Hackathon harness for measuring bio-misuse safeguards across frontier AI models. Wraps Microsoft's [PyRIT](https://github.com/Azure/PyRIT) with a model registry, attack matrix, and DuckDB-backed memory. Output is a heatmap of attack success rates over the matrix `Models × Attack Methods × Categories`.

48-hour hackathon scope — prefer clean, functional, modular scripts over abstraction. The `init-setup.md` file is the original brief and pins the architectural constraints (DuckDB only, LiteLLM for API normalization, PyRIT-native orchestrators).

## Commands

```bash
# Setup
cd backend
python -m venv .venv && source .venv/bin/activate
pip install -r ../requirements.txt
cp ../.env.example ../.env   # then fill in any subset of API keys

# Smoke test (single model, 5 prompts, prints memory contents)
python hello_world.py

# Per-attack live smoke test — writes one JSON per attack to results/{provider}/{method}/
python validate_attacks.py

# Full matrix (all available models × 4 attack methods × all prompts)
python matrix_runner.py

# Tests
pytest                                  # full suite
pytest tests/test_config.py             # one file
pytest tests/test_config.py::TestModelRegistry::test_registry_has_at_least_five_models   # one test
```

Both runner scripts call `load_dotenv()` at import time and skip any model whose API key env var is empty — at least one key is enough to run.

### Crescendo debug: resume progress

Partial JSONL from a stopped `--crescendo-debug` run is **not** lost: each line is a finished cell. To **continue** without re-running completed prompts, use **`--crescendo-resume-from-jsonl PATH`** and keep the **same** pairing and prompt source as the original run (e.g. same `--crescendo-kimi-attacks-anthropic` and same `--prompt-file` or `--crescendo-bench`).

**Append to the same artifact** (recommended): set **`--debug-output`** to the **exact** `.jsonl` file (not just the `results/crescendo/` directory), and use that same path for **`--crescendo-resume-from-jsonl`**. The runner reads `metadata.prompt_id` from the file, skips those IDs, and runs the remaining BioRT prompts in file order. Implementation: `load_completed_prompt_ids_from_jsonl` and `skip_prompt_ids` in `backend/crescendo_debug.py`.

**Example (Kimi attacks, Anthropic defends, long bench, append + resume from repo root):**

```bash
python backend/matrix_runner.py \
  --crescendo-debug --crescendo-debug-full \
  --crescendo-kimi-attacks-anthropic \
  --prompt-file prompts/prompts_long.json \
  --debug-output results/crescendo/crescendo_defender-anthropic_attacker-moonshot_prompts_long_<STAMP>.jsonl \
  --crescendo-resume-from-jsonl results/crescendo/crescendo_defender-anthropic_attacker-moonshot_prompts_long_<STAMP>.jsonl
```

If every `prompt_id` is already in the file, the run has nothing left to do. If a prompt was in progress and never got a line, that `prompt_id` is not skipped and will run on resume.

## Architecture

Three layers, all gluing PyRIT primitives together:

**1. Model registry — `backend/config/models.py`**

- `MODEL_REGISTRY`: hard-coded list of **7** `ModelConfig` entries (DeepSeek V4 Flash, Claude Sonnet 4.6, GPT-5.4, Gemini 3 Pro, Grok 4, Kimi K2.5, Llama-3.3 70B). Each carries display name, provider, OpenAI-compatible endpoint, env var, model id, optional `temperature` / `extra_body`, and a `role` hint (`frontier_closed` | `frontier_supplementary` | `open_weight_control`).
- `get_available_models()` filters the registry by which env vars are set; emits a single `Active: [...] | Skipped: [...]` banner at INFO; raises `EnvironmentError` if none are present.
- `build_target(cfg)` returns a `pyrit.prompt_target.OpenAIChatTarget` for every registered provider — all of them expose OpenAI-compatible chat/completions endpoints, including **Anthropic via its OpenAI SDK compatibility layer** (`https://api.anthropic.com/v1/`), xAI natively, and Together.ai natively. If a future provider does **not** expose an OpenAI-compatible API, raise `NotImplementedError` in `build_target` — the runners catch and skip it.

**2. Attack library — `backend/attacks.py`**

- `ATTACK_METHODS` is a `dict[str, Callable]` of factory functions, each returning a fresh PyRIT attack instance per cell. Adding a new method = add a `_build_<name>_attack` function and register it in `ATTACK_METHODS`. Factories take `(target, adversary, objective)` even when they don't need all three — keep the signature uniform.
- Current methods: `direct` (`PromptSendingAttack`), `base64` (`PromptSendingAttack` + `Base64Converter`), `pair` (`RedTeamingAttack` with `RTASystemPromptPaths.TEXT_GENERATION`, `max_turns=3`), `crescendo` (`CrescendoAttack`, `max_turns=5`, `max_backtracks=5`).
- PAIR and Crescendo currently use a `SelfAskTrueFalseScorer` **placeholder** — these are the StrongREJECT injection points, marked with `# HOOK: StrongREJECT scoring` comments. Both factories also `logger.warning` at build time and the result records carry `scorer="placeholder_self_ask"` (via `attacks.PLACEHOLDER_SCORER`) so a downstream reader can never confuse placeholder output with calibrated StrongREJECT output. `attacks.METHODS_REQUIRING_ADVERSARY` lists which methods are scientifically invalid when target == adversary; `backend/validate_attacks.py` and `backend/matrix_runner.py` both refuse to run them in that configuration.

**3. Matrix runner — `backend/matrix_runner.py`**

- Adversary / scorer LLM defaults to provider `moonshot` (Kimi) and can be overridden with `ADVERSARY_PROVIDER` in `.env`. The runner skips PAIR/Crescendo cells where the only buildable target equals the adversary (target-vs-target is invalid).
- `run_single_cell()` wraps each attack execution in a retry loop (`MAX_RETRIES=3`, exponential backoff from `BASE_BACKOFF_SECONDS=5`). On exhausted retries it returns a `CellResult` populated with `status`, `error_type`, `error_message`, `traceback`, and real `elapsed_seconds` — the matrix never aborts mid-run, and failures are post-mortem-friendly. Memory labels (`model`, `method`, `category`) are attached to every conversation so they can be filtered out of `CentralMemory` later. Adversary identity and scorer id are stamped onto every `CellResult`.

**4. Memory layer — PyRIT `CentralMemory`**

- Both runners call `initialize_pyrit_async(memory_db_type=IN_MEMORY)`. Despite the spec calling for DuckDB persistence, `IN_MEMORY` is currently SQLite under the hood and **does not persist to disk** — switch to `DUCK_DB` (and supply a path) when results need to outlive the process. `results/` (repo root) is gitignored for JSON outputs from `backend/validate_attacks.py`, `backend/matrix_runner.py`, and similar tools — agents should infer shape from **`results_example/`** instead. `eval_results/` (repo root) holds judge outputs from `backend/score_results.py`.
- Conversations are retrieved via `CentralMemory.get_memory_instance().get_message_pieces(conversation_id=...)`.

## Evaluation Frameworks

Two academic frameworks inform the scoring and experimental design:

**HarmBench (evaluation methodology)**

- Defines the red teaming task as designing test cases to elicit harmful behavior from a target model.
- **Primary metric:** Attack Success Rate (ASR) — percentage of test cases that elicit the target behavior.
- **Standardized generation length:** 512 tokens. Varying this can shift ASR by up to 30%.
- **Classifier:** fine-tuned Llama 2 13B for robust success/failure classification (catches "refuse then comply" and broken-code edge cases).
- **Key finding:** robustness is independent of model size — training data and alignment algorithms matter more than parameter count.
- Evaluates 18 methods across white-box (GCG, PEZ), black-box (PAIR, TAP), and transfer categories.
- Our PAIR and Crescendo methods map directly to HarmBench's black-box/prompt-based attack category.

**RIOT (Red-teaming with In-context Optimization) — scoring methodology**

- Treats red-teaming as an in-context optimization problem with three roles:
  - **Attacker LLM:** generates and iteratively refines jailbreak prompts (our adversary target).
  - **Target LLM:** the model under test (our objective target).
  - **Judge LLM:** scores responses 1–10 and provides reasoning (this is where StrongREJECT plugs in).
- Optimization loop: Attacker generates prompt → Target responds → Judge scores → Attacker refines using full history.
- Score of 10 = full jailbreak with detailed harmful response; low scores = refusal detected.
- Near 100% ASR on several aligned models; attacks transfer across model families.
- The `# HOOK: StrongREJECT scoring` markers in `backend/attacks.py` are the Judge integration points.

**How they connect to our codebase:**

- HarmBench provides the evaluation framework (ASR metric, standardized conditions, 512-token generation).
- RIOT's Attacker-Target-Judge triad maps to our adversary → target → scorer architecture in PAIR/Crescendo.
- The `SelfAskTrueFalseScorer` placeholder in `backend/attacks.py` will be replaced with a bio-domain-calibrated StrongREJECT judge.
- `PLACEHOLDER_SCORER_METHODS` and the `scorer="placeholder_self_ask"` stamp on result records ensure no placeholder scores are mistaken for calibrated output.

## Conventions

- **Prompt datasets.** For Claude / codebase exploration, use **`prompts/mock_prompts.json`** (abstract placeholders per category). Production runs often point `--prompt-file` at files under **`backend/prompts/`** (e.g. `prompts_long.json` — full BioRT-Bench schema with `prompt_id`, `prompt_text`, `abstraction_check`); that tree is commonly gitignored and should not be opened by agents unless explicitly requested. The five paper-aligned categories are `synthesis_evasion`, `lab_automation_uplift`, `dual_use_protocol`, `acquisition_kyc`, `virology`. `backend/tests/test_prompts.py::EXPECTED_CATEGORIES` enforces this shape — update the test if the schema changes intentionally.
- **Add a model:** append a `ModelConfig` to `MODEL_REGISTRY`, add the env var to `.env.example`, and confirm `backend/tests/test_config.py::test_registry_has_at_least_five_models` still passes (it asserts `>= 5`, so adding more is free). If the provider isn't OpenAI-compatible, raise `NotImplementedError` in `build_target` so the runners can skip it.
- **Add an attack method:** write a `_build_<name>_attack(target, adversary, objective)` function in `backend/attacks.py`, register it in `ATTACK_METHODS`. If it uses a placeholder scorer, add it to `PLACEHOLDER_SCORER_METHODS`. If it requires a separate adversary, add it to `METHODS_REQUIRING_ADVERSARY`.
- **Async everywhere.** All PyRIT entry points are `async`; both scripts use `asyncio.run(main())`. Don't mix sync PyRIT calls into the async path.
- **Token cost awareness.** PAIR uses ~2-3x tokens per cell (adversary + target + scorer). Crescendo uses 5+ turns (capped at 5/5, aligned with common paper reporting). Adversary defaults to Kimi via `ADVERSARY_PROVIDER=moonshot`; keep it explicit in `.env` for reproducibility.
- Python 3.10+ (`from __future__ import annotations` + PEP 604 union syntax used throughout).
