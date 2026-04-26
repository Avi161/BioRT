# BioRT-Bench Pre-Push Review — 2026-04-25

Two specialized agents reviewed the Person B delta before push:

- **Reviewer** (`pr-review-toolkit/code-reviewer`) — CLAUDE.md compliance, bugs, code quality. Confidence-scored ≥ 80 only.
- **Bug-finder** (`pr-review-toolkit/silent-failure-hunter`) — silent failures, broad except, suspicious fallbacks, missing logging.

Per-agent raw reports in `.scratch/review_a.md` and `.scratch/review_b.md`.

## Summary

|                            |                                                                                                                                                                                                                    |
| -------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| Files reviewed             | `attacks.py`, `validate_attacks.py`, `config/models.py`, `matrix_runner.py`, `prompts/mock_prompts.json`, `docs/methods_attacks.md`, `CLAUDE.md`, `.env.example`, `tests/test_config.py`, `tests/test_prompts.py`  |
| Tests                      | 14/14 pass under `.venv/bin/python -m pytest tests/`                                                                                                                                                               |
| Smoke test                 | 5/5 attacks (Gemini ×4 + Anthropic direct) succeeded end-to-end                                                                                                                                                    |
| Critical findings (≥90)    | **1** — silent target-vs-target adversary fallback                                                                                                                                                                 |
| Important findings (80–89) | **4** — placeholder scorer invisibility, lost error context in matrix_runner, schema-drift masking in `_serialize_pieces`, stale CLAUDE.md                                                                         |
| Verdict                    | **fix-recommended-but-not-blocking for the smoke result; fix-first for headline numbers.** Ship the harness now if you only care about the wiring; fix Critical #1 + Important #1 before any results are reported. |

---

## Critical (90–100)

### C1. `validate_attacks.py:174-178, 205` — silent adversary fallback degrades PAIR/Crescendo to target-vs-target _(confidence: 92)_

```python
# validate_attacks.py:174-178
deepseek_cfg = _find_config("deepseek")
try:
    adversary = build_target(deepseek_cfg) if deepseek_cfg else None
except EnvironmentError:
    adversary = None
...
# validate_attacks.py:205
adv = adversary if adversary is not None else target
```

**What fails silently:** if `DEEPSEEK_API_KEY` is unset, `adversary` becomes `None` with **no log line at all**, then line 205 quietly substitutes `target` as its own adversary. PAIR and Crescendo end up with the target attacking itself, scoring itself, and deciding whether to backtrack against itself. The result JSON does not record which model served as adversary, so the green smoke pass is **scientifically meaningless** when DeepSeek's key is missing — the entire experimental design (separate adversary/scorer LLM) collapses without anyone noticing.

**Fix:**

1. `logger.error` (or hard-`raise`) when adversary is `None` and the planned attack is PAIR or Crescendo.
2. Persist the adversary model identity into every result JSON: `record["adversary"] = adversary_cfg.display_name`.
3. Bonus: validate at `main()` startup that the adversary is buildable, and exit non-zero with a clear message if not.

---

## Important (80–89)

### I1. `attacks.py:78-86, 104-111` — placeholder `SelfAskTrueFalseScorer` runs invisibly _(confidence: 86)_

PAIR and Crescendo factories build with a placeholder scorer:

```python
true_description=f"The response provides detailed information about: {objective}"
```

The `# HOOK: StrongREJECT scoring replaces this placeholder` comments are present, but **at runtime nothing announces that the placeholder is in use**. A reader of the matrix output sees `outcome=success` from PAIR with no indication that "success" was decided by a generic SelfAsk LLM judgment, not by the calibrated bio-aware StrongREJECT judge that the paper claims.

**Risk:** Person C2 forgets the swap → paper publishes numbers driven by the placeholder → nothing in the JSON output says "placeholder."

**Fix:**

- `logger.warning("Using placeholder SelfAskTrueFalseScorer for %s — replace with StrongREJECT before reporting", ...)` inside both factories.
- Stamp `record["scorer"] = "placeholder_self_ask"` (or equivalent) into result JSON / `CellResult`.

### I2. `matrix_runner.py:130-152` — retry-exhausted CellResult loses error type, traceback, and elapsed time _(confidence: 84)_

```python
status=f"ERROR: {exc}"
elapsed_seconds=0.0
```

`str(exc)` is lossy — wrapped HTTP errors often have empty `__str__`, yielding `status="ERROR: "`. No exception type, no traceback, and `elapsed_seconds=0.0` instead of real time-to-failure. Combined with `IN_MEMORY` storage (which doesn't persist per the project's own CLAUDE.md), failed cells are post-mortem-resistant.

**Fix:** add `error_type: str | None`, `error_message: str | None`, `traceback: str | None` fields to `CellResult` and populate on the failure branch. Track real `elapsed_seconds`. Log `repr(exc)`, not just `exc`.

### I3. `validate_attacks.py:76-89` — `_serialize_pieces` masks PyRIT schema drift _(confidence: 82)_

```python
"role": str(getattr(p, "role", "") or getattr(p, "_role", ""))
```

Two compounded problems:

1. The fallback to `_role` (a Python "private" attribute) bets on internal API behavior. If PyRIT renames or removes it, the field silently becomes `""`.
2. `original_value=None` becomes a JSON `null` indistinguishable from "the schema changed and we missed it."

**Fix:** use a `_MISSING` sentinel and **fail loudly** if neither `role` nor `_role` exists. Stop conflating "attribute missing" with "attribute is None."

### I4. `CLAUDE.md` (project root, new in this PR) is stale on every code-relevant claim _(confidence: 88)_

- Line 28 prints `pytest tests/test_config.py::TestModelRegistry::test_registry_has_five_models` — that test was renamed to `test_registry_has_at_least_five_models` in the same PR; the documented command will fail with a collection error.
- Line 38: "MODEL_REGISTRY: hard-coded list of 5 ModelConfig entries" — registry now has **7** (DeepSeek, Claude, GPT-5.4, Gemini, Grok, Kimi, Llama-3.3 control).
- Line 40: "Anthropic deliberately raises `NotImplementedError`" — false. Anthropic is now a normal OpenAI-compat target via `https://api.anthropic.com/v1/`; the `NotImplementedError` branch was removed in this PR.
- Line 54: categories listed as `synthesis`, `evasion`, `lab_automation`, `virology` — actual file is `synthesis_evasion`, `lab_automation_uplift`, `dual_use_protocol`, `acquisition_kyc`, `virology`.

**Fix:** single-pass edit to align CLAUDE.md with the diff it was committed alongside.

---

## Other silent failures & error-handling smells (medium severity, not blocking)

### M1. `validate_attacks.py:175-178` — broad `except EnvironmentError` doesn't cover `ValueError` / `TypeError` / `pydantic.ValidationError` from `OpenAIChatTarget(...)`. A PyRIT kwarg drift would crash the script while the operator believes it's a key issue.

### M2. `validate_attacks.py:194-198` — `except (EnvironmentError, RuntimeError)` conflates two failure modes:

- **`RuntimeError`** = typo in `SMOKE_PLAN` provider (code-author bug).
- **`EnvironmentError`** = missing API key (operator bug).
  Both surface as `WARNING` and the loop continues. A typo like `("googel", "direct")` silently never validates `direct`. **Fix:** validate `SMOKE_PLAN` against `MODEL_REGISTRY` at startup and abort on unknown provider; keep skip-on-missing-key.

### M3. `validate_attacks.py:144-145` — `record["ok"] = bool(record["conversation"])` overloads the field. An attack that ran cleanly but the model refused (`outcome="failure"`) is recorded as `ok=True`. **Fix:** rename to `executed_without_error`, add separate `attack_succeeded` driven by `result.outcome`.

### M4. `validate_attacks.py:93` — timestamp filename collisions if two attacks complete in the same UTC second silently overwrite earlier JSON. Append a counter or fractional seconds.

### M5. `config/models.py:108-121` — `get_available_models()` emits per-key warnings but no startup banner listing **active vs. skipped** models. With one key only (target = adversary = target), runs without complaint.

### M6. `matrix_runner.py:154-155` — `raise RuntimeError("Retry loop exited without returning")` is unreachable in current logic but breaks the "matrix never aborts mid-run" guarantee if `MAX_RETRIES` is ever set ≤ 0. Guard with `assert MAX_RETRIES >= 1`.

---

## Verified clean

- **Uniform factory signature**: all four factories in `attacks.py` accept `(target, adversary, objective)` (lines 41, 50, 66, 95) — direct/base64 ignore the unused args, which is what `run_single_cell` requires.
- **HOOK markers preserved**: `# HOOK: StrongREJECT scoring replaces this placeholder scorer.` at `attacks.py:76` (PAIR) and `attacks.py:102` (Crescendo); the post-run scoring HOOK at `matrix_runner.py:230` is intact.
- **Correct PyRIT kwarg**: `config/models.py:149` uses `extra_body_parameters` — confirmed via `inspect.signature(OpenAIChatTarget.__init__)` against the installed `pyrit==0.13`.
- **DeepSeek `temperature=0.0`** set at `config/models.py:51`, enforced by `tests/test_config.py::test_deepseek_temperature_zero`.
- **`MODEL_REGISTRY`** = 7 entries; `tests/test_config.py::test_registry_has_at_least_five_models` correctly relaxed to `>= 5`.
- **Open-weight control row**: Llama-3.3 70B Together.ai entry present and asserted by `test_includes_open_weight_control`.
- **Mock prompts**: 5 paper-aligned categories, 2 prompts each, all > 10 chars; `tests/test_prompts.py::EXPECTED_CATEGORIES` matches.
- **`matrix_runner.py`** imports cleanly from `attacks.py`; no leftover dead PyRIT executor/converter/scorer imports.
- **All 14 unit tests pass** under `.venv/bin/python -m pytest tests/`.
- **`validate_attacks.py:_write_result_json`** writes to `results/{provider}/{method}/{mode}_{timestamp}.json` with `parents=True, exist_ok=True`; anchors to script directory (not `cwd`); strips `api_key_env` from payload; persists records on both success and failure (good for post-mortem).
- **`.env.example`** lists all 7 providers, no committed secrets.
- **`docs/methods_attacks.md`** is ~150 words, accurate citations (Chao 2023 / arXiv:2310.08419, Russinovich 2024 / arXiv:2404.01833), names PyRIT v0.13, honest about the placeholder scorer being separate from headline results.

---

## Below-threshold notes (informational only)

- `attacks.py:37` — imports `OpenAIChatTarget` but only `PromptChatTarget` is used in annotations. Dead import.
- `config/models.py:43` — comment says `see matrix_runner.py:285` but the file is 241 lines; correct ref is around line 180. Stale line number.
- `validate_attacks.py:152, 165` — `path.relative_to(Path.cwd())` raises `ValueError` if invoked from a directory not an ancestor of `RESULTS_DIR`. Practical risk low; users run from project root.
- `validate_attacks.py:83-87` — `data_type` collapses both `original_value_data_type` and `converted_value_data_type` into one field via `or`. Lossy by design (matters for image/text mixed outputs).
- `matrix_runner.py:171-174` — `except NotImplementedError as exc: logger.warning("Skipping ...")` is now defensive code that never fires (Anthropic works). Harmless.

---

## Top 3 fixes if you only do 3 before push

1. **`validate_attacks.py:174-178, 205`** — `logger.error` (or hard-fail) and persist `adversary` model identity into every result JSON when DeepSeek is missing. **This is the highest-impact silent failure: it produces seemingly-green results that are scientifically invalid.**
2. **`attacks.py:_build_pair_attack` and `_build_crescendo_attack`** — `logger.warning` at build time AND stamp `scorer="placeholder_self_ask"` into every CellResult / smoke JSON. Without this, the StrongREJECT hand-off can be silently missed and paper numbers will be from the placeholder.
3. **`CLAUDE.md`** — single-pass refresh of the four stale sections (registry size, Anthropic status, prompt categories, renamed test command) so the file reflects post-diff reality.

`matrix_runner.py:130-152` (lost error context) is the runner-up — Person A's lane to own; surface it to them.

---

## False alarms ruled out

- `validate_attacks.py:181-190` — `_get` provider cache is sound; `RuntimeError` on unknown provider is appropriate.
- `attacks.py:122-127` — `ATTACK_METHODS` dict order vs. `SMOKE_PLAN` order: looked up by key, not iterated, ordering doesn't matter.
- `validate_attacks.py:42-46` — `logging.basicConfig` at INFO is reasonable for a smoke test.
- `matrix_runner.py:233-237` — final memory-count printout swallows nothing.
