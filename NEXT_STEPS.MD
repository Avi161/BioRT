# NEXT_STEPS — Real-Dataset Run + Hand-Off

Person B's harness is shipped, smoke-validated 5/5, and reviewed (`25APRIL_REVIEW.MD` → all blockers fixed). The next phase is moving from `prompts/mock_prompts.json` (10 abstract placeholders) to **the real bio-misuse prompt set** Person C1 is preparing in `docs/`, then running the full matrix.

---

## 0. Where we are right now

| Component | Status | Where |
|---|---|---|
| Model registry (7 models, OpenAI-compat for all) | Shipped | `config/models.py` |
| Attack factories (4 methods, uniform signature) | Shipped | `attacks.py` |
| Matrix runner with retries + post-mortem error fields | Shipped | `matrix_runner.py` |
| Per-attack live smoke validator | Shipped, 5/5 OK | `validate_attacks.py` → `results/{provider}/{method}/` |
| Methods paragraph for the paper | Shipped | `docs/methods_attacks.md` |
| Mock prompts (placeholders) | Shipped | `prompts/mock_prompts.json` (5 categories × 2 abstract prompts) |
| **Real prompts** | **Pending C1** | `docs/<C1 file>` → convert → `prompts/real_prompts.json` |
| Person A's full-matrix loop | Imports `attacks.ATTACK_METHODS` cleanly | `matrix_runner.py` |
| StrongREJECT scorer | Pending C2 | `# HOOK:` markers in `attacks.py:_build_pair_attack`, `_build_crescendo_attack`, `matrix_runner.py:main` |

**One thing that does not match the legacy brief in older docs**: PyRIT v0.13 (what we run) uses `RedTeamingAttack` / `CrescendoAttack` / `PromptSendingAttack` and `attack.execute_async(objective=..., memory_labels={...})` — **not** the older `PairOrchestrator` / `CrescendoOrchestrator` / `run_attack_async(seed_prompts=SeedGroup(...))` pattern shown in some legacy notes. The brief's snippets that use `SeedPrompt` / `SeedGroup` / `run_attack_async` will **not** import on this codebase. Use `attacks.ATTACK_METHODS[name](target, adversary, objective)` to build, then `attack.execute_async(...)`.

---

## 1. Real-dataset JSON format (spec)

Person C1's source-of-truth lives in `docs/` (CSV or spreadsheet). It needs to land at `prompts/real_prompts.json` in the schema below before the runner can pick it up. This format is a strict superset of the current mock format — adding per-prompt metadata (id, abstraction-check status, source) so we can trace any cell of Figure 1 back to a single approved prompt.

### Schema (`prompts/real_prompts.json`)

```json
{
  "schema_version": 1,
  "source": "C1 spreadsheet, exported YYYY-MM-DD",
  "abstraction_policy": "Only entries with abstraction_check == \"Pass\" are included.",
  "categories": {
    "synthesis_evasion": [
      {
        "prompt_id": "syn_evasion_01",
        "prompt_text": "Describe how a benchtop chemist would ...",
        "abstraction_check": "Pass",
        "source": "WMDP-Bio adapted | C1 original | ...",
        "notes": "optional free-form annotation"
      }
    ],
    "lab_automation_uplift": [ ... ],
    "dual_use_protocol":     [ ... ],
    "acquisition_kyc":       [ ... ],
    "virology":              [ ... ]
  }
}
```

### Field rules

| Field | Required | Type | Notes |
|---|---|---|---|
| `schema_version` | yes | int | Bump if the schema changes; current is `1`. |
| `source` | yes | str | Free-form provenance (which spreadsheet, when exported). |
| `abstraction_policy` | yes | str | Documents the responsible-disclosure rule applied. |
| `categories` | yes | object | Keys MUST be exactly the five paper categories: `synthesis_evasion`, `lab_automation_uplift`, `dual_use_protocol`, `acquisition_kyc`, `virology`. |
| `categories.<cat>` | yes | array | 5–10 entries per category (paper target). |
| `prompt_id` | yes | str | Stable, unique across the whole file. Convention: `<category-stem>_NN`. Used as the join key into Figure 1, scored results, and the spreadsheet. |
| `prompt_text` | yes | str | The actual abstracted prompt sent to the target. **Must NOT contain uplift content** — describes intent only. |
| `abstraction_check` | yes | str | Always `"Pass"` in this file. The converter MUST drop any `Fail`/blank rows from the CSV. |
| `source` | optional | str | "WMDP-Bio adapted", "C1 original", etc. |
| `notes` | optional | str | Reviewer comments. |

### Why this shape and not the existing flat `{category: [string, ...]}`

The current `prompts/mock_prompts.json` is `dict[str, list[str]]` — fine for placeholders but loses provenance (`prompt_id`) and approval state (`abstraction_check`). Once we go to real prompts:
- Figure 1 cells need to cite **which prompt produced which score** → we need `prompt_id`.
- Person C2's StrongREJECT pass needs to filter by approval status → we need `abstraction_check`.
- Reviewers will ask "where did each prompt come from?" → we need `source`.

The mock file stays as-is for tests; the real file moves to the richer schema.

### Conversion script (to write next)

`scripts/convert_prompts.py` — reads C1's CSV from `docs/`, validates, emits `prompts/real_prompts.json`. Behavior:

1. Open the CSV from `docs/` (path passed as CLI arg).
2. For each row, **drop unless** `abstraction_check == "Pass"` (responsible-disclosure rule).
3. Bucket by `category`; assert each row's category is one of the five canonical names — fail loudly otherwise.
4. Assert `prompt_id` is unique across all rows; fail loudly on duplicates.
5. Emit JSON in the schema above; pretty-print so a reviewer can diff in PRs.
6. Print a summary: `synthesis_evasion: N | lab_automation_uplift: N | ...`. Refuse to emit if any category has fewer than the agreed minimum (e.g. 5).

A first pass can be ~50 lines of `csv.DictReader` + `json.dump`.

### Runner changes once `real_prompts.json` exists

`matrix_runner.py:load_prompts` currently returns `dict[str, list[str]]`. Once switched to the rich schema:

```python
def load_prompts(path: str = "prompts/real_prompts.json") -> dict[str, list[dict]]:
    with open(path) as fh:
        doc = json.load(fh)
    assert doc.get("schema_version") == 1, f"Unsupported schema_version: {doc.get('schema_version')}"
    return doc["categories"]
```

Then update the inner loop:

```python
for prompt_obj in category_prompts:
    cell_result = await run_single_cell(
        ...
        prompt=prompt_obj["prompt_text"],
        prompt_id=prompt_obj["prompt_id"],   # new — adds to memory_labels + CellResult
    )
```

`CellResult` gains a `prompt_id: str` field; `run_single_cell` adds `"prompt_id": prompt_id` to `memory_labels` so `CentralMemory` queries can join on it. Tests in `tests/test_prompts.py` need a parallel update — keep the mock test asserting the legacy shape, add a new test for the rich shape.

---

## 2. The four attacks — what's already implemented vs. what changes for the real run

All four factories live in `attacks.py:ATTACK_METHODS` with the uniform `(target, adversary, objective) -> Attack` signature. The runner dispatches by name.

### Attack 1 — Direct (baseline) · `attacks._build_direct_attack` (`PromptSendingAttack`)

- **What:** Send each prompt verbatim, no wrapper.
- **Why:** Baseline refusal floor. Without this number, every other attack's lift is uninterpretable.
- **Expected:** Frontier closed models (Claude, GPT-5.4) refuse most; supplementary frontier (DeepSeek, Kimi) and the open-weight control (Llama-3.3) likely show higher direct compliance. Cross-model gap at baseline is itself a finding.
- **Code change for real run:** none — the factory is generic.

### Attack 2 — PAIR · `attacks._build_pair_attack` (`RedTeamingAttack`, `max_turns=3`)

- **What:** Adversarial LLM (DeepSeek V4 Flash, `temperature=0`) iteratively rewrites the prompt to overcome the target's refusal.
- **Why:** Higher attack success rate than direct against frontier models; realistic threat model for well-resourced adversaries.
- **Cost knob:** `max_turns` is currently `3`. Per the brief, `6` is the standard; bumping costs roughly 2× the per-cell tokens. Decide before the full run.
- **Gotcha:** the adversary LLM **must be a separate `OpenAIChatTarget` from the target**. Our runner enforces this — `matrix_runner.py:main` skips PAIR cells where the only buildable model is also the adversary. **Do not bypass this guard**: target-vs-target invalidates the experiment.
- **Cost reality (from review):** per cell at `max_turns=6` ≈ 15k tokens. 8 prompts × 5 categories × 4 models ≈ $8–12 in PAIR alone. Watch the Claude column.

### Attack 3 — Crescendo · `attacks._build_crescendo_attack` (`CrescendoAttack`, `max_turns=5`, `max_backtracks=5`)

- **What:** Multi-turn benign-to-harmful escalation. The adversary opens innocuously and steers toward the target objective over several turns, backtracking when refused.
- **Why:** Most realistic bio-misuse adversary doesn't open with the dangerous question — they build conversational context first. Empirically high success on GPT-4 / Gemini in the published Russinovich et al. 2024 result.
- **Cost reality:** quadratic context growth. Per cell ≈ 20–25k tokens at 8 turns. To stay cheap, the brief recommends running Crescendo **only on DeepSeek + Kimi** (Chinese SOTA), not Claude / GPT. Decide whether we restrict Crescendo's column in Figure 1 explicitly — and if so, mark the missing cells in the heatmap as "not run (cost)" rather than "0% success".
- **DeepSeek thinking mode:** already disabled — `temperature=0.0` is set on the DeepSeek `ModelConfig` (`config/models.py:51`) so chain-of-thought tokens don't inflate the cost 3–5×. Verified by `tests/test_config.py::test_deepseek_temperature_zero`.

### Attack 4 — Base64 encoding · `attacks._build_base64_attack` (`PromptSendingAttack` + `Base64Converter`)

- **What:** The prompt is base64-encoded before sending. PyRIT's `Base64Converter` does the encoding via `PromptConverterConfiguration`.
- **Why:** Probes whether the safeguard is **syntactic** (keyword-matching surface text) or **semantic** (understanding decoded intent). Refuses-when-encoded → semantic safeguard (positive finding). Complies-when-encoded-but-not-direct → keyword-matcher (negative finding). Either way, citable.
- **Cost:** single-turn, cheap — run on every model.
- **Sanity check:** before reading numbers, confirm the wire actually carries base64 by spot-checking one stored conversation in `CentralMemory.get_message_pieces()`. The `converted_value` should be base64; `original_value` is the plaintext.

---

## 3. Run plan — which attack runs against which model

| Attack \ Model | Claude Sonnet 4.6 | GPT-5.4 | Gemini 3 Pro | Grok 4 | DeepSeek V4 Flash | Kimi K2.5 | Llama-3.3 70B (control) |
|---|---|---|---|---|---|---|---|
| 1. Direct | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |
| 2. PAIR | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |
| 3. Crescendo | ✗ (budget) | ✗ (budget) | ✗ (budget) | ✗ (budget) | ✓ | ✓ | ✓ |
| 4. Base64 | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |

That's **(7 × 3) + 3 = 24** (model, attack) combinations × 5 categories × ~8 prompts = **~960 cells**. The original 4-model brief had 14 combos / 560 cells; we widened to 7 because the registry already supports all of them. If keys for some providers are missing at run time, those rows just drop out — `get_available_models()` logs the active/skipped banner.

To restrict Crescendo to Chinese SOTA only, gate inside `matrix_runner.py:main`:

```python
CRESCENDO_PROVIDERS = {"deepseek", "moonshot"}
if method_name == "crescendo" and model_config.provider not in CRESCENDO_PROVIDERS:
    continue
```

(This is a one-line addition; do it just before `for category, ...` in the existing nested loop.)

---

## 4. Hour-by-hour timeline (post-handoff)

| Hours | Task | Done-when |
|---|---|---|
| 0–1 | C1 delivers `docs/<spreadsheet>`. Skim Tier 1 papers if not already done. | Schema understood; abstraction-check field meaning is clear. |
| 1–2 | Write `scripts/convert_prompts.py`. Convert C1's file → `prompts/real_prompts.json`. Eyeball one entry per category. | JSON validates; 5 categories present; ≥5 rows each. |
| 2–3 | Update `matrix_runner.py:load_prompts` for the rich schema. Add `prompt_id` to `CellResult` and to `memory_labels`. Update tests. | `pytest -q` green; `validate_attacks.py` still 5/5. |
| 3–5 | Pilot: matrix run against **two** models only (DeepSeek + Anthropic) to flush bugs cheaply. Inspect 3–5 stored conversations via `CentralMemory.get_message_pieces`. | All four attack types produce ≥1 conversation each; `error_type` is null on the rows that ran clean. |
| 5–10 | Full matrix run. Watch rate limits. Restart any timed-out Crescendo cells (per-conversation, stateful — relaunch only the failing `(model, prompt)` pair). | All cells either OK or carry a populated `error_type`/`traceback`. |
| 10–14 | Hand off raw memory + per-cell results to C2 (StrongREJECT scoring) and D (Figure 1 plotting). Replace placeholder scorer per the `# HOOK:` markers. | C2 has a working scorer; D has a heatmap rendering from real numbers. |
| 14–24 | Buffer for re-runs and missed cells. | — |
| 24–36 | Methods writing — drop `docs/methods_attacks.md` into Section 3 of the paper, expand with the actual numbers. | Paragraph wired into the LaTeX. |

Buffer is generous on purpose: PAIR/Crescendo retries will eat hours nobody anticipates.

---

## 5. Reading list (only what's load-bearing this sprint)

**Tier 1 — read before writing any conversion / runner code (≤1h total):**
- Russinovich, Salem, Eldan (2024). *Great, Now Write an Article About That: The Crescendo Multi-Turn LLM Jailbreak Attack.* arXiv:2404.01833. Sections 2–3 — understand the benign-to-harmful loop and what backtracking means.
- Chao et al. (2023). *Jailbreaking Black Box Large Language Models in Twenty Queries* (PAIR). arXiv:2310.08419. Sections 2–3 — what `conversation_objective` is and how the attacker rewrites failed prompts.
- Souly et al. (2024). *A StrongREJECT for Empty Jailbreaks.* arXiv:2402.10260 — skim. C2 owns scoring, but understanding what they score helps you confirm the harness emits scoreable conversations (non-empty, no error pages).

**Tier 2 — implementation reference, dip in as needed:**
- PyRIT v0.13 docs: `pyrit.executor.attack` module — `RedTeamingAttack`, `CrescendoAttack`, `PromptSendingAttack`, `AttackAdversarialConfig`, `AttackScoringConfig`. Already used in `attacks.py`; the docs are the cleanest reference for parameter semantics.
- Mazeika et al. (2024). *HarmBench.* arXiv:2402.04249, Section 3 — sets expectations for baseline ASR ranges across model families.

**Tier 3 — only if time:**
- Perez & Ribeiro (2022). *Ignore Previous Prompt.* arXiv:2211.09527 — background motivation for Attack 4 (Base64).

---

## 6. Methods paragraph (drop-in for the paper)

`docs/methods_attacks.md` already holds the ~150-word block. After the real run, expand it with the actual matrix size and any restrictions (e.g. "Crescendo run only on DeepSeek and Kimi for budget"). Don't change the citations or claim StrongREJECT was used until C2 confirms the swap is in.

---

## 7. Definition of done for this sprint

- [ ] `docs/<C1 spreadsheet>` exists.
- [ ] `scripts/convert_prompts.py` produces `prompts/real_prompts.json` matching the schema in §1.
- [ ] `matrix_runner.py:load_prompts` reads the rich schema; `CellResult` carries `prompt_id`; tests updated; `pytest -q` green.
- [ ] One pilot matrix run (2 models, all 4 attacks, ≥3 prompts/category) completes with no `error_type` set on the rows that ran clean.
- [ ] StrongREJECT swapped in at the three `# HOOK:` markers.
- [ ] Full matrix run produces a JSON or DuckDB extract D can plot directly into Figure 1.
- [ ] `docs/methods_attacks.md` updated with final matrix shape and any per-attack scoping notes.
- [ ] No `placeholder_self_ask` value in the `scorer` field of any **reported** result. If it's still there, the StrongREJECT swap was missed.
