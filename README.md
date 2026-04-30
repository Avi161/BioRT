# AIxBio — Bio-Misuse Safeguard Measurement Harness

Orchestration layer for testing bio-misuse safeguards across frontier AI models. Built on [PyRIT](https://github.com/Azure/PyRIT) (Microsoft Python Risk Identification Tool).

## Repository Layout

- `backend/` — Python harness (models, attacks, prompts, runners, judge, tests)
- `frontend/` — metrics dashboard UI (React + Vite + TypeScript)
- root docs/config — project docs, env templates, license

## Test Matrix

| Dimension          | Values                                                                                                       |
| ------------------ | ------------------------------------------------------------------------------------------------------------ |
| **Models**         | DeepSeek V4 Flash, Claude Sonnet 4.6, GPT-5.4, Gemini 3 Pro, Grok 4, Kimi K2.5, Llama-3.3 70B (control)      |
| **Attack Methods** | Direct Request, Base64 Encoding, PAIR, Crescendo                                                             |
| **Categories**     | synthesis_evasion, lab_automation_uplift, dual_use_protocol, acquisition_kyc, virology                       |

## Quick Start (Backend)

### 1. Install

```bash
cd backend
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r ../requirements.txt
```

### 2. Configure API Keys

```bash
cp ../.env.example ../.env
```

Edit `../.env` and add your API keys. Runners skip models whose keys are missing, so at least one key is enough to run.

### 3. Run Hello World

```bash
cd backend
python hello_world.py
```

### 4. Run Attack Smoke Validation

```bash
cd backend
python validate_attacks.py
```

### 5. Run the Full Matrix

```bash
cd backend
python matrix_runner.py
```

Re-running the same command resumes from where it left off — cells already on disk under `--output-root` (default `../results`) are skipped.

### 6. Score Results

```bash
cd backend
python score_results.py            # scores everything under ../results -> ../eval_results
python print_eval_summary.py       # prints per-model / per-method counts by status
```

`score_results.py` runs the bio-aware judge in `backend/judge.py` against each cell from a matrix run. It does **not** use PyRIT — it calls the judge directly via `litellm`.

## Crescendo Debug Mode

Multi-turn Crescendo self-play with full transcript logging and resume support runs from the **same** `backend/matrix_runner.py` via the `--crescendo-debug` subcommand:

```bash
# From repo root:
python backend/matrix_runner.py --crescendo-debug --crescendo-debug-full \
  --crescendo-kimi-attacks-anthropic \
  --prompt-file prompts/prompts_long.json

# Or from backend/:
cd backend
python matrix_runner.py --crescendo-debug --crescendo-debug-full \
  --crescendo-kimi-attacks-anthropic \
  --prompt-file ../prompts/prompts_long.json
```

This is a **separate** code path from `--method crescendo` (which runs one cell of the *standard matrix*). The debug mode does full attacker-vs-defender self-play with transcript logging and writes to `results/crescendo/`. See `CLAUDE.md` for the full flag list and the resume-from-JSONL pattern.

## Frontend (Metrics UI)

```bash
cd frontend
npm install
npm run dev
```

## Project Structure

```text
backend/config/models.py      Model registry & target factory
backend/attacks.py            Attack method factories
backend/prompts/              Prompt datasets (gitignored — feature branches only)
backend/matrix_runner.py      Matrix runner + Crescendo debug entry (one CLI, two modes)
backend/validate_attacks.py   Per-attack smoke runner
backend/judge.py              Bio-aware LLM judge
backend/score_results.py      Eval orchestrator over matrix-runner JSONL
backend/print_eval_summary.py Eval-results status counter
backend/crescendo_debug.py    Crescendo-debug helpers (imported by matrix_runner.py)
backend/tests/                Backend test suite
prompts/mock_prompts.json     Public placeholder dataset
frontend/src/                 Metrics dashboard components
```

## Prompts

`prompts/mock_prompts.json` (root) is a public placeholder dataset. The real bio-misuse prompts (`backend/prompts/`) are **gitignored on `main`** and live on feature branches only — see the `.gitignore` rule and `CLAUDE.md` for the schema.

## Scoring

Results are scored with a bio-aware judge (`backend/judge.py`) calibrated for the five categories above. Run the matrix first, then `python score_results.py` to produce per-cell judge outputs. The legacy `# HOOK: StrongREJECT scoring` markers in `backend/attacks.py` mark where the placeholder `SelfAskTrueFalseScorer` is wired into PAIR / Crescendo at attack-time; the post-hoc judge in `score_results.py` is the canonical scoring path.
