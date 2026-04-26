# AIxBio — Bio-Misuse Safeguard Measurement Harness

Orchestration layer for testing bio-misuse safeguards across frontier AI models. Built on [PyRIT](https://github.com/Azure/PyRIT) (Microsoft Python Risk Identification Tool).

## Repository Layout

- `backend/` — Python harness (models, attacks, prompts, runners, tests)
- `frontend/` — metrics dashboard UI (React + Vite + TypeScript)
- root docs/config — project docs, env templates, license

## Test Matrix

| Dimension | Values |
|-----------|--------|
| **Models** | GPT-5.4, Claude Sonnet 4.6, Gemini 3 Pro, DeepSeek V4 Flash, Kimi 2.5 |
| **Attack Methods** | Direct Request, PAIR, Crescendo, Base64 Encoding |
| **Categories** | synthesis_evasion, lab_automation_uplift, dual_use_protocol, acquisition_kyc, virology |

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

Edit `../.env` and add your API keys. The runner skips models whose keys are missing, so at least one key is enough to run.

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
backend/prompts/              Prompt datasets
backend/matrix_runner.py      Full experiment runner
backend/validate_attacks.py   Per-attack smoke runner
backend/tests/                Backend test suite
frontend/src/                 Metrics dashboard components
```

## Scoring (Phase 2)

StrongREJECT scoring integration points are marked with `# HOOK: StrongREJECT scoring` comments in `backend/attacks.py` and `backend/matrix_runner.py`.
