# NEXT_STEPS — Benchmark to Paper Execution Plan

This is the source-of-truth runbook for the next teammate.

## End goal
1. Produce trustworthy benchmark metrics (judged + calibrated)
2. Display those benchmarks in the frontend dashboard
3. Package final outputs for the research paper

---

## 0) Current repo status

- `backend/` contains the PyRIT harness, prompts, tests, and scripts
- `frontend/` contains a React + Vite + TypeScript dashboard scaffold
- Adversary defaults to Kimi (`ADVERSARY_PROVIDER=moonshot`)
- Backend tests + smoke validation pass
- Frontend lint + build pass
- Prompt converter exists: `backend/scripts/convert_prompts.py`
- `backend/prompts/prompts_long.json` is the completed real long-prompt dataset
- Short prompt dataset is still pending (not yet present as JSON)

---

## 1) Environment setup and baseline validation

## 1.1 Backend
```bash
cd backend
python -m venv .venv
source .venv/bin/activate
pip install -r ../requirements.txt
```

## 1.2 Env config
Edit `../.env` and ensure:
- `MOONSHOT_API_KEY` is set (adversary)
- defender keys set for models you plan to run
- `ADVERSARY_PROVIDER=moonshot`

## 1.3 Baseline checks (must pass)
```bash
cd backend
source ../.venv/bin/activate
pytest -q
python validate_attacks.py
```

---

## 2) Finalize short prompt dataset (long set already complete)

## Why
The long prompt set is already complete. Remaining prompt work is to create the short set with the same stable schema so both can be benchmarked/reportable.

## Steps
1. Keep `backend/prompts/prompts_long.json` unchanged as the completed long set.
2. Obtain C1 short-prompt CSV export in `docs/`.
3. Convert short set:
```bash
cd backend
source ../.venv/bin/activate
python scripts/convert_prompts.py \
  --input ../docs/<c1_short_export>.csv \
  --output prompts/prompts_short.json \
  --source "C1 short spreadsheet export YYYY-MM-DD" \
  --min-per-category 5
```
4. Validate short output quickly:
- 5 canonical categories present
- all retained rows have `abstraction_check == "Pass"`
- unique `prompt_id` across file
5. Ensure prompt IDs do not collide between long and short datasets (recommended prefixing convention per set).

## Deliverable
- `backend/prompts/prompts_long.json` (locked, complete)
- `backend/prompts/prompts_short.json` (new, validated)

---

## 3) Finish judging: StrongREJECT-derived integration

## Why
PAIR/Crescendo currently include placeholder logic. Reported metrics must use real judging.

## Required hook points
- `backend/attacks.py` (PAIR scorer HOOK)
- `backend/attacks.py` (Crescendo scorer HOOK)
- `backend/matrix_runner.py` (post-run scoring HOOK)

Find quickly:
```bash
cd backend
rg "HOOK: StrongREJECT scoring"
```

## Steps
1. Implement judge module (recommended path):
   - `backend/scoring/strongreject.py`
2. Replace placeholder scorer usage for PAIR + Crescendo.
3. Add post-run scoring pass in matrix runner:
   - load conversation by `conversation_id`
   - score final response
   - attach judge outputs to result rows
4. Add scorer metadata to rows (`scorer`, `judge_version`, etc.).

## Hard guardrail
No reported row may contain `placeholder_self_ask`.

---

## 4) Calibrate judge vs human labels

## Why
Paper claims require calibrated/validated scoring.

## Steps
1. Create hand-labeled subset file:
   - `backend/calibration/hand_labels.json`
2. Implement calibration script:
   - `backend/calibration/calibrate_judge.py`
3. Compute/save calibration metrics:
   - accuracy + precision/recall (or preferred)
   - confusion matrix
   - threshold policy if scores are continuous

## Deliverables
- calibration artifact file(s)
- fixed threshold/decision rule for final run

---

## 5) Execute benchmark runs (pilot -> full)

## 5.1 Pilot run (cheap safety check)
- 2 defender models
- all 4 attack methods
- >=3 prompts per category

## 5.2 Full run
- run full configured matrix after pilot is clean
- monitor retries/failures, especially Crescendo

## Required per-cell fields in outputs
- model, method, category, prompt_id
- outcome + error fields
- judged score/label
- latency

---

## 6) Export frontend-ready metrics artifact

## Why
Frontend and paper plotting should use one normalized source.

## Output path
- `backend/results/metrics_latest.json`

## Minimum schema contract
```json
{
  "generated_at": "...",
  "models": ["..."],
  "attack_methods": ["direct", "base64", "pair", "crescendo"],
  "categories": ["..."],
  "cells": [
    {
      "model": "...",
      "method": "...",
      "category": "...",
      "prompt_id": "...",
      "score": 0,
      "succeeded": false,
      "latency_seconds": 0.0
    }
  ]
}
```

---

## 7) Display benchmarks in frontend

Current frontend placeholders exist in:
- `frontend/src/components/*`
- `frontend/src/lib/normalizeResults.ts`
- `frontend/src/data/mockMetrics.ts`

## Steps
1. Replace mock data path with real loader for `metrics_latest.json`.
2. Update `normalizeResults.ts` for final backend schema.
3. Keep mock fallback for local dev.
4. Finalize required UI outcomes:
   - heatmap (Model x Attack)
   - filters (model/method/category)
   - summary cards (totals, ASR, avg score, failures)
   - drill-down table (prompt-level rows)
   - explicit `not run` vs `0 success` states

---

## 8) Verification gates before claiming results

## Backend gates
```bash
cd backend
source ../.venv/bin/activate
pytest -q
python validate_attacks.py
```

## Frontend gates
```bash
cd frontend
npm run lint
npm run build
```

## Data integrity gates
- No reported row with `placeholder_self_ask`
- Calibration outputs present and referenced
- `metrics_latest.json` schema stable and complete

---

## 9) Paper handoff package

Produce and freeze:
1. final judged benchmark dataset (JSON/CSV)
2. heatmap input table(s)
3. judge calibration summary
4. updated `docs/methods_attacks.md` with final run config + scoring statement

---

## 10) Task board (ordered)

1. C1 short-prompt conversion complete (long set already complete)
2. StrongREJECT judge integration complete
3. Judge calibration complete
4. Pilot benchmark clean
5. Full benchmark complete
6. Normalized metrics artifact exported
7. Frontend wired to real metrics
8. Benchmark dashboard finalized
9. Paper package assembled

---

## Definition of done

- [ ] Short real prompts converted/validated; long set retained as source of truth
- [ ] StrongREJECT-derived judging integrated at all HOOKs
- [ ] Calibration complete and documented
- [ ] Pilot + full benchmark runs complete
- [ ] `backend/results/metrics_latest.json` generated
- [ ] Frontend displays real benchmark metrics
- [ ] Backend + frontend verification gates pass
- [ ] Paper-facing artifacts prepared and ready for write-up
