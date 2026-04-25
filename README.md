# AIxBio — Bio-Misuse Safeguard Measurement Harness

Orchestration layer for testing bio-misuse safeguards across frontier AI models. Built on [PyRIT](https://github.com/Azure/PyRIT) (Microsoft Python Risk Identification Tool).

## Test Matrix

| Dimension | Values |
|-----------|--------|
| **Models** | GPT-5.4, Claude Sonnet 4.6, Gemini 3 Pro, DeepSeek V4 Flash, Kimi 2.5 |
| **Attack Methods** | Direct Request, PAIR, Crescendo, Base64 Encoding |
| **Categories** | Synthesis, Evasion, Lab-automation, Virology |

## Quick Start

### 1. Install

```bash
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

### 2. Configure API Keys

```bash
cp .env.example .env
```

Edit `.env` and add your API keys. The runner skips any model whose key is missing, so you only need at least one key to get started.

### 3. Run Hello World (Hour-18 Milestone)

Sends 5 prompts to a single model and verifies memory logging:

```bash
python hello_world.py
```

### 4. Run the Full Matrix

Iterates over all available models, attack methods, and prompt categories:

```bash
python matrix_runner.py
```

## Project Structure

```
config/models.py         Model registry & PyRIT target factory
prompts/mock_prompts.json   Placeholder prompts (2 per category)
hello_world.py           Single-model smoke test
matrix_runner.py         Full experiment runner
results/                 Runtime artifacts (gitignored)
```

## Scoring (Phase 2)

StrongREJECT scoring is not yet wired in. Integration points are marked with `# HOOK: StrongREJECT scoring` comments in `matrix_runner.py`. The scorer will evaluate final assistant responses and feed into the heatmap analysis.

## Notes

- **Memory:** PyRIT uses SQLite-backed in-memory storage. All conversations are queryable via `CentralMemory` during the session.
- **Anthropic:** Claude requires a LiteLLM proxy wrapper (not yet implemented). The runner auto-skips it until configured.
- **Rate limits:** The matrix runner retries failed calls with exponential backoff (3 attempts, 5s base).
