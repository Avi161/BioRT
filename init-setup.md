# Project Context: Bio-Misuse Safeguard Measurement Hackathon
We are building an offline measurement instrument to test the robustness of bio-misuse safeguards on frontier AI models. The output will be a 4-page research report featuring a heatmap of attack success rates across 5 specific models and 4 attack methods. 

I am responsible for the "Harness" — the orchestration layer. My job is to wire up Microsoft's PyRIT (Python Risk Identification Tool), configure the model APIs, and build the experiment runner that iterates through our test matrix and logs results locally to DuckDB.

Please help me build this out. We are on a tight 48-hour hackathon deadline, so prioritize clean, functional, and modular scripting over over-engineering. 

## Architectural Constraints (CRITICAL)
- **Database:** DuckDB (via PyRIT's built-in memory/storage layer). Do not set up a standalone database server. 
- **API Normalization:** Use `litellm` where necessary to standardize API calls to the OpenAI format, especially for DeepSeek and Kimi.

## Technical Requirements
- **Framework:** PyRIT (Microsoft)
- **Models:** GPT-5.4, Claude Sonnet 4.6, Gemini 3 Pro, DeepSeek V4 Flash, Kimi 2.5
- **Language:** Python 3.10+

## Step-by-Step Build Instructions

Please generate the following files and structure:

### 1. Project Setup (`requirements.txt` & `.env.example`)
Create a `requirements.txt` that includes `pyrit`, `python-dotenv`, `duckdb`, `litellm`, `pandas`, and `seaborn` (for later analysis). 
Create an `.env.example` file with placeholders for the following API keys:
- OPENAI_API_KEY (for GPT-5.4)
- ANTHROPIC_API_KEY (for Sonnet 4.6)
- GEMINI_API_KEY (for Gemini 3 Pro)
- DEEPSEEK_API_KEY (for DeepSeek V4 Flash)
- MOONSHOT_API_KEY (for Kimi 2.5)

### 2. The "Hello World" Script (`hello_world.py`)
Write a script that achieves our Hour-18 milestone. It should:
- Load environment variables.
- Initialize PyRIT's `DuckDBMemory` to a local file (e.g., `results/pyrit_memory.db`).
- Set up a single target model (e.g., `gpt-5.4` using PyRIT's target wrappers or LiteLLM).
- Run a simple "direct request" attack (no-attack baseline) using 5 mock bio-misuse prompts.
- Ensure the interaction is successfully logged to the DuckDB memory layer.
- Include a simple query at the end of the script to print the database contents to the console to verify local storage worked.

### 3. The Matrix Experiment Runner (`matrix_runner.py`)
This is the core orchestration script. Write a script that:
- Initializes the DuckDB memory layer.
- Defines a configuration matrix containing:
  - **Models:** GPT-5.4, Claude Sonnet 4.6, Gemini 3 Pro, DeepSeek V4 Flash, Kimi 2.5. 
  - **Attack Methods:** Direct Request (baseline), PAIR, Crescendo, and Base64 Encoding. *Use PyRIT's built-in orchestrators/converters for these.*
  - **Categories:** Synthesis, Evasion, Lab-automation, Virology.
- Loads a mock JSON file of prompts (`mock_prompts.json`) mapped to the categories above.
- Contains nested loops to iterate over Model × Method × Prompt.
- Uses Python `logging` to output progress (e.g., "Running Crescendo attack on Sonnet 4.6 for prompt ID 4...").
- Handles API rate limits and exceptions gracefully (e.g., retry logic or skipping on hard failure) so the full matrix run doesn't crash halfway through.

### 4. Mock Data (`mock_prompts.json`)
Generate a small JSON file with 2 abstract, harmless placeholder prompts per category (Synthesis, Evasion, Lab-automation, Virology) so we can test the `matrix_runner.py` immediately. 

**Execution Note:** Please write the code assuming the latest version of PyRIT. Add extensive comments explaining how to execute the run and where the StrongREJECT scoring logic will be injected later.