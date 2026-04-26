## Technical Brief: RIOT (Red-teaming with In-context OpTimization)

### 1. Overview
[cite_start]**RIOT** is a framework that treats red-teaming as an **in-context optimization** problem[cite: 3, 5]. [cite_start]Unlike white-box attacks that require gradient access (e.g., GCG) or black-box attacks that require specialized training, RIOT leverages the **in-context learning** capabilities of LLMs to iteratively refine jailbreak prompts[cite: 1, 5]. [cite_start]The core goal is to generate adversarial prompts that bypass safety filters to elicit specific harmful behaviors[cite: 1, 4].

### 2. Methodology
RIOT utilizes an iterative feedback loop involving three distinct roles:

* [cite_start]**Attacker LLM:** Responsible for generating and refining the jailbreak prompt[cite: 5, 8]. [cite_start]It takes the target behavior and the history of previous attempts as input to produce a new attack[cite: 6, 8].
* [cite_start]**Target LLM:** The model being red-teamed[cite: 1, 8]. [cite_start]It receives the Attacker's prompt and generates a completion[cite: 8].
* [cite_start]**Judge LLM:** Acts as a scorer (Person C2) to evaluate the Target model's response[cite: 5, 8]. [cite_start]It provides a numerical score (typically 1–10) and an explanation of why the attack succeeded or failed[cite: 8].

### 3. Optimization Loop
The optimization process follows these steps:
1.  [cite_start]**Initial Prompting:** The Attacker generates an initial jailbreak attempt for a specific harmful behavior[cite: 8].
2.  [cite_start]**Target Response:** The Target LLM generates a completion based on the adversarial prompt[cite: 8].
3.  [cite_start]**Judgment:** The Judge LLM evaluates the completion based on predefined criteria, assigning a score and providing feedback[cite: 8].
4.  [cite_start]**In-Context Refinement:** The Attacker receives the full history (all previous prompts, Target responses, and Judge feedback)[cite: 8, 9]. [cite_start]It uses this context to optimize the next prompt, learning from what "almost worked" or what triggered a refusal[cite: 5, 9].

### 4. Scoring & Validation (The "Judge" Role)
Understanding the Judge's scoring logic is critical for the Attacker to validate that it is generating "scoreable" responses:
* [cite_start]**Success Definition:** A score of 10 usually indicates a full jailbreak where the model provides a detailed, helpful response to a harmful query[cite: 14].
* [cite_start]**Refusal Detection:** Low scores are given to "I'm sorry" style refusals or empty responses[cite: 14].
* [cite_start]**Constraint Satisfaction:** The Attacker uses the Judge’s explanation to pivot strategies—for example, if the Judge notes that the model refused because of "directness," the Attacker might attempt a "roleplay" strategy in the next iteration[cite: 8, 14].

### 5. Key Findings & Performance
* [cite_start]**Efficiency:** RIOT is highly query-efficient, often achieving success in significantly fewer iterations than other black-box methods[cite: 1, 10].
* [cite_start]**Transferability:** Attacks optimized via RIOT on one model (e.g., GPT-3.5) often show high transferability to other models (e.g., GPT-4 or Llama-2)[cite: 10, 11].
* [cite_start]**State-of-the-Art Results:** RIOT achieves a near 100% Attack Success Rate (ASR) on several popular aligned models, demonstrating that in-context optimization is a powerful tool for discovering vulnerabilities[cite: 1, 11].

### 6. Comparison with HarmBench
[cite_start]While HarmBench provides the *standardized evaluation framework* (behaviors, generation length, and static classifiers), RIOT is the *attack algorithm* that can be used within such a framework to find vulnerabilities dynamically through iterative feedback[cite: 1, 3].