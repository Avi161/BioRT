## Technical Brief: HarmBench Evaluation Framework

### 1. Overview
[cite_start]HarmBench is a standardized evaluation framework designed to rigorously assess automated red teaming methods and LLM safety[cite: 3, 7]. [cite_start]It addresses the lack of standardization in the field by establishing three key properties for evaluations: breadth, comparability, and robust metrics[cite: 8, 29].

### 2. Methodology (Section 3)
[cite_start]The framework defines the red teaming task as designing test cases to elicit a specific harmful behavior $y$ from a target model[cite: 121].

* [cite_start]**Primary Metric:** Success is measured by the **Attack Success Rate (ASR)**, which is the percentage of test cases that elicit the target behavior[cite: 122, 128].
* [cite_start]**Breadth:** The framework includes 510 unique behaviors across four functional categories (Standard, Contextual, Copyright, and Multimodal) and seven semantic categories (e.g., Cybercrime, Chemical/Biological weapons)[cite: 68, 211, 220, 222].
* [cite_start]**Comparability:** HarmBench standardizes the generation length to **512 tokens**[cite: 168]. [cite_start]Research showed that varying this parameter can change ASR by up to **30%**, rendering unstandardized comparisons misleading[cite: 164, 166, 167].
* [cite_start]**Robust Metrics:** To ensure accuracy, HarmBench utilizes a fine-tuned **Llama 2 13B classifier**[cite: 304, 832]. [cite_start]This classifier is trained to identify "successful" attacks even if a model initially refuses before complying or if the generated code contains errors[cite: 174, 298, 299].

### 3. Attack Method Categories
[cite_start]HarmBench evaluates 18 red teaming methods across various access levels[cite: 9, 338]:
* [cite_start]**White-Box:** Token-level optimization attacks like **GCG**, **PEZ**, and **GBDA**[cite: 340, 871, 881, 883].
* [cite_start]**Black-Box/Prompt-Based:** Adaptive prompting methods such as **PAIR** and **TAP**[cite: 340, 894, 895].
* [cite_start]**Transfer:** Attacks optimized against proxy models and then tested on target models, such as **GCG-Transfer**[cite: 340, 877].

### 4. Baseline Expectations & Findings
* [cite_start]**Model Size vs. Robustness:** A key finding is that **robustness is independent of model size**[cite: 32, 389, 407]. [cite_start]Within model families, a 7B model often shows similar robustness to a 70B model, suggesting training data and algorithms are the primary factors in safety[cite: 382, 408, 409].
* [cite_start]**Attack Effectiveness:** * No current attack or defense is uniformly effective; all attacks fail on some models, and all models are vulnerable to some attacks[cite: 32, 389, 401, 402].
    * [cite_start]**Contextual behaviors** (those requiring specific background information) generally yield **higher ASRs** than standard behaviors[cite: 391, 1095].
    * [cite_start]**Copyright behaviors** typically show lower ASR because the evaluation uses a strict hashing-based standard requiring verbatim generation[cite: 392, 866, 1048].
* [cite_start]**Adversarial Training:** The paper introduces **R2D2 (Robust Refusal Dynamic Defense)**, an adversarial training method that significantly reduces GCG attack success rates compared to standard models[cite: 10, 314, 451].

### 5. Evaluation Pipeline
[cite_start]The HarmBench pipeline proceeds in three standardized steps[cite: 292]:
1.  [cite_start]**Generating Test Cases:** The red teaming method produces test cases for specific harmful behaviors[cite: 293].
2.  [cite_start]**Generating Completions:** Test cases are passed to the target LLM using greedy decoding to produce output strings (standardized to 512 tokens)[cite: 123, 168, 294].
3.  [cite_start]**Evaluating Completions:** Completions are processed by the standardized classifier to determine the final ASR[cite: 296, 303].