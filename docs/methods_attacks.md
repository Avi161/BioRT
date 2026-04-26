# Methods — Attack Library

The harness implements four jailbreak methods drawn directly from published red-teaming literature, all built on Microsoft's PyRIT framework (v0.13). The methods are configured in `backend/attacks.py` and dispatched through `ATTACK_METHODS`, which the runner iterates over for every (model, prompt) pair.

**Direct request (baseline)** sends each prompt to the target verbatim with no transformation, implemented via PyRIT's `PromptSendingAttack`. This establishes the no-attack refusal floor against which every adversarial method is compared.

**Base64 encoding** wraps the same `PromptSendingAttack` with PyRIT's `Base64Converter`, so the target receives the prompt as a base64-encoded string. This probes whether safeguards generalise across input encodings — a long-known and still-effective family of obfuscation jailbreaks.

**PAIR** (Prompt Automatic Iterative Refinement; Chao et al. 2023) drives the target with an adversarial LLM that iteratively rewrites its prompt to elicit a compliant response. We use PyRIT's `RedTeamingAttack` orchestrator with the standard `RTASystemPromptPaths.TEXT_GENERATION` system prompt and `max_turns=3`.

**Crescendo** (Russinovich et al. 2024) is a multi-turn benign-to-harmful escalation in which the adversarial LLM begins with innocuous queries and gradually steers toward the target objective, backtracking when refused. Implemented via PyRIT's `CrescendoAttack` with `max_turns=5` and `max_backtracks=5` (cheaper; typical reported averages in the Crescendo paper are around 5).

Both PAIR and Crescendo require an adversarial-LLM target; we currently default to Kimi (`ADVERSARY_PROVIDER=moonshot`) for cost control. The inner-loop "did the response satisfy the objective" decision is currently a `SelfAskTrueFalseScorer` placeholder; it does **not** affect headline results, which are produced by the calibrated bio-aware StrongREJECT-derived judge applied in a separate scoring pass over the stored conversations.

## References

- Chao, P., Robey, A., Dobriban, E., Hassani, H., Pappas, G. J., & Wong, E. (2023). *Jailbreaking Black Box Large Language Models in Twenty Queries.* arXiv:2310.08419.
- Russinovich, M., Salem, A., & Eldan, R. (2024). *Great, Now Write an Article About That: The Crescendo Multi-Turn LLM Jailbreak Attack.* arXiv:2404.01833.
- Microsoft. *Python Risk Identification Tool (PyRIT)*, v0.13. https://github.com/Azure/PyRIT.
