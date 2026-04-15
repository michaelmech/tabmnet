Create an `AGENTS.md` file for this repository.

Context:
This repo is for implementing a TabM-style parameter-efficient ensemble inside TabNet. The goal is to adapt the shared-parameter ensembling ideas from TabM to the DreamQuark TabNet codebase, starting from simple, testable integration points and expanding only when the basics are working. TabM is an MLP-based model that represents an ensemble of implicit submodels trained jointly, with shared weights and lightweight per-member adapters; strong variants include a simple BE head, deeper BatchEnsemble integration, and careful initialization choices.  The local `tabm.py` file already provides reusable components such as `EnsembleView` and `LinearBatchEnsemble`, plus initialization utilities that mirror the TabM design.  The local TabNet codebase includes the standard TabNet encoder / decoder / feature transformer / attentive transformer structure in `tab_network.py`. 

What I want in `AGENTS.md`:
Write a practical contributor guide for coding agents working in this repo. Make it opinionated, concrete, and optimized for autonomous implementation work. The audience is Codex / code agents, not humans new to ML.

The file should include these sections:

1. **Mission**

   * State that the repository’s purpose is to implement and evaluate TabM-inspired parameter-efficient ensembling for TabNet.
   * Explain the target direction: preserve TabNet’s external training API where possible while adding an ensemble dimension internally.
   * Make clear that correctness, minimal diffs, and incremental progress matter more than ambitious rewrites.

2. **Primary design constraints**

   * Reuse `tabm.py` primitives whenever possible instead of re-implementing BatchEnsemble logic. In particular, prefer `EnsembleView` and `LinearBatchEnsemble`. 
   * Preserve existing TabNet public interfaces unless there is a compelling reason not to.
   * Keep shape semantics explicit at every stage, especially when introducing `(B, K, D)` tensors.
   * Prefer modular changes over monolithic rewrites.
   * Keep baseline TabNet behavior available.
   * Avoid silently changing preprocessing, loss definitions, masking semantics, or explainability outputs.

3. **Recommended implementation order**

   * Start with the easiest baseline: replace only the final prediction head with a BatchEnsemble head and average predictions over ensemble members to preserve the existing API. This is the intended first milestone.
   * Next, BatchEnsemble-ify the feature transformer / GLU blocks.
   * Then explore BatchEnsemble in the attentive transformer / mask generator.
   * Only after those are working, consider full end-to-end per-head masking or pretraining support.
   * Mention that prior local drafts already reflect this staged approach, including head-only and progressively deeper integrations.

4. **Current repository hints**

   * Mention that `tab_network.py` contains the original TabNet modules to extend or subclass. 
   * Mention that `pretraining.py` and `pretraining_utils.py` exist and may need updates later, but supervised regression/classification should come first.
   * Mention that draft implementation files in the repo appear to capture multiple experimental directions and should be mined for ideas before writing new code.
   * Mention `grokfast.py` only as optional experimentation, not core to the first implementation pass. 

5. **Architecture guidance**

   * Explain the two allowed integration styles:

     * “API-preserving aggregation”: introduce `(B, K, D)` internally, then average back to `(B, D)` at controlled boundaries.
     * “Full per-head propagation”: keep `(B, K, D)` through more of the network, but only when each downstream module is explicitly adapted.
   * Stress that agents must document tensor shapes in code comments for every modified forward path.
   * Instruct agents to be disciplined about where averaging happens.
   * Note that shared batch norm / ghost batch norm is acceptable as a first pass, even if per-head normalization is explored later. This matches the local implementation drafts.

6. **Implementation rules**

   * Do not break import paths unnecessarily.
   * Prefer subclassing or additive modules over invasive edits when possible.
   * Add small helper classes only when they reduce shape confusion.
   * Keep new constructor arguments explicit, such as `k` and `scaling_init`.
   * Default choices should be conservative and easy to disable.
   * Regression support can be first; multi-task or full parity can come later if unfinished.
   * Do not claim TabM-equivalence unless the implementation actually matches the relevant adapter placement and initialization behavior from the paper / local module.

7. **Testing expectations**

   * Require lightweight smoke tests for tensor shapes and forward passes.
   * Require at least one test that confirms a head-only BE model returns the same outer API shape as standard TabNet.
   * Require checks for:

     * `(B, D) -> (B, K, D)` expansion
     * output averaging behavior
     * no shape regressions in masks / explainability calls
     * train/eval forward pass sanity
   * Encourage tiny synthetic-data training runs rather than expensive benchmarks in the first pass.

8. **Experiment discipline**

   * Tell agents to separate “working baseline,” “experimental branch,” and “research idea.”
   * Tell agents to leave clear TODO markers for incomplete advanced work.
   * Tell agents not to overfit the repo around one unfinished prototype.
   * Encourage preserving failed-but-informative design notes in comments only when brief and useful.

9. **Definition of done for milestones**

   * Milestone 1: BE final head integrated, training runs, outputs averaged, baseline API preserved.
   * Milestone 2: feature transformer BE path works with clear shape handling.
   * Milestone 3: attentive transformer BE variant works and is test-covered.
   * Milestone 4: optional pretraining support.

10. **Agent workflow**

    * Before editing, inspect existing local drafts for similar code.
    * Prefer smallest viable patch.
    * After editing, run targeted tests or minimal scripts.
    * Summarize exactly what changed, what remains incomplete, and what assumptions were made.



# CLAUDE.md

Behavioral guidelines to reduce common LLM coding mistakes. Merge with project-specific instructions as needed.

**Tradeoff:** These guidelines bias toward caution over speed. For trivial tasks, use judgment.

## 1. Think Before Coding

**Don't assume. Don't hide confusion. Surface tradeoffs.**

Before implementing:
- State your assumptions explicitly. If uncertain, ask.
- If multiple interpretations exist, present them - don't pick silently.
- If a simpler approach exists, say so. Push back when warranted.
- If something is unclear, stop. Name what's confusing. Ask.

## 2. Simplicity First

**Minimum code that solves the problem. Nothing speculative.**

- No features beyond what was asked.
- No abstractions for single-use code.
- No "flexibility" or "configurability" that wasn't requested.
- No error handling for impossible scenarios.
- If you write 200 lines and it could be 50, rewrite it.

Ask yourself: "Would a senior engineer say this is overcomplicated?" If yes, simplify.

## 3. Surgical Changes

**Touch only what you must. Clean up only your own mess.**

When editing existing code:
- Don't "improve" adjacent code, comments, or formatting.
- Don't refactor things that aren't broken.
- Match existing style, even if you'd do it differently.
- If you notice unrelated dead code, mention it - don't delete it.

When your changes create orphans:
- Remove imports/variables/functions that YOUR changes made unused.
- Don't remove pre-existing dead code unless asked.

The test: Every changed line should trace directly to the user's request.

## 4. Goal-Driven Execution

**Define success criteria. Loop until verified.**

Transform tasks into verifiable goals:
- "Add validation" → "Write tests for invalid inputs, then make them pass"
- "Fix the bug" → "Write a test that reproduces it, then make it pass"
- "Refactor X" → "Ensure tests pass before and after"

For multi-step tasks, state a brief plan:
```
1. [Step] → verify: [check]
2. [Step] → verify: [check]
3. [Step] → verify: [check]
```

Strong success criteria let you loop independently. Weak criteria ("make it work") require constant clarification.

---

**These guidelines are working if:** fewer unnecessary changes in diffs, fewer rewrites due to overcomplication, and clarifying questions come before implementation rather than after mistakes.