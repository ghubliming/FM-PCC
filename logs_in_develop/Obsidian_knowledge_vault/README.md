# Obsidian Knowledge Vault & Historical Notes Archive

> ⚠️ **CRITICAL DIRECTIVE FOR AI ASSISTANTS / AGENTS**
>
> **DO NOT USE THIS DIRECTORY BY DEFAULT.**
> This folder contains a raw archive of personal Markdown notes exported from Obsidian (early brainstorms, paper studies, math derivations, historical experiment logs, and bug investigation records).
>
> **MANY NOTES IN THIS DIRECTORY CONTAIN OUTDATED, EXPERIMENTAL, OR SUPERSEDED INFORMATION.**
>
> Follow the strict protocol outlined below and detailed in [`AI_USAGE_GUIDELINES.md`](./AI_USAGE_GUIDELINES.md).

---

## 1. Directory Overview & Purpose

This directory serves as a central repository for historical knowledge, research notes, and idea evolution imported from the author's Obsidian vault. It contains:

* **Theoretical Research & Math**: Paper reading notes, continuous vs. discrete time derivations, flow matching formulations, ODE/SDE integrations, score matching / probability flow derivations.
* **Architecture Evolution & Ideation**: Brainstorming notes, design logs for Gen1 through Gen15+, Vector Field (VF) conditioning, MeanFlow, AlphaFlow, Action Chunking / Horizons, D3IL / DPCC integrations.
* **Exploratory Data & Bug Logs**: Historical test result audits, evaluation discrepancies (e.g., ODE step misalignments), trial-and-error logs from Colab/SLURM runs.

---

## 2. Mandatory Rules for AI / LLM Usage

### 🛑 Rule 1: Default Stance — Do Not Access or Ingest
* In regular programming, code generation, refactoring, debugging, or standard queries, **DO NOT scan, read, or cite files in this directory**.
* Treat this directory as a **quarantined cold archive**.
* **The active codebase and current configuration files are the single source of truth.** Claims or code snippets found in these notes must never override current active repository code.

---

### 🟢 Rule 2: Allowed Access Triggers

AI agents may ONLY access this directory under two conditions:

1. **Explicit User Command**:
   * The user directly instructs the AI to read, search, or check Obsidian notes (e.g., *"Search my obsidian notes for..."*, *"Check what note [[X]] says"*, *"Look up Obsidian_knowledge_vault"*).
   * In this case, the AI is permitted to read and analyze freely as requested.

2. **High-Necessity Deep Retrieval (Real Need Only)**:
   * When required context is missing from the active codebase and recent logs, and there is a genuine need for:
     * **Historical Idea Genesis**: Understanding *why* a specific architecture, formula, or heuristic was originally introduced.
     * **Deep Math & Paper Details**: Detailed mathematical steps, derivations, or theoretical lemmas written during paper analysis.
     * **Historical Anomaly / Bug Audits**: Deep retrospective analysis on early experimental data or historical bugs.

---

### ⚠️ Rule 3: Mandatory Response Disclosure / Notice
If the AI utilizes or cites any information, equation, or rationale from this directory, **it MUST prepend or include a prominent disclaimer in the output/chatbox**:

```markdown
> ⚠️ **Notice: Retrieved from Historical Obsidian Notes (`logs_in_develop/Obsidian_knowledge_vault/<note_name>.md`)**
> *Information in this note represents historical/exploratory context and may contain outdated, speculative, or superseded implementations. Please verify against active codebase.*
```

---

### 🔍 Rule 4: Mandatory Cross-Verification
* Never assume code snippets, hyperparameters, or mathematical constants in these notes reflect the current repository state.
* Always cross-reference against active files in `/workspaces/FM-PCC/`.
* If a note contradicts active code, active code takes precedence.

---

## 3. Reference Files

* For detailed operational rules for AI agents: [`AI_USAGE_GUIDELINES.md`](./AI_USAGE_GUIDELINES.md)
* For general repository development history: [`../MASTER_TEST_HISTORY.md`](../MASTER_TEST_HISTORY.md)
