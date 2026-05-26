# AI Agent Rules for Jupyter Notebook & ML Projects

## 1. Core Workflow & Context
* ALWAYS read `context.md` and `README.md` before executing any task.
* IF a prompt is ambiguous or missing necessary information (e.g., target variable, file path), **STOP AND ASK** me for clarification before generating code. Do not hallucinate assumptions.

## 2. Jupyter Notebook Guidelines
* **Cell Management:** Break code into logical, bite-sized cells. Avoid writing excessively long cells (keep under 50-70 lines per cell). One concept/step per cell.
* **Centralized Imports:** Dedicate the VERY FIRST cell exclusively for library imports. If a new library is needed later, DO NOT import it in the current cell. Instead, add the import statement to the top import cell.
* **Output Limit:** NEVER print entire dataframes or large arrays. Use `df.head()`, `df.info()`, `df.shape`, or limit array outputs to prevent context window overflow.
* **Markdown Cell Styling:** Use extremely concise, unnumbered headings for notebook sections (e.g., `# Library`, `# Data`, `# Training`). Avoid descriptive fillers or dividing the notebook into too many cluttered sub-sections.

## 3. Code Quality & Comments
* Write concise, clear, and easy-to-understand comments. Do not over-comment obvious code. Explain the "WHY", not just the "WHAT".
* **Reproducibility:** Always include a `random_state=42` or a `set_seed(42)` function for numpy, pandas, and models to ensure consistent results.

## 4. Model Optimization
* Always seek ways to optimize model quality and evaluation metrics.
* Proactively suggest Feature Engineering steps, Hyperparameter tuning strategies (e.g., Optuna, GridSearchCV), or alternative algorithms if the current model underperforms.