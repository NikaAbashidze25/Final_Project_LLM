# Multi-LLM Collaborative Debate System

A production-ready system where multiple Large Language Models independently solve problems, cross-evaluate each other through structured peer review, refine their solutions based on feedback, and submit to a final judge LLM that selects the best answer. The system is designed to combat hallucination through diverse perspectives and adversarial review.

---

## Table of Contents

- [Project Overview](#project-overview)
- [Architecture](#architecture)
- [Workflow — All Stages](#workflow--all-stages)
- [Phase 1: Problem Dataset](#phase-1-problem-dataset)
- [Phase 2: System Implementation](#phase-2-system-implementation)
- [Phase 3: Evaluation & Analysis](#phase-3-evaluation--analysis)
- [Project Structure](#project-structure)
- [Setup & Installation](#setup--installation)
- [How to Run](#how-to-run)
- [API Configuration](#api-configuration)
- [Data Schemas](#data-schemas)
- [Evaluation Metrics](#evaluation-metrics)
- [Deliverables Checklist](#deliverables-checklist)
- [Important Notes](#important-notes)

---

## Project Overview

This system pits three LLMs against each other as independent **Solvers** to tackle a challenging problem, then has each Solver critique its peers in a structured **peer review** round. Each Solver then **refines** its solution based on feedback received. Finally, a fourth LLM acts as a **Judge** and selects the strongest final answer, which is returned to the user.

**Goal:** Reduce hallucination rates and improve answer quality compared to single-model or simple majority-vote baselines.

**Models used (configurable):** GPT-4, Claude, Gemini, Grok — or any combination, including using the same free model four times with different system prompts.

---

## Architecture

```
┌──────────────────────────────────────────────────────────────────┐
│                        Input: Problem                            │
└─────────────────────────────┬────────────────────────────────────┘
                              │
                    ┌─────────▼──────────┐
                    │  Stage 0: Role     │
                    │  Self-Assessment   │
                    │  (all 4 LLMs)      │
                    └─────────┬──────────┘
                              │
                    ┌─────────▼──────────┐
                    │  Stage 0.5:        │
                    │  Algorithmic Role  │
                    │  Assignment        │
                    └──┬──────┬──────┬───┘
                       │      │      │
              ┌────────▼┐  ┌──▼────┐ ┌▼────────┐
              │ Solver 1│  │Solver2│ │ Solver 3│
              └────────┬┘  └──┬────┘ └┬────────┘
                       │      │       │
              Stage 1: Independent Solutions (no communication)
                       │      │       │
              ┌────────▼──────▼───────▼────────┐
              │        Stage 2: Peer Review     │
              │  Each Solver reviews the other  │
              │  two (2 reviews per Solver)     │
              └────────┬──────┬───────┬─────────┘
                       │      │       │
              Stage 3: Refinement (each Solver gets 2 reviews)
                       │      │       │
              ┌────────▼──────▼───────▼─────────┐
              │  Stage 4: Final Judge            │
              │  Receives all original solutions,│
              │  all peer reviews, all refined   │
              │  solutions → picks winner        │
              └────────────────┬─────────────────┘
                               │
                    ┌──────────▼──────────┐
                    │   Final Answer      │
                    │   returned to user  │
                    └─────────────────────┘
```

---

## Workflow — All Stages

### Stage 0: Role Self-Assessment

Every one of the four LLMs (GPT-4, Claude, Gemini, Grok) receives the original problem and is asked to self-assess which role it is best suited for.

**Input to each LLM:**
- The problem statement
- A list of available roles: `Solver`, `Judge`
- Instructions to return a structured JSON response

**Required output from each LLM:**

```json
{
  "role_preferences": ["Solver", "Judge"],
  "confidence_by_role": {
    "Solver": 0.85,
    "Judge": 0.75
  },
  "reasoning": "I should be Solver because I'm strong at mathematical reasoning and step-by-step derivations. I would make a decent Judge but my comparative strengths lie in solution generation."
}
```

All four models must return this self-assessment before proceeding.

---

### Stage 0.5: Algorithmic Role Assignment

After collecting all four self-assessments, a **deterministic algorithm** (not an LLM) assigns the final roles.

**Rules for the algorithm:**
- Exactly 3 models must be assigned the role `Solver`.
- Exactly 1 model must be assigned the role `Judge`.
- Prefer models that listed `Judge` in their `role_preferences` and have higher `confidence_by_role["Judge"]` scores for the Judge seat.
- Among the remaining three, assign `Solver` roles.
- In the case of ties, use a predefined priority order (e.g., GPT-4 > Claude > Gemini > Grok).
- The assignment must be reproducible given the same inputs.

**Output:** A mapping such as:

```json
{
  "Solver_1": "claude",
  "Solver_2": "gpt-4",
  "Solver_3": "gemini",
  "Judge": "grok"
}
```

This assignment is logged and used for all subsequent stages.

---

### Stage 1: Independent Solution Generation

Each of the three Solvers independently generates a complete solution to the problem.

**Strict rules:**
- No communication or information sharing between Solvers at this stage.
- Each Solver must output a step-by-step solution with explicit reasoning.
- Each Solver must state a final answer clearly.
- Each Solver must provide a confidence score for their answer.

**Required output from each Solver:**

```json
{
  "solver_id": "solver_1",
  "model": "claude",
  "solution_steps": [
    {"step": 1, "description": "...", "reasoning": "..."},
    {"step": 2, "description": "...", "reasoning": "..."}
  ],
  "final_answer": "...",
  "confidence": 0.82,
  "assumptions": ["Assuming n >= 1", "Assuming integers only"]
}
```

All three solutions are stored and passed into Stage 2.

---

### Stage 2: Peer Review Round

Each Solver reviews the solutions produced by the **other two** Solvers. This means:
- Solver 1 reviews Solver 2 and Solver 3.
- Solver 2 reviews Solver 1 and Solver 3.
- Solver 3 reviews Solver 1 and Solver 2.

Total reviews produced: **6** (2 per Solver).

**Each Solver receives:**
- The original problem statement
- The full solution (all steps) of the peer being reviewed
- Instructions to be critical, precise, and constructive

**Required output for each review:**

```json
{
  "reviewer_id": "solver_1",
  "solution_reviewed": "solver_2",
  "evaluation": {
    "strengths": [
      "Clear step-by-step derivation in steps 1–3",
      "Correct formula application in step 4"
    ],
    "weaknesses": [
      "Step 5 makes an unjustified logical leap",
      "Edge case where n=0 is not handled"
    ],
    "errors": [
      {
        "location": "Step 5",
        "error_type": "logical_error",
        "description": "Claims X implies Y, but this is false when Z is negative. Counterexample: ...",
        "severity": "critical"
      },
      {
        "location": "Step 2",
        "error_type": "arithmetic_error",
        "description": "3 * 7 = 21, not 24 as stated",
        "severity": "minor"
      }
    ],
    "suggested_changes": [
      "Reconsider step 5 using the counterexample where Z < 0",
      "Add a check for n=0 before applying the recursive formula"
    ]
  },
  "overall_assessment": "promising_but_flawed"
}
```

Allowed values for `overall_assessment`: `"correct"`, `"promising_but_flawed"`, `"fundamentally_wrong"`, `"unclear"`.

Allowed values for `error_type`: `"logical_error"`, `"arithmetic_error"`, `"missing_case"`, `"wrong_formula"`, `"unjustified_assumption"`, `"other"`.

Allowed values for severity: `"critical"`, `"major"`, `"minor"`.

---

### Stage 3: Refinement Based on Feedback

Each Solver receives the **two peer reviews** written about their solution. They must then produce a refined solution that explicitly addresses each critique.

**Each Solver must:**
- Address every critique point-by-point.
- Either accept the critique and correct the error, or reject the critique with a clear justification.
- Produce a complete revised solution.
- Update their confidence score.

**Required output from each Solver:**

```json
{
  "solver_id": "solver_1",
  "model": "claude",
  "critique_responses": [
    {
      "from_reviewer": "solver_2",
      "critique": "Step 5 makes an unjustified leap",
      "response": "Reviewer is correct. I assumed commutativity without proving it. Fixed by adding explicit proof in revised step 5.",
      "accepted": true
    },
    {
      "from_reviewer": "solver_3",
      "critique": "Missing edge case where n=0",
      "response": "This edge case does not apply because the problem specifies n >= 1 in the constraints.",
      "accepted": false
    }
  ],
  "refined_solution_steps": [
    {"step": 1, "description": "...", "reasoning": "..."},
    {"step": 2, "description": "...", "reasoning": "..."}
  ],
  "refined_final_answer": "...",
  "confidence": 0.90,
  "changes_summary": "Corrected step 5 to include explicit commutativity proof. Maintained original edge case handling with added justification."
}
```

All three refined solutions, along with the critique responses, are stored and passed to Stage 4.

---

### Stage 4: Final Judgment

The Judge LLM receives the full debate context and selects the best final answer.

**Judge receives:**
- The original problem statement
- All three original solutions (from Stage 1)
- All six peer reviews (from Stage 2)
- All three refined solutions (from Stage 3)

**Judge must evaluate:**
- Logical correctness of each solution
- Quality of the refinement process (did the Solver improve under critique?)
- Consistency of the final answer
- Confidence and justification quality

**Required output from the Judge:**

```json
{
  "judge_model": "grok",
  "winner": "solver_1",
  "confidence": 0.88,
  "ranking": {
    "solver_1": 1,
    "solver_3": 2,
    "solver_2": 3
  },
  "reasoning": "Solver 1's solution is strongest because it correctly handled the base case, provided a valid inductive step, and its refinement accepted a critical fix from peer review. Solver 3 reached the same answer through a less rigorous path. Solver 2's final answer is incorrect due to an arithmetic error in step 4 that was not corrected.",
  "correct_answer": "...",
  "notes": "All three Solvers agreed on approach but diverged at step 5. Solver 1's handling of step 5 post-review is the most defensible."
}
```

The system then copies `correct_answer` (or the `refined_final_answer` from the `winner`) and returns it to the user.

---

## Phase 1: Problem Dataset

### Requirements

- Construct a dataset of exactly **25 challenging problems**.
- Each problem must have a **verifiable correct answer** (ground truth).
- Problems must be **difficult enough that single LLM attempts commonly fail**.
- Store problems in a structured format (JSON or CSV) with fields: `id`, `category`, `problem`, `correct_answer`, `difficulty`, `source_notes`.

### Problem Categories

#### 1. Mathematical / Logical Reasoning (6–7 problems)
- Complex combinatorics, probability puzzles, number theory proofs.
- Problems where LLMs commonly make calculation errors or unjustified logical leaps.
- **Example:** "In how many ways can you tile a 3×8 rectangle with 2×1 dominoes?"

#### 2. Physics & Scientific Reasoning (6–7 problems)
- Multi-step physics problems requiring formula application and unit analysis.
- Counterintuitive scenarios (Monty Hall-style physics problems).
- **Example:** "A ladder leans against a frictionless wall. Derive the minimum coefficient of friction needed with the floor to prevent slipping."

#### 3. Logic Puzzles & Constraint Satisfaction (6 problems)
- Multi-agent reasoning (knights and knaves, truth-tellers and liars).
- Constraint satisfaction with interdependent rules.
- **Example:** "Five people of different nationalities live in five colored houses. Given 15 clues about their pets, drinks, and cigarette brands, who owns the fish?"

#### 4. Strategic Game Theory (5–6 problems)
- Optimal strategy derivation in games with incomplete information.
- Backward induction, Nash equilibria calculations.
- **Example:** "In a two-player auction where bids are sealed and the highest bidder pays the second-highest bid, what is the optimal bidding strategy?"

### Dataset Format

```json
[
  {
    "id": "prob_001",
    "category": "mathematical_reasoning",
    "problem": "In how many ways can you tile a 3×8 rectangle with 2×1 dominoes?",
    "correct_answer": "153",
    "difficulty": "hard",
    "source_notes": "Classic combinatorics tiling problem"
  }
]
```

---

## Phase 2: System Implementation

### LLM Roles

| Role | Count | Description |
|------|-------|-------------|
| Solver | 3 | Independently solves the problem; participates in peer review and refinement |
| Judge | 1 | Reviews all solutions and reviews, selects the best final answer |

### Model Options

**Option A (Free):** Use a single free model (e.g., Gemini Flash, Llama via Groq) for all four roles with different system prompts and API calls.

**Option B (Paid):** Use four distinct models — GPT-4, Claude, Gemini, Grok — each in a different role. Minimum cost is typically $5 per model for API access.

Both options are acceptable. Points will **not** be deducted for using only free models, but points **will** be deducted for incomplete implementation.

### System Prompt Design

Each role requires a carefully crafted system prompt:

- **Solver system prompt:** Instructs the model to solve independently, reason step-by-step, avoid looking for consensus, and be explicit about assumptions.
- **Reviewer system prompt:** Instructs the model to be critical, identify errors by type and severity, and provide actionable suggestions.
- **Refiner system prompt:** Instructs the model to engage with each critique, accept valid corrections, and defend rejected critiques.
- **Judge system prompt:** Instructs the model to evaluate all evidence objectively, not favor the model it is closest to, and justify its ranking.

### API Call Structure

Each stage makes structured API calls. All responses must be parsed and validated before proceeding to the next stage. If a response is malformed, implement **retry logic** (up to 3 retries with adjusted prompts).

### Error Handling Requirements

- Validate JSON structure of every LLM response before using it.
- If a Solver fails to produce a valid response after retries, log the failure and exclude that Solver from peer review (the remaining two still proceed).
- If the Judge fails, fall back to selecting the Solver with the highest `confidence` score in their refined solution.
- All errors must be logged with timestamps, model name, stage, and error description.

---

## Phase 3: Evaluation & Analysis

### Quantitative Metrics

#### System-Level Performance

| Metric | Definition |
|--------|-----------|
| **Overall Accuracy** | % of the 25 problems where the final returned answer matches the ground truth |
| **Improvement Rate** | % of problems where at least one Solver's refined answer is more correct than their original answer |
| **Consensus Rate** | % of problems where all 3 Solvers' refined answers match each other |
| **Judge Accuracy** | When Solvers disagree, % of cases where the Judge correctly picks the Solver with the right answer |

#### Per-Model Performance

Track for each model individually:
- Accuracy as a Solver (% of problems where their solution was correct)
- Accuracy as a Judge (% of times the Judge's selected winner had the correct answer)
- Peer review quality (how often a critique flagged a real error)

### Baseline Comparisons

Three systems must be compared:

| System | Description |
|--------|-------------|
| **Single-LLM Baseline** | Ask GPT-4 (or Claude, Gemini, Grok) each problem once, no debate |
| **Simple Voting Baseline** | 3 independent solutions, pick majority answer (no peer review, no refinement) |
| **Full Debate System** | Complete pipeline: self-assessment → role assignment → solve → review → refine → judge |

### Required Plots

All of the following plots **must** be generated and included in the repository:

1. **Overall accuracy bar chart** — Comparing Single-LLM Baseline vs Voting Baseline vs Full Debate System.
2. **Per-category accuracy** — Breakdown by problem category (Math, Physics, Logic, Game Theory) across all three systems.
3. **Improvement rate over refinement** — For each problem, whether refinement helped, hurt, or had no effect.
4. **Consensus rate by category** — Heatmap or bar chart showing how often Solvers agreed by problem type.
5. **Judge accuracy vs. Solver disagreement** — Scatter or bar showing Judge performance when Solvers disagree.
6. **Per-model accuracy** — Each model's accuracy as Solver and as Judge.
7. **Confidence calibration** — Scatter plot of Solver reported confidence vs. actual correctness.

Use `matplotlib` or `seaborn` for all plots. Save as `.png` files in a `/plots` directory.

---

## Project Structure

```
multi-llm-debate/
│
├── README.md                        # This file
│
├── data/
│   ├── problems.json                # 25 problems with ground truth answers
│   └── results/
│       ├── stage0_role_assessments/ # Raw JSON from Stage 0
│       ├── stage1_solutions/        # Raw JSON from Stage 1
│       ├── stage2_reviews/          # Raw JSON from Stage 2
│       ├── stage3_refinements/      # Raw JSON from Stage 3
│       ├── stage4_judgments/        # Raw JSON from Stage 4
│       └── final_answers.json       # Consolidated final answers per problem
│
├── src/
│   ├── __init__.py
│   ├── config.py                    # API keys, model names, constants
│   ├── models/
│   │   ├── __init__.py
│   │   ├── base_llm.py              # Abstract LLM interface
│   │   ├── gpt4.py                  # GPT-4 implementation
│   │   ├── claude.py                # Claude implementation
│   │   ├── gemini.py                # Gemini implementation
│   │   └── grok.py                  # Grok implementation
│   │
│   ├── stages/
│   │   ├── __init__.py
│   │   ├── stage0_assessment.py     # Role self-assessment logic
│   │   ├── stage0_5_assignment.py   # Deterministic role assignment algorithm
│   │   ├── stage1_solve.py          # Independent solution generation
│   │   ├── stage2_review.py         # Peer review logic
│   │   ├── stage3_refine.py         # Refinement logic
│   │   └── stage4_judge.py          # Final judgment logic
│   │
│   ├── prompts/
│   │   ├── solver_system_prompt.txt
│   │   ├── reviewer_system_prompt.txt
│   │   ├── refiner_system_prompt.txt
│   │   ├── judge_system_prompt.txt
│   │   └── role_assessment_prompt.txt
│   │
│   ├── pipeline.py                  # Orchestrates all stages end-to-end
│   ├── evaluator.py                 # Computes all metrics and baselines
│   └── utils.py                     # JSON validation, retry logic, logging
│
├── notebooks/
│   ├── 01_dataset_construction.ipynb
│   ├── 02_system_demo.ipynb         # Walk through one problem end-to-end
│   └── 03_evaluation_analysis.ipynb # All plots and metric computation
│
├── plots/
│   ├── overall_accuracy.png
│   ├── per_category_accuracy.png
│   ├── improvement_rate.png
│   ├── consensus_rate.png
│   ├── judge_accuracy.png
│   ├── per_model_accuracy.png
│   └── confidence_calibration.png
│
├── logs/
│   └── run_YYYYMMDD_HHMMSS.log      # Auto-generated run logs
│
├── requirements.txt
└── .env.example                     # Template for API keys
```

---

## Setup & Installation

### Prerequisites

- Python 3.10 or higher
- `pip` or `conda`

### Installation

```bash
# Clone the repository
git clone https://github.com/NikaAbashidze25/Final_Project_LLM.git
cd Final_Project_LLM

# Create and activate a virtual environment
python -m venv venv
source venv/bin/activate        # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### Environment Variables

Copy `.env.example` to `.env` and fill in your API keys:

```bash
cp .env.example .env
```

`.env` file structure:

```
OPENAI_API_KEY=sk-...
ANTHROPIC_API_KEY=sk-ant-...
GOOGLE_API_KEY=...
GROK_API_KEY=...

# Optional: override default models
OPENAI_MODEL=gpt-4o
ANTHROPIC_MODEL=claude-opus-4-8
GEMINI_MODEL=gemini-1.5-pro
GROK_MODEL=grok-2
```

If using a single free model for all four roles, only one key is needed. Set the others to empty strings.

### `requirements.txt`

```
openai>=1.0.0
anthropic>=0.25.0
google-generativeai>=0.5.0
python-dotenv>=1.0.0
requests>=2.31.0
pydantic>=2.0.0
matplotlib>=3.8.0
seaborn>=0.13.0
pandas>=2.0.0
numpy>=1.26.0
jupyter>=1.0.0
tqdm>=4.66.0
```

---

## How to Run

### Run the full pipeline on all 25 problems

```bash
python src/pipeline.py --problems data/problems.json --output data/results/
```

### Run on a single problem (for testing)

```bash
python src/pipeline.py --problem-id prob_001 --problems data/problems.json --output data/results/
```

### Run evaluation and generate all plots

```bash
python src/evaluator.py --results data/results/ --output plots/
```

### Run via Jupyter notebooks

```bash
jupyter notebook notebooks/
```

Open notebooks in order:
1. `01_dataset_construction.ipynb` — build and inspect the 25 problems
2. `02_system_demo.ipynb` — run one problem through all 4 stages interactively
3. `03_evaluation_analysis.ipynb` — compute metrics and generate all plots

---

## API Configuration

### Using Multiple Models (Option B)

Set all four API keys in `.env`. The `config.py` file maps each role to a specific model:

```python
ROLE_MODEL_MAP = {
    "Solver_1": "claude",
    "Solver_2": "gpt-4",
    "Solver_3": "gemini",
    "Judge": "grok"
}
```

This is overridden by the Stage 0.5 assignment algorithm at runtime for each problem.

### Using a Single Free Model (Option A)

Set only one API key and configure `config.py` to route all roles to that model:

```python
FREE_MODEL_MODE = True
FREE_MODEL_PROVIDER = "gemini"   # or "groq", "together", etc.
```

In free model mode, each role still gets a distinct system prompt and makes a separate API call — the only difference is all calls go to the same underlying model.

### Rate Limiting

- Add a `time.sleep()` delay between consecutive API calls to the same provider.
- Default: 1 second between calls. Configurable in `config.py`.
- Implement exponential backoff on 429 (rate limit) errors.

---

## Data Schemas

### Problem Schema (`problems.json`)

```json
{
  "id": "string (prob_001 to prob_025)",
  "category": "mathematical_reasoning | physics_reasoning | logic_puzzle | game_theory",
  "problem": "string — full problem statement",
  "correct_answer": "string — verifiable ground truth",
  "difficulty": "medium | hard | very_hard",
  "source_notes": "string — optional attribution or notes"
}
```

### Final Results Schema (`final_answers.json`)

```json
{
  "problem_id": "prob_001",
  "winner_solver": "solver_1",
  "final_answer": "153",
  "correct_answer": "153",
  "is_correct": true,
  "judge_confidence": 0.88,
  "consensus": false,
  "improvement_occurred": true,
  "role_assignment": {
    "Solver_1": "claude",
    "Solver_2": "gpt-4",
    "Solver_3": "gemini",
    "Judge": "grok"
  }
}
```

---

## Evaluation Metrics

### Definitions

```
Overall Accuracy     = (# problems with correct final answer) / 25

Improvement Rate     = (# problems where ≥1 Solver's refined answer is more correct
                        than their original answer) / 25

Consensus Rate       = (# problems where all 3 Solvers' refined answers match) / 25

Judge Accuracy       = (# problems where Solvers disagree AND Judge picks the correct one)
                       / (# problems where Solvers disagree)
```

### Baseline Definitions

```
Single-LLM Baseline  = Run only GPT-4 (or your best model) once per problem, no debate.
                       Record accuracy over 25 problems.

Voting Baseline      = Run 3 LLMs independently (no reviews, no refinement).
                       Pick the majority answer. Record accuracy over 25 problems.

Full System          = Complete pipeline as described above. Record accuracy over 25 problems.
```

All three baselines must be evaluated on the **same 25 problems** to ensure a fair comparison.

---

## Deliverables Checklist

- [ ] GitHub repository with meaningful commit history (not a single upload commit)
- [ ] `README.md` — this file, fully filled in with setup instructions
- [ ] `data/problems.json` — 25 problems with ground truth answers
- [ ] Complete implementation of all 5 stages in `src/stages/`
- [ ] Working end-to-end pipeline (`src/pipeline.py`)
- [ ] Evaluation script (`src/evaluator.py`)
- [ ] 3 Jupyter notebooks (dataset construction, system demo, evaluation)
- [ ] All 7 required plots saved in `/plots`
- [ ] `.env.example` file (without real API keys)
- [ ] `requirements.txt` with pinned versions
- [ ] Run logs from at least one complete evaluation run in `/logs`
- [ ] `data/results/final_answers.json` — consolidated results for all 25 problems

---

## Important Notes

### On API Costs

- Most models (GPT-4, Claude, Gemini Pro, Grok) require paid API access.
- The minimum top-up is typically $5 per provider.
- **Free alternative:** Use a free-tier model (Gemini Flash, Llama via Groq, etc.) for all four roles with different system prompts. Points will not be deducted for this.
- API rate limit failures are **not accepted as an excuse** for non-delivery. Plan your API usage in advance and implement retry logic.

### On Commits

- This is a group project. Every member must have meaningful commits.
- Do not write all code locally and push in a single commit.
- Commit frequently and use descriptive commit messages.

### On JSON Validation

- Every LLM response must be parsed and validated against the expected schema before moving to the next stage.
- Use `pydantic` models or equivalent for schema validation.
- Log any validation failures and trigger retry logic automatically.

### On Reproducibility

- The role assignment algorithm (Stage 0.5) must be deterministic: same inputs → same outputs.
- Save all intermediate outputs to disk after each stage so runs can be resumed if interrupted.
- Use a fixed random seed where applicable.

---

## Authors

- Nika Abashidze
- (teammate 2)
- (teammate 3)

---

## License

This project is for academic use only.
