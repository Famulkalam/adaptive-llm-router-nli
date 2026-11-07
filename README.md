# Adaptive LLM Router for Natural Language Inference

Routes each natural language inference (NLI) example to either a cheap
zero-shot GPT-4o prompt or a more expensive 6-shot prompt, based on simple
linguistic complexity features, and compares the result with always using
the expensive prompt and with a free local model.

**Headline result:** the router kept 82.8% accuracy (against 84.0% for
always using 6-shot GPT-4o) while cutting token cost by 53.5%. A free
DistilRoBERTa cross-encoder reached 83.6%, which is the more practical
finding: for this task a small specialised model matches the LLM.

> Developed from group coursework on NLI at the University of Edinburgh
> Business School (2026), in which I led the evaluation pipeline.

---

## Results

| Strategy | Accuracy | Macro F1 | Relative cost |
|:---|:---:|:---:|:---|
| 6-shot GPT-4o | 84.0% | 0.839 | High |
| DistilRoBERTa cross-encoder (local) | 83.6% | 0.836 | None (runs locally) |
| Adaptive router (GPT-4o) | 82.8% | 0.831 | 53.5% below 6-shot |
| Zero-shot GPT-4o | 78.7% | 0.786 | Low |
| Chain-of-thought GPT-4o | 72.0% | 0.715 | Highest |

Notes for reading the table:

- The evaluation sets are small (a few hundred MultiNLI pairs), so
  differences of about 1 to 2 percentage points between the top three rows
  are within sampling noise.
- The local baseline is the off-the-shelf `cross-encoder/nli-distilroberta-base`
  model, which was already trained on MultiNLI, so it is tested on its home
  dataset. It was not fine-tuned in this project.
- Experiments in `Final_Technical_Report.md` were run at different stages
  with different sample sizes; `main.py` as committed reproduces the
  250-pair setup described below.

## The problem

Running every example through a 6-shot GPT-4o prompt spends tokens on cases
a zero-shot prompt already gets right. Chain-of-thought prompting, often
assumed to help, lowered accuracy here.

## The routing rule

Three features are extracted from each premise and hypothesis pair:

| Feature | Threshold | Signal |
|:---|:---:|:---|
| Character length | > 220 chars | Dense, complex input |
| Negation count | ≥ 3 | Multi-layered logic |
| Lexical overlap | < 0.4 (Jaccard) | Low surface similarity |

If **all three** conditions are met, the example goes to the 6-shot prompt;
otherwise it goes to zero-shot. Government and OUP genre examples go to a
6-shot plus chain-of-thought hybrid. (OUP only appears in MultiNLI's
mismatched validation split, so that part of the rule does not fire in the
default `main.py` run.)

---

## Quick start

### Prerequisites
- Python 3.10+
- OpenAI API key

### Installation

```bash
git clone https://github.com/Famulkalam/adaptive-llm-router-nli.git
cd adaptive-llm-router-nli

python3 -m venv venv
source venv/bin/activate

pip install -r requirements.txt
```

### Configuration
```bash
cp .env.example .env
# Edit .env and add your OpenAI API key
```

### Run the Adaptive Pipeline
```bash
python main.py
```

### Run BERT Baseline (No API key needed)
```bash
python run_bert.py
```

> [!WARNING]
> **Cost warning:** running the pipeline calls the OpenAI API and costs money; 6-shot and chain-of-thought prompts are several times more expensive than zero-shot. Lower `samples_per_genre` in `main.py` (default 50) to test on a small batch first.

---

## Project structure

```
adaptive-llm-router-nli/
├── main.py                     # Async pipeline entry point
├── run_inference.py             # Original sync inference runner
├── run_bert.py                  # BERT baseline evaluation
├── generate_final_comparison.py # Comparison chart generator
├── Final_Technical_Report.md    # Technical report
│
├── src/
│   ├── data/
│   │   ├── loader.py            # MultiNLI dataset loader (HuggingFace)
│   │   ├── preprocessor.py      # Stratified sampling & cleaning
│   │   └── features.py          # Linguistic feature extraction
│   │
│   ├── llm/
│   │   ├── base.py              # Abstract LLM interface
│   │   ├── openai_client.py     # Sync OpenAI client
│   │   ├── async_openai.py      # Async OpenAI client
│   │   ├── async_batch.py       # Concurrent batch processor
│   │   ├── batch.py             # Sync batch processor
│   │   ├── gemini.py            # Google Gemini client
│   │   └── mock_gemini.py       # Mock client for testing
│   │
│   ├── gatekeeper/
│   │   ├── adaptive_gatekeeper.py  # Heuristic complexity router
│   │   ├── classifier.py          # ML-based gatekeeper (v1)
│   │   └── router.py              # Routing logic
│   │
│   ├── prompts/
│   │   ├── templates.py         # Zero/Few/CoT/6-shot templates
│   │   └── manager.py           # Prompt management system
│   │
│   └── evaluation/
│       ├── metrics.py           # Classification metrics
│       ├── cost_analysis.py     # Cost-benefit analysis
│       └── visualizations.py    # Chart generation
│
├── notebooks/
│   └── analysis.ipynb           # Interactive analysis notebook
│
├── .env.example                 # Environment variable template
├── .gitignore                   # LLMOps-safe exclusions
└── requirements.txt             # Python dependencies
```

---

## Finding: chain-of-thought hurt accuracy

On this data, chain-of-thought prompting lowered accuracy:

- **GPT-4o-mini**: zero-shot 85.3% vs chain-of-thought 68.0% (17.3 percentage points lower)
- **GPT-4o**: few-shot 84.0% vs chain-of-thought 72.0% (12.0 points lower)

Inspecting errors suggested the model over-analysed conversational text and
read casual phrasing as contradiction. Chain-of-thought was therefore left
out of the default pipeline.

---

## Evaluation dataset

- **Source**: [MultiNLI](https://huggingface.co/datasets/multi_nli) (Multi-Genre NLI)
- **Split**: `validation_matched` (default in `main.py`)
- **Sampling**: 50 samples × 5 genres = 250 pairs (stratified, label-balanced)
- **Genres**: Fiction, Government, Slate, Telephone, Travel
- **Labels**: Entailment, Neutral, Contradiction

The dataset is downloaded automatically via the `datasets` library on first run, so no manual download is needed.

---

## Technologies

| Component | Technology |
|:---|:---|
| LLM Provider | OpenAI (GPT-4o, GPT-4o-mini) |
| Local Baseline | DistilRoBERTa cross-encoder (`cross-encoder/nli-distilroberta-base`, pre-trained on MultiNLI) |
| Async Engine | asyncio + AsyncOpenAI |
| Feature Extraction | NLTK, custom heuristics |
| Evaluation | scikit-learn, pandas |
| Visualization | matplotlib, seaborn |

---

## Licence

This project is for academic and research purposes.

---

## Author

Famul Kalam, MSc Business Analytics, University of Edinburgh.
[LinkedIn](https://www.linkedin.com/in/famulkalam)
