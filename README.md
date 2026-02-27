# 🧠 Adaptive LLM Router for Natural Language Inference

> **53.5% cheaper. 82.8% accurate. One intelligent routing decision.**

An MSc-level LLMOps system that dynamically routes NLI tasks between cheap Zero-shot prompts and expensive Few-shot prompts based on linguistic complexity — achieving near-flagship accuracy at half the cost.

---

## 🏆 Performance Leaderboard

| Strategy | Accuracy | Macro F1 | Cost | Verdict |
|:---|:---:|:---:|:---:|:---|
| **Few-Shot (GPT-4o)** | **84.0%** | **0.839** | $$ | Accuracy Champion |
| **BERT (DistilRoBERTa)** | **83.6%** | **0.836** | $0 | Value Champion |
| **Adaptive Router** | **82.8%** | **0.831** | $ | **ROI Champion (53.5% savings)** |
| Zero-Shot (GPT-4o) | 78.7% | 0.786 | $ | Baseline |
| Chain-of-Thought | 72.0% | 0.715 | $$$ | Over-reasoning trap |

---

## 🔍 The Problem

Large Language Models are powerful but expensive. Running every NLI sample through a 6-shot GPT-4o prompt burns tokens on cases that a simple Zero-shot prompt handles perfectly. Meanwhile, Chain-of-Thought (CoT) reasoning — long considered the gold standard — actually **hurts** performance on conversational text.

## 💡 The Solution: Adaptive Heuristic Routing

Our system extracts three linguistic features from each premise–hypothesis pair:

| Feature | Threshold | Signal |
|:---|:---:|:---|
| **Character Length** | > 220 chars | Dense, complex input |
| **Negation Count** | ≥ 3 | Multi-layered logic |
| **Lexical Overlap** | < 0.4 (Jaccard) | Low surface similarity |

If **all three** conditions are met → route to **6-Shot** (expensive but precise).  
Otherwise → route to **Zero-Shot** (cheap and fast).  
Special case: **Government** and **OUP** genres → **6-Shot + CoT hybrid**.

---

## 🚀 Quick Start

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
> **💰 Cost Warning**: Running the full pipeline via GPT-4o will incur OpenAI API costs (~$0.02–$0.04 per 250 samples). Use the `--sample_size` flag or modify `samples_per_genre` in `main.py` to test on a smaller batch first.

---

## 📂 Project Structure

```
adaptive-llm-router-nli/
├── main.py                     # Async pipeline entry point
├── run_inference.py             # Original sync inference runner
├── run_bert.py                  # BERT baseline evaluation
├── generate_final_comparison.py # Comparison chart generator
├── Final_Technical_Report.md    # 10-page technical report
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

## 🧪 Key Discovery: The CoT Paradox

We found that **Chain-of-Thought prompting reduces accuracy** for NLI tasks:

- **GPT-4o-mini**: Zero-shot (85.3%) vs CoT (68.0%) → **17% drop**
- **GPT-4o**: Few-shot (84.0%) vs CoT (72.0%) → **12% drop**

CoT forces the model to over-analyze conversational text, interpreting casual speech patterns as logical contradictions. This finding led us to eliminate CoT from the default pipeline entirely.

---

## 📊 Evaluation Dataset

- **Source**: [MultiNLI](https://huggingface.co/datasets/multi_nli) (Multi-Genre NLI)
- **Sampling**: 50 samples × 5 genres = 250 samples (stratified, label-balanced)
- **Genres**: Fiction, Government, Slate, Telephone, Travel
- **Labels**: Entailment, Neutral, Contradiction

The dataset is downloaded automatically via the `datasets` library on first run — no manual downloads needed.

---

## 🔧 Technologies

| Component | Technology |
|:---|:---|
| LLM Provider | OpenAI (GPT-4o, GPT-4o-mini) |
| Local Baseline | DistilRoBERTa (cross-encoder/nli-distilroberta-base) |
| Async Engine | asyncio + AsyncOpenAI |
| Feature Extraction | NLTK, custom heuristics |
| Evaluation | scikit-learn, pandas |
| Visualization | matplotlib, seaborn |

---

## 📄 License

This project is for academic and research purposes.

---

## 👤 Author

**Famul Kalam**  
[GitHub](https://github.com/Famulkalam)
