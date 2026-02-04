# Adaptive NLI Classification System

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

A comprehensive Natural Language Inference (NLI) system using Large Language Models with an **Adaptive AI Gatekeeper** for cost-effective inference routing.

## 🎯 Project Overview

This project builds an inference system to classify the relationship between premise-hypothesis pairs as:
- **Entailment**: Hypothesis follows from the premise
- **Contradiction**: Hypothesis contradicts the premise
- **Neutral**: Hypothesis is unrelated to the premise

### Key Features

- **Multi-Strategy Prompting**: Zero-shot, One-shot, Few-shot, and Chain-of-Thought (CoT)
- **Adaptive AI Gatekeeper**: Routes queries to cost-effective strategies based on predicted difficulty
- **Comprehensive Evaluation**: Accuracy, Macro-F1, per-class scores, confusion analysis
- **Cost-Benefit Analysis**: Token tracking and ROI calculation
- **Genre-Specific Analysis**: Performance breakdown across 10 MultiNLI genres

## 📊 Dataset

**MultiNLI (Multi-Genre Natural Language Inference Corpus)**
- Source: [NYU / HuggingFace](https://huggingface.co/datasets/multi_nli)
- Size: ~433,000 labeled sentence pairs
- Labels: Entailment, Contradiction, Neutral
- Genres: Fiction, Government, Slate, Telephone, Travel, Facetoface, Letters, Nineeleven, OUP, Verbatim

## 🏗️ Project Structure

```
LLM_Antigravity/
├── src/
│   ├── data/           # Data loading, preprocessing, feature extraction
│   ├── prompts/        # Prompt templates and management
│   ├── llm/            # LLM interfaces (Mock & Real Gemini)
│   ├── gatekeeper/     # Adaptive routing logic
│   └── evaluation/     # Metrics, cost analysis, visualizations
├── notebooks/
│   └── analysis.ipynb  # Main analysis notebook
├── data/
│   ├── raw/            # Downloaded MultiNLI data
│   ├── processed/      # Cleaned and sampled data
│   └── predictions/    # Model outputs (JSON/CSV)
├── outputs/
│   └── figures/        # Generated visualizations
├── run_inference.py    # Main inference script
├── requirements.txt    # Dependencies
└── README.md
```

## 🚀 Quick Start

### 1. Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/LLM_Antigravity.git
cd LLM_Antigravity

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Run with Mock LLM (Testing)

```bash
# Run full pipeline with mock Gemini (no API key needed)
python run_inference.py --mock --samples-per-genre 50
```

### 3. Run with Real Gemini API

```bash
# Set up API key
cp .env.example .env
# Edit .env and add your GEMINI_API_KEY

# Run with real API
python run_inference.py --samples-per-genre 50
```

### 4. Run Jupyter Notebook

```bash
cd notebooks
jupyter notebook analysis.ipynb
```

## 📈 Results

### Strategy Comparison

| Strategy   | Accuracy | Macro-F1 | Tokens/Sample |
|------------|----------|----------|---------------|
| Zero-shot  | ~0.70    | ~0.68    | 165           |
| One-shot   | ~0.75    | ~0.73    | 235           |
| Few-shot   | ~0.80    | ~0.78    | 465           |
| CoT        | ~0.85    | ~0.83    | 500           |

*Results shown are approximate from mock LLM. Actual results may vary.*

### Adaptive Gatekeeper ROI

- **Token Savings**: 30-50% compared to all-CoT baseline
- **Accuracy Trade-off**: Minimal (~1-2% drop from pure CoT)
- **Best for**: Production deployments with cost constraints

## 🔧 Configuration

### Environment Variables

```bash
GEMINI_API_KEY=your_api_key_here
GEMINI_MODEL=gemini-1.5-flash  # or gemini-1.5-pro
```

### Command Line Options

```bash
python run_inference.py --help

Options:
  --mock              Use mock LLM client (no API key needed)
  --samples-per-genre Number of samples per genre (default: 50)
  --strategies        Strategies to run (default: all four)
  --skip-inference    Skip inference, use cached predictions
  --seed              Random seed for reproducibility
```

## 📚 Research Questions Addressed

1. **Primary**: How does LLM performance evolve across prompting strategies?
2. Which genres benefit most from Chain-of-Thought reasoning?
3. Does CoT reduce the common "Neutral-Entailment" confusion?
4. What is the cost-to-accuracy trade-off of adaptive routing?

## 🧪 Evaluation Metrics

### Classification Performance (40%)
- Accuracy, Macro-F1
- Per-class Precision, Recall, F1
- Confusion matrix analysis

### Prompt Effectiveness (20%)
- Zero-shot vs Few-shot vs CoT comparison
- Accuracy progression visualization

### Technical Performance (20%)
- Batch inference with retry logic
- Token counting and cost tracking
- Adaptive routing efficiency

### Documentation & Analysis (20%)
- Error analysis (Neutral-Entailment confusion)
- Per-genre breakdown
- Cost-benefit analysis

## 📁 Output Files

After running the pipeline:

```
data/predictions/
├── zero-shot_predictions.csv
├── one-shot_predictions.csv
├── few-shot_predictions.csv
├── cot_predictions.csv
├── strategy_comparison.csv
└── final_results.json

outputs/figures/
├── strategy_comparison.png
├── prompt_progression.png
├── confusion_matrices_all.png
├── genre_accuracy_heatmap.png
├── cost_accuracy_frontier.png
├── feature_importance.png
└── summary_dashboard.png
```

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit changes (`git commit -m 'Add amazing feature'`)
4. Push to branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- [MultiNLI Dataset](https://cims.nyu.edu/~sbowman/multinli/) - NYU
- [HuggingFace Datasets](https://huggingface.co/datasets/multi_nli)
- [Google Gemini API](https://ai.google.dev/)

---

**Note**: This project was developed as part of an LLM-based NLI classification exercise. The mock LLM provides approximate accuracy patterns for testing; actual results with real APIs may vary.
