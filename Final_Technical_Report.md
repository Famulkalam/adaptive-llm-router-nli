# FINAL TECHNICAL REPORT: Adaptive NLI Classification System
**Project Title**: Optimized Natural Language Inference using Large Language Models and Specialized Baselines
**Author**: Famul Kalam
**Date**: February 4, 2026

---

## 1. Executive Summary
This report details the development and evaluation of an adaptive Natural Language Inference (NLI) system. The project journey encompassed five distinct phases: architectural validation using mock clients, performance benchmarking on `gpt-4o-mini`, flagship evaluation on `gpt-4o`, the implementation of a high-efficiency asynchronous heuristic pipeline, and a final comparison with a specialized BERT baseline. 

The final system achieves a peak accuracy of **84.0%** (Few-shot GPT-4o) and provides an optimized "Adaptive" mode that delivers **82.8%** accuracy while reducing token costs by **53.5%**. Furthermore, benchmarking against a local **BERT (DistilRoBERTa)** model revealed that specialized small models can achieve competitive accuracy (**83.6%**) at zero marginal cost, providing a clear roadmap for production scalability.

---

## 2. Introduction
Natural Language Inference (NLI) is a core task in NLP that requires determining the logical relationship (entailment, neutral, or contradiction) between a premise and a hypothesis. While flagship LLMs offer high accuracy, their cost and latency are often prohibitive for large-scale production. This project aims to bridge the gap between "State of the Art" and "Cost Efficiency" through an Adaptive Gatekeeper system.

---

## 3. Methodology & Technical Architecture
The system was designed as a modular Python package with the following components:

### 3.1 Data Pipeline
*   **Source**: The MultiNLI (Multi-Genre NLI) dataset from HuggingFace.
*   **Preprocessing**: Removal of "unknown" and ambiguous labels; stratified sampling of 50 samples per genre across 10 distinct genres (Fiction, Government, Slate, Telephone, Travel, etc.). Total dataset size: 500 samples.

### 3.2 Feature Engineering
We extracted three primary linguistic features to determine routing complexity:
1.  **Sentence Length**: Total word and character count of the premise-hypothesis pair.
2.  **Negation Count**: Detects presence of strict negation tokens (`not`, `never`, `no`, etc.).
3.  **Lexical Overlap**: Calculated using Jaccard Similarity between token sets (Intersection over Union).

### 3.3 The Adaptive Gatekeeper
A routing system that decides which prompting strategy to use for each sample. 
*   **Initial Approach**: Logistic Regression trained on linguistic features.
*   **Final Approach**: Heuristic-based routing (`Char_Len > 220` AND `Negations >= 3` AND `Overlap < 0.4`).

---

## 4. Phase 2: GPT-4o-mini & The "CoT Paradox"
The first real API testing was conducted on `gpt-4o-mini`. 

### 4.1 Results Table
| Strategy | Accuracy | Macro F1 |
|:--- |:---: |:---: |
| **Zero-shot** | **85.3%** | **0.852** |
| Chain-of-Thought | 68.0% | 0.675 |
| Adaptive | 81.3% | 0.810 |

### 4.2 Critical Insight: The CoT Paradox
We observed that forcing the "mini" model to reason step-by-step (CoT) significantly *decreased* performance. The model became overly pedantic, interpreting simple conversational nuances as "logical contradictions," leading to a 17% drop in accuracy compared to the simplest Zero-shot prompt.

---

## 5. Phase 3: GPT-4o Performance (The Flagship Benchmark)
Transitioning to the flagship `gpt-4o` model restored traditional scaling behaviors.

### 5.1 Results Table
| Strategy | Accuracy | Macro F1 |
|:--- |:---: |:---: |
| **Few-shot (5-shot)** | **84.0%** | **0.839** |
| Zero-shot | 78.7% | 0.786 |
| Chain-of-Thought | 72.0% | 0.715 |

### 5.2 Genre Breakdown Analysis
*   **Travel/Slate**: High performance due to clear, narrative structures.
*   **Telephone**: Lowest performance (46.7% for CoT). Spoken fillers ("uh-huh", "well") confuse the logic gate.
*   **Government**: Dense technical language benefitted from the Hybrid Few-shot/CoT strategy.

---

## 6. Phase 4: The Refined Async Pipeline
To prepare the system for MSc-level technical execution, we refactored the pipeline for **Asynchronous processing** and implemented a **ROI Calculator**.

### 6.1 System Optimization
*   **Async Batching**: Concurrent API calls reduced total runtime from minutes to seconds.
*   **Heuristic Refinement**: The gatekeeper was tuned to be "stingy," routing only the most complex cases to expensive 6-Shot prompts.

### 6.2 Token ROI Analysis
| Configuration | Total Tokens | Cost (Est) |
|:--- |:---: |:---: |
| Always Few-Shot | 125,000 | $0.037 |
| **Adaptive Gatekeeper** | **58,116** | **$0.017** |
| **Savings (%)** | **53.51%** | |

---

## 7. Phase 5: BERT vs. GPT-4o (Specialization vs. Scale)
The final experiment introduced a specialized **DistilRoBERTa-MNLI** model.

### 7.1 Comparisons
*   **BERT Accuracy**: **83.6%**
*   **Latency**: 6 seconds for the entire 250-sample set (local inference).
*   **Cost**: **$0.00**.

### 7.2 Strategy Leaderboard
1.  **Few-Shot (GPT-4o)**: 84.0% (The Ceiling)
2.  **BERT Baseline**: 83.6% (The Value King)
3.  **Refined Adaptive**: 82.8% (The Efficiency Hybrid)
4.  **Zero-Shot**: 78.7% (Baseline)

---

## 8. Conclusion & Recommendations
The project successfully demonstrated that while flagship LLMs provide the highest performance "ceiling," a well-engineered adaptive system or a specialized small model often yields better "Value per Performance."

### 8.1 Final Verdict
1.  **Simplicity is Robust**: For NLI, complex reasoning (CoT) is often counter-productive for general text.
2.  **Specialization is Powerful**: A BERT model fine-tuned on MNLI is 99% as accurate as GPT-4o Few-shot at 0% of the cost.
3.  **Adaptive Routing works**: Even in the absence of local models, routing based on linguistic complexity can cut operational costs by over 50% with negligible loss in precision.

---

## 9. Appendix: Technical Deliverables
*   **Source Code**: Modular package in `src/`.
*   **Entry Point**: `main.py` (Async implementation).
*   **Comparison Tool**: `generate_final_comparison.py`.
*   **BERT Run**: `run_bert.py`.
*   **Artifacts**: 
    *   `outputs/figures/final_comparison.png`
    *   `outputs/figures/genre_heatmap.png`

**End of Report.**
