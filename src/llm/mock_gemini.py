"""
Mock Gemini Client

Simulates Gemini API responses for testing without API costs.
Uses rule-based logic with configurable accuracy to simulate realistic behavior.
"""

import random
import time
import re
from typing import Optional, Dict, Any

from .base import BaseLLM, LLMResponse


class MockGeminiClient(BaseLLM):
    """
    Mock Gemini client for testing.
    
    Simulates different accuracy levels based on prompting strategy:
    - Zero-shot: ~70% accuracy
    - One-shot: ~75% accuracy
    - Few-shot: ~80% accuracy
    - CoT: ~85% accuracy
    
    Also simulates:
    - Realistic token counts
    - Variable latency
    - Occasional parsing difficulties
    """
    
    # Base accuracy by strategy (calibrated to approximate benchmarks)
    STRATEGY_ACCURACY = {
        "zero-shot": 0.70,
        "one-shot": 0.75,
        "few-shot": 0.80,
        "cot": 0.85
    }
    
    # Token counts by strategy (approximate)
    STRATEGY_TOKENS = {
        "zero-shot": {"prompt": 150, "response": 15},
        "one-shot": {"prompt": 220, "response": 15},
        "few-shot": {"prompt": 450, "response": 15},
        "cot": {"prompt": 350, "response": 150}  # CoT has longer responses
    }
    
    # Confusion matrix biases (which errors are more common)
    # Format: (true_label, predicted_label) -> relative probability
    CONFUSION_BIASES = {
        ("neutral", "entailment"): 0.4,  # Neutral often confused with entailment
        ("entailment", "neutral"): 0.3,
        ("contradiction", "neutral"): 0.2,
        ("neutral", "contradiction"): 0.15,
        ("entailment", "contradiction"): 0.1,
        ("contradiction", "entailment"): 0.1,
    }
    
    def __init__(
        self,
        model_name: str = "gemini-1.5-flash-mock",
        base_latency_ms: float = 100,
        random_seed: Optional[int] = None
    ):
        """
        Initialize the mock client.
        
        Args:
            model_name: Name to use in responses
            base_latency_ms: Base latency to simulate
            random_seed: Random seed for reproducibility
        """
        super().__init__(model_name)
        self.base_latency_ms = base_latency_ms
        
        if random_seed is not None:
            random.seed(random_seed)
    
    def count_tokens(self, text: str) -> int:
        """
        Approximate token count (1 token ≈ 4 characters).
        
        Args:
            text: Text to count
            
        Returns:
            Approximate token count
        """
        return len(text) // 4 + 1
    
    def _extract_true_label_from_features(
        self,
        premise: str,
        hypothesis: str
    ) -> str:
        """
        Use simple heuristics to guess a plausible label.
        This simulates what an LLM might do with the text.
        
        Args:
            premise: Premise text
            hypothesis: Hypothesis text
            
        Returns:
            A plausible label
        """
        premise_lower = premise.lower()
        hypothesis_lower = hypothesis.lower()
        
        # Calculate word overlap
        premise_words = set(re.findall(r'\b\w+\b', premise_lower))
        hypothesis_words = set(re.findall(r'\b\w+\b', hypothesis_lower))
        
        # Remove common stopwords
        stopwords = {'the', 'a', 'an', 'is', 'are', 'was', 'were', 'be', 'been', 
                     'being', 'have', 'has', 'had', 'do', 'does', 'did', 'will',
                     'would', 'could', 'should', 'may', 'might', 'must', 'shall',
                     'can', 'need', 'dare', 'ought', 'used', 'to', 'of', 'in',
                     'for', 'on', 'with', 'at', 'by', 'from', 'as', 'into',
                     'through', 'during', 'before', 'after', 'above', 'below',
                     'between', 'under', 'again', 'further', 'then', 'once',
                     'here', 'there', 'when', 'where', 'why', 'how', 'all',
                     'each', 'few', 'more', 'most', 'other', 'some', 'such',
                     'no', 'nor', 'not', 'only', 'own', 'same', 'so', 'than',
                     'too', 'very', 's', 't', 'just', 'don', 'now', 'and', 'or'}
        
        premise_content = premise_words - stopwords
        hypothesis_content = hypothesis_words - stopwords
        
        if not hypothesis_content:
            return random.choice(["entailment", "neutral", "contradiction"])
        
        overlap = len(premise_content & hypothesis_content) / len(hypothesis_content)
        
        # Check for negation patterns
        negation_in_premise = any(neg in premise_lower for neg in ['not', "n't", 'never', 'no ', 'none'])
        negation_in_hypothesis = any(neg in hypothesis_lower for neg in ['not', "n't", 'never', 'no ', 'none'])
        negation_mismatch = negation_in_premise != negation_in_hypothesis
        
        # Heuristic decision
        if negation_mismatch and overlap > 0.3:
            return "contradiction"
        elif overlap > 0.5:
            return "entailment"
        elif overlap > 0.2:
            return random.choice(["entailment", "neutral"])
        else:
            return "neutral"
    
    def _simulate_prediction(
        self,
        strategy: str,
        true_label: str,
        premise: str,
        hypothesis: str
    ) -> str:
        """
        Simulate model prediction with strategy-appropriate accuracy.
        
        Args:
            strategy: Prompting strategy
            true_label: The actual label (if known)
            premise: Premise text
            hypothesis: Hypothesis text
            
        Returns:
            Predicted label
        """
        accuracy = self.STRATEGY_ACCURACY.get(strategy, 0.70)
        
        # Determine if prediction should be correct
        if random.random() < accuracy:
            return true_label
        else:
            # Make a realistic error based on confusion biases
            other_labels = [l for l in ["entailment", "neutral", "contradiction"] if l != true_label]
            
            # Weight by confusion bias
            weights = []
            for label in other_labels:
                bias = self.CONFUSION_BIASES.get((true_label, label), 0.1)
                weights.append(bias)
            
            total = sum(weights)
            weights = [w/total for w in weights]
            
            return random.choices(other_labels, weights=weights)[0]
    
    def _generate_cot_response(self, label: str, premise: str, hypothesis: str) -> str:
        """
        Generate a Chain-of-Thought style response.
        
        Args:
            label: Final label
            premise: Premise text
            hypothesis: Hypothesis text
            
        Returns:
            CoT-style response
        """
        templates = {
            "entailment": """1. The premise states: "{premise_short}"
2. The hypothesis claims: "{hypothesis_short}"
3. Analysis: The hypothesis can be directly inferred from the information in the premise.
4. The key elements in the hypothesis are supported by the premise.

Classification: entailment""",
            
            "contradiction": """1. The premise states: "{premise_short}"
2. The hypothesis claims: "{hypothesis_short}"
3. Analysis: The hypothesis contains claims that directly contradict the premise.
4. These two statements cannot both be true simultaneously.

Classification: contradiction""",
            
            "neutral": """1. The premise states: "{premise_short}"
2. The hypothesis claims: "{hypothesis_short}"
3. Analysis: The hypothesis introduces information not mentioned in the premise.
4. The premise neither confirms nor denies the hypothesis.

Classification: neutral"""
        }
        
        # Truncate for display
        premise_short = premise[:50] + "..." if len(premise) > 50 else premise
        hypothesis_short = hypothesis[:50] + "..." if len(hypothesis) > 50 else hypothesis
        
        return templates[label].format(
            premise_short=premise_short,
            hypothesis_short=hypothesis_short
        )
    
    def generate(
        self,
        prompt: str,
        strategy: str = "zero-shot",
        max_tokens: int = 100,
        temperature: float = 0.0,
        true_label: Optional[str] = None,
        premise: Optional[str] = None,
        hypothesis: Optional[str] = None,
        **kwargs
    ) -> LLMResponse:
        """
        Generate a mock response.
        
        Args:
            prompt: Input prompt
            strategy: Prompting strategy
            max_tokens: Max tokens (ignored in mock)
            temperature: Temperature (affects randomness slightly)
            true_label: Known true label (for realistic accuracy simulation)
            premise: Original premise (for feature-based simulation)
            hypothesis: Original hypothesis (for feature-based simulation)
            
        Returns:
            LLMResponse with simulated response
        """
        start_time = time.time()
        
        # Simulate latency
        latency_variation = random.uniform(0.8, 1.5)
        strategy_multiplier = {"cot": 2.0, "few-shot": 1.3}.get(strategy, 1.0)
        actual_latency = self.base_latency_ms * latency_variation * strategy_multiplier
        time.sleep(actual_latency / 1000)  # Convert to seconds
        
        # Determine the label to predict
        if true_label is None:
            # If no true label provided, use heuristics
            if premise and hypothesis:
                true_label = self._extract_true_label_from_features(premise, hypothesis)
            else:
                true_label = random.choice(["entailment", "neutral", "contradiction"])
        
        # Simulate prediction with appropriate accuracy
        predicted_label = self._simulate_prediction(strategy, true_label, premise or "", hypothesis or "")
        
        # Generate response text
        if strategy == "cot":
            response_text = self._generate_cot_response(predicted_label, premise or "", hypothesis or "")
        else:
            response_text = predicted_label
        
        # Calculate tokens
        token_config = self.STRATEGY_TOKENS.get(strategy, {"prompt": 200, "response": 20})
        prompt_tokens = self.count_tokens(prompt)
        response_tokens = self.count_tokens(response_text)
        
        # Update stats
        self.call_count += 1
        self.total_tokens += prompt_tokens + response_tokens
        
        actual_latency_ms = (time.time() - start_time) * 1000
        
        return LLMResponse(
            text=response_text,
            prompt_tokens=prompt_tokens,
            response_tokens=response_tokens,
            model=self.model_name,
            strategy=strategy,
            latency_ms=actual_latency_ms,
            success=True
        )


if __name__ == "__main__":
    # Test the mock client
    client = MockGeminiClient(random_seed=42)
    
    test_cases = [
        {
            "premise": "The cat sat on the mat.",
            "hypothesis": "An animal was sitting.",
            "true_label": "entailment"
        },
        {
            "premise": "The store is closed on Sundays.",
            "hypothesis": "The store is open every day.",
            "true_label": "contradiction"
        },
        {
            "premise": "She went to the library to study.",
            "hypothesis": "She checked out five books.",
            "true_label": "neutral"
        }
    ]
    
    print("Mock Gemini Client Test")
    print("=" * 60)
    
    for strategy in ["zero-shot", "one-shot", "few-shot", "cot"]:
        print(f"\n{strategy.upper()}")
        print("-" * 40)
        
        correct = 0
        for case in test_cases:
            response = client.generate(
                prompt=f"Premise: {case['premise']}\nHypothesis: {case['hypothesis']}",
                strategy=strategy,
                true_label=case["true_label"],
                premise=case["premise"],
                hypothesis=case["hypothesis"]
            )
            
            predicted = response.get_label()
            is_correct = predicted == case["true_label"]
            correct += is_correct
            
            print(f"  True: {case['true_label']:15} Pred: {predicted:15} {'✓' if is_correct else '✗'}")
        
        print(f"  Accuracy: {correct}/{len(test_cases)}")
    
    print(f"\nTotal stats: {client.get_stats()}")
