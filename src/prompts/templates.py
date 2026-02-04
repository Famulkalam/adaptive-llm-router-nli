"""
Prompt Templates for NLI Classification

Contains all prompt templates for zero-shot, one-shot, few-shot, and 
Chain-of-Thought (CoT) prompting strategies.
"""

from dataclasses import dataclass
from typing import List, Dict, Optional


@dataclass
class NLIExample:
    """An example for few-shot prompting."""
    premise: str
    hypothesis: str
    label: str
    reasoning: Optional[str] = None  # For CoT examples


class PromptTemplates:
    """Collection of prompt templates for NLI classification."""
    
    # System instruction (shared across all prompts)
    SYSTEM_INSTRUCTION = """You are an expert in Natural Language Inference (NLI). Your task is to classify the logical relationship between a premise and a hypothesis.

The possible classifications are:
- **entailment**: The hypothesis logically follows from the premise. If the premise is true, the hypothesis must also be true.
- **contradiction**: The hypothesis contradicts the premise. The premise and hypothesis cannot both be true.
- **neutral**: The hypothesis is neither entailed by nor contradicts the premise. The premise does not provide enough information to determine the truth of the hypothesis.

You must respond with exactly one word: entailment, contradiction, or neutral."""

    # Zero-shot template
    ZERO_SHOT_TEMPLATE = """{system_instruction}

Premise: {premise}
Hypothesis: {hypothesis}

Classification:"""

    # One-shot template
    ONE_SHOT_TEMPLATE = """{system_instruction}

Here is an example:

Premise: The cat sat on the mat in the living room.
Hypothesis: An animal is in a room inside a house.
Classification: entailment

Now classify this pair:

Premise: {premise}
Hypothesis: {hypothesis}

Classification:"""

    # Few-shot template header
    FEW_SHOT_TEMPLATE_HEADER = """{system_instruction}

Here are some examples:

"""

    # Few-shot example format
    FEW_SHOT_EXAMPLE_FORMAT = """Premise: {premise}
Hypothesis: {hypothesis}
Classification: {label}

"""

    # Few-shot template footer
    FEW_SHOT_TEMPLATE_FOOTER = """Now classify this pair:

Premise: {premise}
Hypothesis: {hypothesis}

Classification:"""

    # Chain-of-Thought template
    COT_TEMPLATE = """You are an expert in Natural Language Inference (NLI). Your task is to classify the logical relationship between a premise and a hypothesis.

The possible classifications are:
- **entailment**: The hypothesis logically follows from the premise.
- **contradiction**: The hypothesis contradicts the premise.
- **neutral**: The hypothesis is neither entailed by nor contradicts the premise.

Please think step-by-step:
1. Identify the key claims in the premise
2. Identify the key claims in the hypothesis
3. Compare the claims and check for logical relationships
4. Determine if the hypothesis is entailed, contradicted, or neutral

Premise: {premise}
Hypothesis: {hypothesis}

Let me analyze this step by step:"""

    # CoT with example
    COT_WITH_EXAMPLE_TEMPLATE = """You are an expert in Natural Language Inference (NLI). Your task is to classify the logical relationship between a premise and a hypothesis.

The possible classifications are:
- **entailment**: The hypothesis logically follows from the premise.
- **contradiction**: The hypothesis contradicts the premise.  
- **neutral**: The hypothesis is neither entailed by nor contradicts the premise.

Here is an example of step-by-step reasoning:

Premise: The company reported record profits in the third quarter, exceeding analyst expectations.
Hypothesis: The company lost money in the third quarter.

Step-by-step analysis:
1. Key claims in premise: The company had "record profits" and "exceeded expectations" in Q3.
2. Key claims in hypothesis: The company "lost money" in Q3.
3. Comparison: "Record profits" directly contradicts "lost money" - these cannot both be true.
4. Conclusion: The hypothesis contradicts the premise.

Classification: contradiction

---

Now analyze this pair:

Premise: {premise}
Hypothesis: {hypothesis}

Let me analyze this step by step:"""

    # Default few-shot examples (diverse and covering all labels)
    DEFAULT_FEW_SHOT_EXAMPLES = [
        NLIExample(
            premise="The children were playing soccer in the park on a sunny afternoon.",
            hypothesis="Some young people were engaged in outdoor physical activity.",
            label="entailment"
        ),
        NLIExample(
            premise="The restaurant was completely empty when we arrived at noon.",
            hypothesis="The restaurant was crowded with customers at lunchtime.",
            label="contradiction"
        ),
        NLIExample(
            premise="The scientist published her research findings in a prestigious journal.",
            hypothesis="The scientist won an award for her work.",
            label="neutral"
        ),
        NLIExample(
            premise="All employees must complete the mandatory training by Friday.",
            hypothesis="There is a training deadline for workers.",
            label="entailment"
        ),
        NLIExample(
            premise="The museum is closed on Mondays for maintenance.",
            hypothesis="The museum welcomes visitors every day of the week.",
            label="contradiction"
        ),
        NLIExample(
            premise="The new smartphone features a high-resolution camera and 5G connectivity.",
            hypothesis="The smartphone is the best-selling model this year.",
            label="neutral"
        )
    ]

    # Hybrid 6-Shot + CoT Template
    SIX_SHOT_COT_TEMPLATE = """{system_instruction}

Here are some examples of step-by-step reasoning:

---
Premise: The cat sat on the mat.
Hypothesis: The cat is on the mat.
Analysis: 
1. Premise says cat sat on mat. 
2. Hypothesis says cat is on mat. 
3. Sitting on mat implies being on the mat.
Classification: entailment

---
Premise: I have no money.
Hypothesis: I am a millionaire.
Analysis:
1. Premise says no money.
2. Hypothesis says millionaire.
3. Having no money directly contradicts being a millionaire.
Classification: contradiction

---
Now analyze this pair step-by-step:

Premise: {premise}
Hypothesis: {hypothesis}

Let me analyze this step by step:"""
    
    @classmethod
    def get_zero_shot_prompt(cls, premise: str, hypothesis: str) -> str:
        """Generate a zero-shot prompt."""
        return cls.ZERO_SHOT_TEMPLATE.format(
            system_instruction=cls.SYSTEM_INSTRUCTION,
            premise=premise,
            hypothesis=hypothesis
        )
    
    @classmethod
    def get_one_shot_prompt(cls, premise: str, hypothesis: str) -> str:
        """Generate a one-shot prompt."""
        return cls.ONE_SHOT_TEMPLATE.format(
            system_instruction=cls.SYSTEM_INSTRUCTION,
            premise=premise,
            hypothesis=hypothesis
        )
    
    @classmethod
    def get_few_shot_prompt(
        cls,
        premise: str,
        hypothesis: str,
        examples: Optional[List[NLIExample]] = None,
        num_examples: int = 5
    ) -> str:
        """Generate a few-shot prompt with examples."""
        if examples is None:
            examples = cls.DEFAULT_FEW_SHOT_EXAMPLES[:num_examples]
        
        prompt = cls.FEW_SHOT_TEMPLATE_HEADER.format(
            system_instruction=cls.SYSTEM_INSTRUCTION
        )
        
        for example in examples:
            prompt += cls.FEW_SHOT_EXAMPLE_FORMAT.format(
                premise=example.premise,
                hypothesis=example.hypothesis,
                label=example.label
            )
        
        prompt += cls.FEW_SHOT_TEMPLATE_FOOTER.format(
            premise=premise,
            hypothesis=hypothesis
        )
        
        return prompt
    
    @classmethod
    def get_cot_prompt(
        cls,
        premise: str,
        hypothesis: str,
        with_example: bool = True
    ) -> str:
        """Generate a Chain-of-Thought prompt."""
        if with_example:
            return cls.COT_WITH_EXAMPLE_TEMPLATE.format(
                premise=premise,
                hypothesis=hypothesis
            )
        else:
            return cls.COT_TEMPLATE.format(
                premise=premise,
                hypothesis=hypothesis
            )
    
    @classmethod
    def get_prompt(
        cls,
        strategy: str,
        premise: str,
        hypothesis: str,
        **kwargs
    ) -> str:
        """
        Get prompt for any strategy.
        
        Args:
            strategy: One of 'zero-shot', 'one-shot', 'few-shot', 'cot'
            premise: Premise sentence
            hypothesis: Hypothesis sentence
            **kwargs: Additional arguments for specific strategies
            
        Returns:
            Formatted prompt string
        """
        strategy = strategy.lower().replace("_", "-")
        
        if strategy == "zero-shot":
            return cls.get_zero_shot_prompt(premise, hypothesis)
        elif strategy == "one-shot":
            return cls.get_one_shot_prompt(premise, hypothesis)
        elif strategy == "few-shot":
            return cls.get_few_shot_prompt(premise, hypothesis, **kwargs)
        elif strategy == "6-shot":
            return cls.get_few_shot_prompt(premise, hypothesis, num_examples=6)
        elif strategy == "6-shot-cot":
            return cls.SIX_SHOT_COT_TEMPLATE.format(
                system_instruction=cls.SYSTEM_INSTRUCTION,
                premise=premise,
                hypothesis=hypothesis
            )
        elif strategy == "cot" or strategy == "chain-of-thought":
            return cls.get_cot_prompt(premise, hypothesis, **kwargs)
        else:
            raise ValueError(f"Unknown strategy: {strategy}")


if __name__ == "__main__":
    # Demo all prompt types
    premise = "The government announced new environmental regulations yesterday."
    hypothesis = "New rules about the environment were introduced."
    
    print("=" * 60)
    print("ZERO-SHOT PROMPT")
    print("=" * 60)
    print(PromptTemplates.get_zero_shot_prompt(premise, hypothesis))
    
    print("\n" + "=" * 60)
    print("ONE-SHOT PROMPT")
    print("=" * 60)
    print(PromptTemplates.get_one_shot_prompt(premise, hypothesis))
    
    print("\n" + "=" * 60)
    print("FEW-SHOT PROMPT (3 examples)")
    print("=" * 60)
    print(PromptTemplates.get_few_shot_prompt(premise, hypothesis, num_examples=3))
    
    print("\n" + "=" * 60)
    print("CHAIN-OF-THOUGHT PROMPT")
    print("=" * 60)
    print(PromptTemplates.get_cot_prompt(premise, hypothesis))
