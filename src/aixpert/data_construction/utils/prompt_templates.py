"""Central location for prompt templates (evaluation, DPO scoring, etc.)."""

from __future__ import annotations


BINARY_FACTUAL_JUDGE_PROMPT = """
System:
You are an extremely strict factuality judge. Your task is to determine whether the assistant’s
response contains ANY of the following:

• factual errors
• hallucinations
• unsupported claims
• unverifiable statements
• fabricated details
• incorrect technical descriptions
• invented studies, names, or entities
• logical contradictions
• vague or misleading explanations
• partial answers that omit critical facts

IMPORTANT:
If the response is vague, incomplete, unsupported, or unverifiable,
classify it as [[1]].

Definitions:
[[0]] → Fully factual, precise, verifiable.
[[1]] → ANY issue.

Return ONLY:
[[0]] or [[1]]

User Question:
{question}

Assistant Response:
{answer}
"""

CORRUPTION_SYSTEM_PROMPT = """
You are a factual corruption generator.

Rewrite a correct answer into a subtly incorrect version with 2–3 factual errors.
It must remain fluent, confident, and plausible.
"""

CORRUPTION_USER_PROMPT = """
PROMPT:
{question}

CORRECT ANSWER:
{answer}

TASK:
Rewrite the answer so that it becomes factually wrong, introducing subtle hallucinations
while sounding confident and coherent.
"""
