

"""
1. load the MATH validation examples
2. format them as string prompts to the language model using the r1_zero prompt
3. generate outputs for each example
4. calculate evaluation metrics
5. serialize the examples, model generations, and corresponding evaluation scores to disk for
analysis in subsequent problems
"""

from cs336_alignment.src.evaluate.llm import LLMEvaluationResult, EvaluationResultStore, evaluate_vllm
from cs336_alignment.drgrpo_grader import r1_zero_reward_fn
from vllm import LLM, SamplingParams
from typing import List, Callable
import json


class RawMathExample:
    def __init__(self, question: str, answer: str):
        self.question = question
        self.answer = answer

    def from_dict(data: dict) -> 'RawMathExample':
        return RawMathExample(
            question=data['question'],
            answer=data['answer']
        )

class FormattedMathExample:
    def __init__(self, prompt: str, raw_example: RawMathExample):
        self.prompt = prompt
        self.raw_example = raw_example
        self.answer = raw_example.answer

    @property
    def formatted_question(self) -> str:
        return self.prompt.replace("{question}", self.raw_example.question)
    


def load_gsm8k_math_validation_examples() -> List[dict]:
    filepath = "/home/wren/learning/online-course/stanford-cs336/assignments/assignment5-alignment/data/gsm8k/test.jsonl"
    with open(filepath, 'r', encoding='utf-8') as f:
        examples = [json.loads(line) for line in f]
    return examples


def format_prompt(example: dict, input_prompt: str) -> FormattedMathExample:
    raw_example = RawMathExample.from_dict(example)
    formatted_example = FormattedMathExample(
        prompt=input_prompt,
        raw_example=raw_example
    )
    return formatted_example


def r1_zero_prompt() -> str:
    file_path = "/home/wren/learning/online-course/stanford-cs336/assignments/assignment5-alignment/cs336_alignment/prompts/r1_zero.prompt"
    with open(file_path, 'r', encoding='utf-8') as f:
        return f.read()

def main() -> None:
    # Load raw math examples
    raw_examples = load_gsm8k_math_validation_examples()
    input_prompt = r1_zero_prompt()

    # Format examples
    formatted_examples = [
        format_prompt(example, input_prompt)
        for example in raw_examples
    ]

    prompts = [ex.formatted_question for ex in formatted_examples]

    # Create a sampling params object, stopping generation on newline.
    eval_sampling_params = SamplingParams(
        temperature=0.0, top_p=1.0, max_tokens=1024, stop=["\n"]
    )

    # Create an LLM.
    local_model_path = "/home/wren/.cache/huggingface/hub/models--Qwen--Qwen2.5-Math-1.5B/snapshots/4a83ca6e4526a4f2da3aa259ec36c259f66b2ab2"
    vllm_model = LLM(model=local_model_path)

    # Evaluate the model and save results
    evaluate_vllm(
        vllm_model=vllm_model,
        reward_fn=r1_zero_reward_fn,
        prompts=prompts,
        ground_truths=[ex.answer for ex in formatted_examples],
        eval_sampling_params=eval_sampling_params
    )


if __name__ == "__main__":
    main()