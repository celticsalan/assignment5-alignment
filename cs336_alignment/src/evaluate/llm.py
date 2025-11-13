from vllm import LLM, SamplingParams, RequestOutput
from typing import Callable, List
import json
from typing import List



class LLMEvaluationResult:
    def __init__(self, prompt: str, generation: str, metrics: dict[str, float]):
        self.prompt = prompt
        self.generation = generation
        self.metrics = metrics

    def __repr__(self) -> str:
        return f"LLMEvaluationResult(prompt={self.prompt!r}, generation={self.generation!r}, metrics={self.metrics!r})"
    
    def from_dict(data: dict) -> 'LLMEvaluationResult':
        return LLMEvaluationResult(
            prompt=data['prompt'],
            generation=data['generation'],
            metrics=data['metrics']
        )
    
    def to_dict(self) -> dict:
        return {
            'prompt': self.prompt,
            'generation': self.generation,
            'metrics': self.metrics
        }

class EvaluationResultStore:
    def __init__(self, filepath: str):
        self.filepath = filepath

    def save(self, results: List[LLMEvaluationResult]) -> None:
        # Serialize results as JSON lines
        with open(self.filepath, 'w', encoding='utf-8') as f:
            for result in results:
                json.dump(result.to_dict(), f, ensure_ascii=False)
                f.write('\n')

    def load(self) -> List[LLMEvaluationResult]:
        # Load results from JSON lines
        results: List[LLMEvaluationResult] = []
        with open(self.filepath, 'r', encoding='utf-8') as f:
            for line in f:
                data = json.loads(line)
                results.append(LLMEvaluationResult.from_dict(data))
        return results

def evaluate_vllm(
        vllm_model: LLM,
        reward_fn: Callable[[str, str], dict[str, float]],
        prompts: List[str],
        ground_truths: List[str],
        eval_sampling_params: SamplingParams
    ) -> None:
    """
    Evaluate a language model on a list of prompts,
    compute evaluation metrics, and serialize results to disk.
    """

    outputs: List[RequestOutput] = vllm_model.generate(prompts, eval_sampling_params)

    results = []
    for output, ground_truth in zip(outputs, ground_truths):
        prompt = output.prompt
        output_text = output.outputs[0].text if output and output.outputs else ""
        rewards = reward_fn(output_text, ground_truth)
        # Compute and log evaluation metrics here
        metrics = {
            "reward": rewards.get("reward", 0),
            "length": len(output_text),
        }
        result = LLMEvaluationResult(prompt=prompt, generation=output_text, metrics=metrics)
        results.append(result)

    # Save all results to disk
    saver = EvaluationResultStore("outputs/results.txt")
    saver.save(results)
