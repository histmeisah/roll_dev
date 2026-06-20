import logging
import multiprocessing
import random
import re
from typing import Optional, Tuple, Any, SupportsFloat, Dict

from datasets import load_dataset, Dataset, DatasetDict
from gem import Env
from gem.envs.math_env import MathEnv as GEMMathEnv
from gem.utils.constants import TERMINAL_STATE
from gem.utils.parsing import extract_last_boxed_answer
import ray

from roll.datasets.global_dataset import GlobalDataset, GlobalDatasetManager
from roll.utils.constants import RAY_NAMESPACE

logger = logging.getLogger(__name__)


_UNBOXED_FINAL_ANSWER_PATTERNS = [
    re.compile(r"(?:final\s+answer|answer)\s*(?:is|=|:)?\s*\\?\(?\s*(-?\d{1,4})\s*\\?\)?", re.IGNORECASE),
    re.compile(r"(?:therefore|thus|so)[^.\n]{0,80}?\b(-?\d{1,4})\b", re.IGNORECASE),
]


def extract_unboxed_integer_answer(text: str) -> Optional[str]:
    """Best-effort diagnostic extraction for AIME-style unboxed answers."""
    if not text:
        return None

    text = re.sub(r"<\|.*?\|>", " ", text)
    for pattern in _UNBOXED_FINAL_ANSWER_PATTERNS:
        matches = pattern.findall(text)
        if matches:
            return matches[-1]

    # Last-resort fallback for logging only. Reward/eval still require boxed.
    matches = re.findall(r"(?<![\w.])-?\d{1,4}(?![\w.])", text)
    return matches[-1] if matches else None


class MathEnv(GEMMathEnv):

    def __init__(
            self,
            dataset_name: Optional[str] = "",
            split: Optional[str] = None,
            dataset: Optional[Dataset] = None,
            question_key: str = "problem",
            answer_key: str = "answer",
            seed: int = 0,
            mode: str = "train",
            format_penalty: float = 0.0,
            **_,
    ):
        Env.__init__(self)
        self.seed = seed
        self.question_key = question_key
        self.answer_key = answer_key
        self.mode = mode
        self.format_penalty = float(format_penalty)

        # Convert train/val mode to sample/traversal for GlobalDataset
        global_dataset_mode = "sample" if self.mode == "train" else "traversal"
        self.dataset = GlobalDataset.options(name=f"{self.mode}_{dataset_name}",
                                             get_if_exists=True,
                                             namespace=RAY_NAMESPACE).remote(dataset_name=dataset_name,
                                                                             split=split,
                                                                             mode=global_dataset_mode)
        self.dataset_manager = GlobalDatasetManager.options(name=f"{self.mode}_dataset_manager",
                                                            get_if_exists=True,
                                                            namespace=RAY_NAMESPACE).remote()
        ray.get(self.dataset_manager.register.remote(dataset_name=dataset_name, dataset_ref=self.dataset))
        self.idx = 0
        self.epoch = 0
        # Process pool is used to enable the timeout mechanism for answer grading in a potential distributed training setup
        self.mp_pool = multiprocessing.Pool(1)

    def reset(self, seed: Optional[None] = None) -> Tuple[str, dict[str, Any]]:
        """Sample a question from the dataset."""
        Env.reset(self, seed)
        data: Optional[Dict] = ray.get(self.dataset.get_data_item.remote(seed=seed))
        if data is None:
            return None, None
        self.first_obs = data[self.question_key]
        self.answer = data[self.answer_key]
        self.idx += 1
        return self.first_obs, {"env_instruction": ""}

    def step(
        self, action: str
    ) -> Tuple[str, SupportsFloat, bool, bool, dict[str, Any]]:
        model_answer = extract_last_boxed_answer(action)
        unboxed_answer = extract_unboxed_integer_answer(action) if model_answer is None else None
        answer_source = "boxed" if model_answer is not None else "none"
        action_is_valid = model_answer is not None
        raw_reward = 0.0
        format_penalty = 0.0

        if model_answer is None:
            reward = self.format_penalty
            format_penalty = self.format_penalty
        else:
            res = self.mp_pool.apply_async(
                self.check_correct, (model_answer, self.answer)
            )
            try:
                is_correct = res.get(timeout=1)
            except (multiprocessing.context.TimeoutError, Exception):
                is_correct = False
            raw_reward = 1.0 if is_correct else 0.0
            reward = raw_reward

        metrics = {
            "action_is_valid": action_is_valid,
            "success": raw_reward > 0,
            "boxed_success": raw_reward > 0,
            "answer_is_parseable": model_answer is not None,
            "unboxed_answer_present": unboxed_answer is not None,
            "raw_reward": raw_reward,
            "format_penalty": format_penalty,
        }
        metrics_agg_mode = {
            "action_is_valid": "mean",
            "success": "last",
            "boxed_success": "last",
            "answer_is_parseable": "mean",
            "unboxed_answer_present": "mean",
            "raw_reward": "last",
            "format_penalty": "mean",
        }
        info = {
            "metrics": metrics,
            "metrics_agg_mode": metrics_agg_mode,
            "model_answer": model_answer,
            "gold_answer": self.answer,
            "answer_source": answer_source,
            "unboxed_answer": unboxed_answer,
        }
        return TERMINAL_STATE, reward, True, True, info
