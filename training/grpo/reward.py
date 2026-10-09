"""Anti-bias reward for GRPO with verl.

    reward_i = 1[predicted letter == ground truth] - 0.1 * KL(P_i || Uniform(A, B, C, D))

P_i is the empirical distribution of predicted letters over the 64 most recent scored rollouts,
including rollout i itself (with 16 prompts x 8 rollouts per step the window spans half a step).
The penalty is computed per rollout, so rollouts of the same prompt receive different values and the
term is not removed by GRPO's group mean-centering. Rollouts without a parsable letter receive no
correctness reward and are not added to the window.

verl entry point: compute_score(data_source, solution_str, ground_truth, extra_info) -> float
(set custom_reward_function.path=<this file> custom_reward_function.name=compute_score).
"""
import math
import re
from collections import Counter

LETTER_RE = re.compile(r"\b([A-D])\b")
LAMBDA_ANTIBIAS = 0.1
WINDOW = 64

# Letters predicted so far in this process, in scoring order.
_PREDICTIONS: list[str] = []


def _extract_letter(s):
    if not isinstance(s, str):
        return None
    m = LETTER_RE.search(s.upper().strip())
    return m.group(1) if m else None


def _kl_uniform(letters):
    """KL(empirical letter distribution || uniform over A-D)."""
    if not letters:
        return 0.0
    n = len(letters)
    counts = Counter(letters)
    kl = 0.0
    for letter in "ABCD":
        p = counts.get(letter, 0) / n
        if p > 0:
            kl += p * math.log(p / 0.25)
    return kl


def compute_score(data_source, solution_str, ground_truth, extra_info=None):
    pred = _extract_letter(solution_str)
    correctness = 1.0 if pred == ground_truth else 0.0
    if pred:
        _PREDICTIONS.append(pred)
    kl = _kl_uniform(_PREDICTIONS[-WINDOW:])
    return correctness - LAMBDA_ANTIBIAS * kl


if __name__ == "__main__":
    scores = [compute_score(None, p, g) for p, g in [("A", "A"), ("A", "B"), ("B", "B"), ("A", "C"), ("A", "D"), ("A", "A")]]
    print("rewards:", [round(s, 4) for s in scores])
