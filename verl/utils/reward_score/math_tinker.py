import math
import re

from verl.utils.reward_score.tinker_utils import (
    extract_boxed,
    grade_answer,
    grade_answer_math_verify,
    run_with_timeout_signal,
)

import logging

logger = logging.getLogger(__name__)

def compute_score(
    solution_str: str,
    ground_truth: str,
    grader: str = "sympy",
    timeout: float = 1.0,
    return_dict: bool = True,
    min_reward: float = 0.0,
    max_reward: float = 1.0,
):
    # print(f"safe_grade: {solution_str}, {ground_truth}, {grader}, {timeout}, {return_dict}, {min_reward}, {max_reward}")
    try:
        given_answer = extract_boxed(solution_str)
    except ValueError:
        if return_dict:
            return {
                "score": min_reward,
                "acc": False,
                "pred": "[NO_BOXED_FOUND]",
            }
        else:
            return min_reward
    
    if grader == "sympy":
        grader_func = grade_answer
    elif grader == "math_verify":
        grader_func = grade_answer_math_verify
    else:
        raise ValueError(f"Invalid grader: {grader}")
    out = run_with_timeout_signal(
        grader_func, args=(given_answer, ground_truth), timeout_seconds=int(math.ceil(timeout))
    )
    if out is None:
        logger.warning(f"Timeout grading {given_answer} against {ground_truth}")
        if return_dict:
            return {
                "score": min_reward,
                "acc": False,
                "pred": "[TIMEOUT]",
            }
        else:
            return min_reward

    assert isinstance(out, bool)
    if return_dict:
        return {
            "score": max_reward if out else min_reward,
            "acc": out,
            "pred": given_answer,
        }
    else:
        return max_reward if out else min_reward

if __name__ == "__main__":
    from verl.utils.reward_score.test_math_verifiers import set_verifier, run_all_tests
    
    print("Testing tinker verifier")
    print("="*50)
    set_verifier("tinker", grader="sympy")
    run_all_tests()
