# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


########################################################
# https://github.com/huggingface/Math-Verify
########################################################

## Parser definition
import logging
from typing import Callable, Optional, Sequence

from math_verify.grader import verify
from math_verify.parser import ExprExtractionConfig, ExtractionTarget, parse
from math_verify.utils import timeout
from sympy.functions import elliptic_f

logger = logging.getLogger(__name__)
logger.setLevel(logging.CRITICAL)  # Suppress warnings from this logger


def math_metric(
    gold_extraction_target: Sequence[ExtractionTarget] = (ExprExtractionConfig(),),
    pred_extraction_target: Sequence[ExtractionTarget] = (ExprExtractionConfig(),),
    aggregation_function: Callable[[list[float]], float] = max,
    precision: int = 6,
) -> Callable[
    [list[str], list[str]], tuple[float, Optional[tuple[list[str], list[str]]]]
]:
    """Creates a language-aware extractive match metric that extracts answers from the model's output.

    Known issues:
    - If the task is to simplify an expression, the metric might overestimate the accuracy. This is because if the model doesn't output any anchor for the extraction (e.g final answer is..),
        it's possible that the the extracted prediction will be the expression to simplify. Because we do simplifications ourselves, it can thus happen that sympy will correctly simplify the expression,
        thus it will match gold, despite model not doing anything. PRs to fix this are welcome.

    Args:
        language: Language
            The language of the samples.
        gold_extraction_target: Sequence[ExtractionTarget]
            Extraction targets to use for gold answers. Defaults to extracting simple math expressions.
        pred_extraction_target: Sequence[ExtractionTarget]
            Extraction targets to use for predictions. Defaults to extracting simple math expressions.
        aggregation_function: Callable[[list[float]], float]
            Function to aggregate scores when multiple golds/predictions are present. Defaults to max.
        fallback_mode: Literal["no_fallback", "first_match"]
            How to perform extraction. Defaults to "first_match".
            - "no_fallback": Only use first successfully parsed matches
            - "first_match": Use the first successfully parsed match + first match irregardless the parsing success
        precision: int
            Number of decimal places to use when comparing numerical values. Defaults to 6.

    Returns:
        A sample level metric that extracts and compares mathematical expressions.

    """

    @timeout(10)
    def get_str_preds_with_timeout(
        extracted_predictions: list[list[str]], extracted_golds: list[list[str]]
    ) -> tuple[list[str], list[str]]:
        golds = [str(gold) for golds in extracted_golds for gold in golds]
        predictions = [str(pred) for preds in extracted_predictions for pred in preds]
        return (golds, predictions)

    def sample_level_fn(
        golds: list[str], predictions: list[str]
    ) -> tuple[float, Optional[tuple[list[str], list[str]]]]:
        extracted_predictions = [
            parse(pred, pred_extraction_target) for pred in predictions
        ]
        extracted_golds = [parse(gold, gold_extraction_target) for gold in golds]

        # Assert on empty gold and warn on empty pred
        if any(len(g) == 0 for g in extracted_golds):
            raise ValueError(
                f"No gold targets found for at least one gold. Gold: {golds}, Pred: {predictions}"
            )

        if all(len(p) == 0 for p in extracted_predictions):
            logger.warning(
                f"We did not manage to extract a prediction in the correct format. Gold: {golds}, Pred: {predictions}"
            )

        # We have to use timeout because the sypmy to str conversion can be very slow
        str_preds = None
        try:
            str_preds = get_str_preds_with_timeout(
                extracted_predictions, extracted_golds
            )
        except Exception:
            logger.warning(
                "Timeout when adding extracted predictions and golds to specific"
            )

        return (
            aggregation_function(
                [
                    (
                        1.0
                        if any(
                            verify(gold, pred, precision) for gold in extracted_golds
                        )
                        else 0.0
                    )
                    for pred in extracted_predictions
                ]
            ),
            str_preds,
        )

    return sample_level_fn



########################################################
import logging

logging.getLogger("math_verify").setLevel(logging.CRITICAL)
# Suppress warnings from this module's logger
logging.getLogger("verl.utils.reward_score.math_verify").setLevel(logging.CRITICAL)

try:
    # from math_verify.errors import TimeoutException
    # from math_verify.metric import math_metric
    from math_verify.parser import ExprExtractionConfig, LatexExtractionConfig
except ImportError:
    print("To use Math-Verify, please install it first by running `pip install math-verify`.")


def compute_score(
        solution_str: str,
        ground_truth: str,
        return_dict: bool = False,
        truncate_before_verify: bool = True,
        min_reward: float = 0.0,
        max_reward: float = 1.0
    ) -> float:
    """
    Compute the score for math verification.
    
    Args:
        solution_str: The model's output string
        ground_truth: The ground truth answer
        return_dict: Whether to return a dictionary (default: False)
        truncate_before_verify: Whether to truncate the solution string before verification (default: True)
        min_reward: Minimum reward value (default: 0)
        max_reward: Maximum reward value (default: 1)
    
    Returns:
        float: The computed score
        or dict of score, accuracy, and prediction representation
    """
    # Limit solution length for efficiency
    if truncate_before_verify:
        solution_str = solution_str[-300:]  # The longest answer in MATH-500 has 159 characters

    verify_func = math_metric(
        gold_extraction_target=(LatexExtractionConfig(),),  # Single extractor for gold: LaTeX format
        pred_extraction_target=(ExprExtractionConfig(), LatexExtractionConfig()),  # Two extractors for predictions: both expression and LaTeX formats
    )
    ret_score = 0.0

    # If there's no \boxed{} in the output, return 0 (empty string case)
    if "\\boxed{" not in solution_str:
        # logging.warning(f"No \\boxed{{}} found in model output.")
        # print(model_output)
        if return_dict:
            return {
                "score": min_reward,
                "acc": False,
                "pred":"[NO_BOXED_FOUND]"
            }
        else:
            return min_reward

    # Wrap the ground truth in \boxed{} format for verification
    ground_truth_boxed = "\\boxed{" + ground_truth + "}"
    
    # Call verification function - returns (score, results) tuple
    # results format: (gold_representations, prediction_representations) or None if timeout
    # Each list can have multiple items due to multiple extraction targets/representations
    try:
        ret_score, results = verify_func([ground_truth_boxed], [solution_str])
        # print("results from verify_func: ", results)
        # print('ret_score from verify_func: ', ret_score)
        pred = results[1][0]
        reward = min_reward if ret_score == 0 else max_reward
        acc = reward == max_reward
    except Exception as e:
        # print("error from verify_func: ", e)
        pred = "[VERIFICATION_FAILED]"
        reward = min_reward
        acc = False
    
    # # Handle timeout (results=None) or empty predictions (results[1] has no valid extractions)
    # if results is None:
    #     pred = "[TIMEOUT]"
    #     reward = min_reward
    #     acc = False
    # elif len(results[1]) == 0:
    #     pred = "[VERIFICATION_FAILED]"
    #     reward = min_reward
    #     acc = False
    # else:
    #     # Take the first prediction representation from results[1]
    #     # results[1] may contain multiple representations like ['{1, 2, 3, 4}', '1,2,3,4']
    #     # due to multiple extraction targets (ExprExtractionConfig + LatexExtractionConfig)
    #     pred = results[1][0]
    #     reward = min_reward if ret_score == 0 else max_reward
    #     acc = reward == max_reward

    if return_dict:
        return {
            "score": reward,
            "acc": acc,
            "pred": pred,
        }
    else:
        return reward


def test_complex_expression_timeout():
    """Test timeout with computationally expensive expression using actual timeout."""
    from math_verify.utils import timeout
    
    # Create a simple version with 0.01 second timeout
    @timeout(0.01)
    def get_str_preds_with_timeout_fast(extracted_predictions, extracted_golds):
        golds = [str(gold) for golds in extracted_golds for gold in golds]
        predictions = [str(pred) for preds in extracted_predictions for pred in preds]
        return (golds, predictions)

    def sample_level_fn_fast(golds, predictions):
        from math_verify.parser import parse, ExprExtractionConfig, LatexExtractionConfig
        from math_verify.grader import verify
        
        extracted_predictions = [
            parse(pred, (ExprExtractionConfig(), LatexExtractionConfig())) for pred in predictions
        ]
        extracted_golds = [parse(gold, (LatexExtractionConfig(),)) for gold in golds]

        if any(len(g) == 0 for g in extracted_golds):
            raise ValueError(
                f"No gold targets found for at least one gold. Gold: {golds}, Pred: {predictions}"
            )

        str_preds = None
        try:
            str_preds = get_str_preds_with_timeout_fast(
                extracted_predictions, extracted_golds
            )
        except Exception as e:
            print(f"Timeout or error in get_str_preds_with_timeout: {e}")
            # Return None to indicate timeout/failure
            str_preds = None

        return (
            max([
                (
                    1.0
                    if any(
                        verify(gold, pred, 6) for gold in extracted_golds
                    )
                    else 0.0
                )
                for pred in extracted_predictions
            ]),
            str_preds,
        )
    
    ground_truth = "1"
    complex_expr = f"\\frac{1}{2} + \\frac{1}{4} + \\frac{1}{8} + \\frac{1}{16} + \\frac{1}{32} + \\frac{1}{64} + \\frac{1}{128} + \\frac{1}{256} + \\frac{1}{512} + \\frac{1}{1024} + \\frac{1}{2048} + \\frac{1}{4096} + \\frac{1}{8192} + \\frac{1}{16384} + \\frac{1}{32768} + \\frac{1}{65536}"
    solution_str = f"\\boxed{{{complex_expr}}}"
    
    print(f"Testing with complex expression: {solution_str[:100]}...")
    
    # Test with our fast timeout function directly
    ground_truth_boxed = "\\boxed{" + ground_truth + "}"
    try:
        ret_score, results = sample_level_fn_fast([ground_truth_boxed], [solution_str])
        print("results from verify_func: ", results)
        print('ret_score from verify_func: ', ret_score)
        if results is None:
            pred = "[VERIFICATION_FAILED]"
            reward = -1.0
            acc = False
        else:
            pred = results[1][0] if len(results[1]) > 0 else "[VERIFICATION_FAILED]"
            reward = -1.0 if ret_score == 0 else 1.0
            acc = reward == 1.0
    except Exception as e:
        print("error from verify_func: ", e)
        pred = "[VERIFICATION_FAILED]"
        reward = -1.0
        acc = False
    
    test_results = {
        "score": reward,
        "acc": acc,
        "pred": pred,
    }
    
    print(f"Complex expression timeout test results: {test_results}")
    assert test_results["score"] == -1.0, f"complex timeout error, score too high: {test_results}"
    assert test_results["acc"] == False, f"complex timeout error, invalid acc: {test_results}"
    assert test_results["pred"] == "[VERIFICATION_FAILED]", f"complex timeout error, missing pred: {test_results}"
    
    print("complex expression timeout test passed")
    print("-"*100)


if __name__ == "__main__":
    from verl.utils.reward_score.test_math_verifiers import set_verifier, run_all_tests
    
    print("Testing math_verify verifier")
    print("="*50)
    set_verifier("math_verify")
    run_all_tests()
    
    # verifier-specific tests
    test_complex_expression_timeout()