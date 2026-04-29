"""
Test module for different math verification methods.
Tests tinker_verify, math_dapo, and math_verify compute_score functions.
"""

# Global variable to store the current compute_score function
compute_score = None

# Colors for terminal output
class Colors:
    GREEN = '\033[92m'
    RED = '\033[91m'
    YELLOW = '\033[93m'
    BLUE = '\033[94m'
    BOLD = '\033[1m'
    END = '\033[0m'

def print_success(message):
    """Print success message in green."""
    print(f"{Colors.GREEN}✓ {message}{Colors.END}")

def print_failure(message):
    """Print failure message in red."""
    print(f"{Colors.RED}✗ {message}{Colors.END}")

def print_info(message):
    """Print info message in blue."""
    print(f"{Colors.BLUE}ℹ {message}{Colors.END}")

def print_section(message):
    """Print section header in bold."""
    print(f"{Colors.BOLD}{message}{Colors.END}")


def set_verifier(verifier_type: str, **kwargs):
    """Set the verifier type and load the corresponding compute_score function.
    
    Args:
        verifier_type: One of "tinker", "dapo", or "math_verify"
    """
    global compute_score
    
    if verifier_type == "tinker":
        from verl.utils.reward_score import math_tinker
        # Capture the grader from set_verifier kwargs
        verifier_grader = kwargs.get("grader")
        def tinker_wrapper(solution_str, ground_truth, return_dict=True, min_reward=-1.0, max_reward=1.0, **kwargs):
            return math_tinker.compute_score(
                solution_str=solution_str,
                ground_truth=ground_truth,
                grader=verifier_grader,
                return_dict=return_dict,
                min_reward=min_reward,
                max_reward=max_reward,
            )
        compute_score = tinker_wrapper
        print_info(f"Using tinker verifier")
    elif verifier_type == "dapo":
        from verl.utils.reward_score import math_dapo
        def dapo_wrapper(solution_str, ground_truth, return_dict=True, min_reward=-1.0, max_reward=1.0, **kwargs):
            result = math_dapo.compute_score(solution_str=solution_str, ground_truth=ground_truth, strict_box_verify=True)
            # Normalize the rewards to match min/max
            if isinstance(result, dict):
                if result["score"] == 1.0:
                    result["score"] = max_reward
                elif result["score"] == -1.0:
                    result["score"] = min_reward
                return result
            else:
                return max_reward if result == 1.0 else min_reward
        compute_score = dapo_wrapper
        print_info(f"Using math_dapo verifier")
    elif verifier_type == "math_verify":
        from verl.utils.reward_score import math_verify
        def math_verify_wrapper(solution_str, ground_truth, return_dict=True, min_reward=-1.0, max_reward=1.0, **kwargs):
            return math_verify.compute_score(
                solution_str=solution_str,
                ground_truth=ground_truth,
                return_dict=return_dict,
                min_reward=min_reward,
                max_reward=max_reward
            )
        compute_score = math_verify_wrapper
        print_info(f"Using math_verify")
    else:
        raise ValueError(f"Unknown verifier type: {verifier_type}")


def test_missing_boxed():
    """Test case where boxed answer is completely missing."""
    test_name = "Missing Boxed Answer"
    print_info(f"Running test: {test_name}")
    
    ground_truth = "1"
    solution_str = "indicating exactly one real root.\n\nSo finally, we will determine the analytical solution using Python and simplify our investigation. Let's complete the code and have the results! \\boxed{}"
    solution_str = solution_str.replace("\\boxed{}", "")
    results = compute_score(
        solution_str=solution_str,
        ground_truth=ground_truth,
        return_dict=True,
        min_reward=-1.0,
        max_reward=1.0,
    )
    print(f"Results: {results}")
    
    try:
        assert results["score"] == -1.0, f"Expected score -1.0, got {results['score']}"
        assert results["acc"] == False, f"Expected acc False, got {results['acc']}"
        assert results["pred"] == "[NO_BOXED_FOUND]", f"Expected pred '[NO_BOXED_FOUND]', got {results['pred']}"
        print_success(f"{test_name} PASSED")
    except AssertionError as e:
        print_failure(f"{test_name} FAILED: {e}")
        raise
    print("-"*100)


def test_empty_boxed():
    """Test case where boxed answer is empty."""
    test_name = "Empty Boxed Answer"
    print_info(f"Running test: {test_name}")
    
    ground_truth = "1"
    solution_str = "indicating exactly one real root.\n\nSo finally, we will determine the analytical solution using Python and simplify our investigation. Let's complete the code and have the results! \\boxed{}"
    results = compute_score(
        solution_str=solution_str,
        ground_truth=ground_truth,
        return_dict=True,
        min_reward=-1.0,
        max_reward=1.0,
    )
    print(f"Results: {results}")
    possible_preds = ["[VERIFICATION_FAILED]", ""]
    
    try:
        assert results["score"] == -1.0, f"Expected score -1.0, got {results['score']}"
        assert results["acc"] == False, f"Expected acc False, got {results['acc']}"
        assert results["pred"] in possible_preds, f"Expected pred in {possible_preds}, got {results['pred']}"
        print_success(f"{test_name} PASSED")
    except AssertionError as e:
        print_failure(f"{test_name} FAILED: {e}")
        raise
    print("-"*100)


def test_many_boxed():
    """Test case with many malformed boxed expressions."""
    test_name = "Many Malformed Boxed"
    print_info(f"Running test: {test_name}")
    
    ground_truth = "1"
    solution_str = "\\boxed{\\dfrac{15}{4}$$', Pred: ['xed{ \\ \\ \\boxed{ livestock \\boxed \\boxed{\n\\boxed \\ \\ \\boxed \\ \\boxed{ careful \\boxed \\boxed{\\ iced \\boxed\\x \\boxed \\ \\boxed \\boxed{\\boxed{\\boxed \\boxed \\boxed{\\boxed \\boxed{\\ \\boxed \\boxed \\boxed{\\boxed\\[.\n\n\\boxed{\\boxed \\boxed{\\boxed{ \\boxed \\boxed{boxed \\boxed \\boxed{boxed \\boxed{x boxed.\n\nboxed{."
    results = compute_score(
        solution_str=solution_str,
        ground_truth=ground_truth,
        return_dict=True,
        min_reward=-1.0,
        max_reward=1.0,
    )
    print(f"Results: {results}")
    possible_preds = ["[VERIFICATION_FAILED]", "[NO_BOXED_FOUND]"]
    
    try:
        assert results["score"] == -1.0, f"Expected score -1.0, got {results['score']}"
        assert results["acc"] == False, f"Expected acc False, got {results['acc']}"
        assert results["pred"] in possible_preds, f"Expected pred in {possible_preds}, got {results['pred']}"
        print_success(f"{test_name} PASSED")
    except AssertionError as e:
        print_failure(f"{test_name} FAILED: {e}")
        raise
    print("-"*100)


def test_simple_correct():
    """Test case with simple correct answer."""
    test_name = "Simple Correct Answer"
    print_info(f"Running test: {test_name}")
    
    ground_truth = "1"
    solution_str = "\\boxed{1}"
    results = compute_score(
        solution_str=solution_str,
        ground_truth=ground_truth,
        return_dict=True,
        min_reward=-1.0,
        max_reward=1.0,
    )
    print(f"Results: {results}")
    
    try:
        assert results["score"] == 1.0, f"Expected score 1.0, got {results['score']}"
        assert results["acc"] == True, f"Expected acc True, got {results['acc']}"
        assert results["pred"] == ground_truth, f"Expected pred '{ground_truth}', got {results['pred']}"
        print_success(f"{test_name} PASSED")
    except AssertionError as e:
        print_failure(f"{test_name} FAILED: {e}")
        raise
    print("-"*100)


def test_latex_parse_error():
    """Test case with LaTeX parsing error."""
    test_name = "LaTeX Parse Error"
    print_info(f"Running test: {test_name}")
    
    ground_truth = "1"
    solution_str = "\\boxed{\\frac1}1}}"
    results = compute_score(
        solution_str=solution_str,
        ground_truth=ground_truth,
        return_dict=True,
        min_reward=-1.0,
        max_reward=1.0,
    )
    print(f"Results: {results}")
    
    try:
        assert results["score"] == -1.0, f"Expected score -1.0, got {results['score']}"
        assert results["acc"] == False, f"Expected acc False, got {results['acc']}"
        print_success(f"{test_name} PASSED")
    except AssertionError as e:
        print_failure(f"{test_name} FAILED: {e}")
        raise
    print("-"*100)


def test_latex_parse_pass():
    """Test case with correct LaTeX that should pass."""
    test_name = "LaTeX Parse Success"
    print_info(f"Running test: {test_name}")
    
    ground_truth = "{1,3} \\cup {2,4}"
    solution_str = f"\\boxed{{1,2,3,4}}"
    results = compute_score(
        solution_str=solution_str,
        ground_truth=ground_truth,
        return_dict=True,
        min_reward=-1.0,
        max_reward=1.0,
    )
    print(f"Results: {results}")
    
    try:
        assert results["score"] == 1.0, f"Expected score 1.0, got {results['score']}"
        assert results["acc"] == True, f"Expected acc True, got {results['acc']}"
        print_success(f"{test_name} PASSED")
    except AssertionError as e:
        print_failure(f"{test_name} FAILED: {e}")
        raise
    print("-"*100)


def test_no_gold():
    """Test case with malformed ground truth."""
    test_name = "Malformed Ground Truth"
    print_info(f"Running test: {test_name}")
    
    ground_truth = '\\left( \\frac{a}{b'
    solution_str = 'dS = \\boxed{-4\\pi}.\n\\]'
    results = compute_score(
        solution_str=solution_str,
        ground_truth=ground_truth,
        return_dict=True,
        min_reward=-1.0,
        max_reward=1.0,
    )
    print(f"Results: {results}")
    print_info(f"{test_name} - No specific assertions (exploratory test)")
    print("-"*100)


def run_all_tests():
    """Run all test functions."""
    test_functions = [
        test_missing_boxed,
        test_empty_boxed,
        test_many_boxed,
        test_simple_correct,
        test_latex_parse_error,
        test_latex_parse_pass,
        test_no_gold,
    ]
    
    passed_tests = []
    failed_tests = []
    
    print_section(f"Running {len(test_functions)} tests...")
    print("="*100)
    
    for test_func in test_functions:
        try:
            test_func()
            passed_tests.append(test_func.__name__)
        except Exception as e:
            failed_tests.append((test_func.__name__, str(e)))
            print_failure(f"Test {test_func.__name__} FAILED with error: {e}")
    
    # Print summary
    print("="*100)
    print_section("TEST SUMMARY")
    print(f"Total tests: {len(test_functions)}")
    print_success(f"Passed: {len(passed_tests)}")
    print_failure(f"Failed: {len(failed_tests)}")
    
    if passed_tests:
        print(f"\\n{Colors.GREEN}✓ Passed tests:{Colors.END}")
        for test in passed_tests:
            print(f"  - {test}")
    
    if failed_tests:
        print(f"\\n{Colors.RED}✗ Failed tests:{Colors.END}")
        for test, error in failed_tests:
            print(f"  - {test}: {error}")
    
    return len(passed_tests), len(failed_tests)


def test_all_verifiers():
    """Test all three verifier types."""
    verifier_types = ["tinker", "dapo", "math_verify"]
    overall_results = {}
    
    print_section("=" * 80)
    print_section("STARTING COMPREHENSIVE VERIFIER TESTING")
    print_section("=" * 80)
    
    for verifier_type in verifier_types:
        print(f"\n{'=' * 60}")
        print_section(f"TESTING {verifier_type.upper()} VERIFIER")
        print(f"{'=' * 60}")
        
        try:
            set_verifier(verifier_type)
            passed, failed = run_all_tests()
            overall_results[verifier_type] = {"passed": passed, "failed": failed}
            
            if failed == 0:
                print_success(f"\n🎉 {verifier_type} verifier: ALL TESTS PASSED!")
            else:
                print_failure(f"\n❌ {verifier_type} verifier: {failed} test(s) failed")
                
        except Exception as e:
            print_failure(f"❌ Error testing {verifier_type} verifier: {e}")
            overall_results[verifier_type] = {"passed": 0, "failed": "ERROR", "error": str(e)}
    
    # Final summary
    print("\n" + "=" * 80)
    print_section("FINAL SUMMARY - ALL VERIFIERS")
    print("=" * 80)
    
    total_passed = 0
    total_failed = 0
    
    for verifier_type, results in overall_results.items():
        if results["failed"] == "ERROR":
            print_failure(f"{verifier_type:12}: ERROR - {results.get('error', 'Unknown error')}")
        else:
            passed = results["passed"]
            failed = results["failed"]
            total_passed += passed
            total_failed += failed
            
            status_symbol = "✅" if failed == 0 else "❌"
            color_func = print_success if failed == 0 else print_failure
            print(f"{status_symbol} {verifier_type:12}: {passed} passed, {failed} failed")
    
    print("\n" + "-" * 40)
    print_section(f"GRAND TOTAL: {total_passed} passed, {total_failed} failed")
    
    if total_failed == 0:
        print_success("🎉 ALL VERIFIERS PASSED ALL TESTS! 🎉")
    else:
        print_failure(f"⚠️  {total_failed} total test failures across all verifiers")
    
    print("=" * 80)


if __name__ == "__main__":
    test_all_verifiers()