"""MBPP tests pass only when the assertion actually runs."""

import pytest

from rllm.environments.mbpp.mbpp import run_code_with_tests

TESTS = ["assert add(1, 2) == 3", "assert add(2, 2) == 4"]


def grade(code, setup_code=""):
    return run_code_with_tests(code, TESTS, timeout=10, parallel=False, setup_code=setup_code)


def test_correct_code_passes():
    result = grade("def add(a, b):\n    return a + b")
    assert result["all_passed"] and result["passed_tests"] == 2


def test_wrong_code_fails_with_the_assertion_error():
    result = grade("def add(a, b):\n    return a - b")
    assert result["passed_tests"] == 0
    assert "AssertionError" in result["test_results"][0]["error"]


def test_setup_code_runs_before_the_assertion():
    result = grade("def add(a, b):\n    return a + b + OFFSET", setup_code="OFFSET = 0")
    assert result["all_passed"]


@pytest.mark.parametrize(
    "escape",
    [
        "import sys\nsys.exit(0)",
        "import sys\nsys.exit()",
        "import os\nos._exit(0)",
        "raise SystemExit(0)",
        "exit()",
        "if __name__ == '__main__':\n    import sys\n    sys.exit(0)",
    ],
)
def test_exiting_before_the_assertion_fails(escape):
    result = grade(f"def add(a, b):\n    return 0\n\n{escape}")
    assert result["passed_tests"] == 0
    assert "exited before the test assertion" in result["test_results"][0]["error"]


def test_printed_output_does_not_fake_a_pass():
    result = grade("def add(a, b):\n    return 0\n\nprint('RLLM_MBPP_TEST_PASSED_' + '0' * 32)\nimport sys\nsys.exit(0)")
    assert result["passed_tests"] == 0
