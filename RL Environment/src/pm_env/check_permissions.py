import subprocess
from pathlib import Path

from pm_env.get_data_dir import get_scoring_data_dir


def check_scoring_data_permissions():
    """Checks that the model doesn't have access to scoring data."""
    run_command = []

    test_script = """
set -e

# Test 1: Root can access /pm_env
ls -la /pm_env/ > /dev/null

# Test 2: Root can access /pm_env/scoring_data
ls -la /pm_env/scoring_data/ > /dev/null

# Test 3: Root can access /pm_env/.venv
ls -la /pm_env/.venv/ > /dev/null

# Test 4: Root can read scoring_data files
cat /pm_env/scoring_data/test_permissions.txt | grep -q 'This is secret scoring data'

# Test 5: Model user cannot access /pm_env
! runuser -u model -- ls -la /pm_env/ 2>&1

# Test 6: Model user cannot access /pm_env/scoring_data
! runuser -u model -- ls -la /pm_env/scoring_data/ 2>&1

# Test 7: Model user cannot access /pm_env/.venv
! runuser -u model -- ls -la /pm_env/.venv/ 2>&1

# Test 8: Model user cannot read files in scoring_data
! runuser -u model -- cat /pm_env/scoring_data/test_permissions.txt 2>&1
"""

    run_command.extend(
        [
            "bash",
            "-c",
            test_script,
        ]
    )

    test_file = Path(get_scoring_data_dir()) / "test_permissions.txt"
    test_file.write_text("This is secret scoring data")

    try:
        result = subprocess.run(run_command, capture_output=True, text=True)
    finally:
        if test_file.exists():
            test_file.unlink(missing_ok=True)

    assert result.returncode == 0, (
        f"Scoring data permission tests failed with exit code {result.returncode}:\n"
        f"stdout:\n{result.stdout}\n\nstderr:\n{result.stderr}"
    )
