import json
import os
import select
import subprocess
import sys
import tempfile
from typing import Final, override

from pm_env.get_data_dir import get_env_data_dir
from pm_env.judges.judge import Judge
from pm_env.schemas.scoring import Scoring
from pm_env.schemas.transcript import Transcript


class ExecutableJudge(Judge):
    """
    Runs an arbitrary executable from the model's working
    directory.

    Example use cases:
    1. Ask student to train a RL agent on CartPolev1 and save the agent to
    to 'agent.pt2'. Then you can use ExecutableJudge with a custom testing
    script to load this agent, run it on the CartPole environment, and
    evaluate performance.
    2. Ask model to curate a subset of a dataset for improved training to
    target a specific capability (i.e. subset CommonCrawl for math capabilities).
    Ask the model to save the dataset subset to a local file. Test by running
    your own optimized training script on this dataset subset, and then
    evaling however you see fit. (This way the model can 'submit' a large
    amount of data without using a lot of tokens to do so).

    Check out the 'executable-judge-example' in the example environments repo for
    example usage.

    Note this judge does not look at the model transcript at all.
    """

    def __init__(
        self,
        subprocess_run_args: list[str],
        continue_threshold: float = -1,
    ) -> None:
        """
        subprocess_run_args should contain all args for your executable
        (including the executable itself)
        i.e., if your testing executable is a python script saved to scoring_dir/:

        subprocess_run_args =
        [
          sys.executable,
          f'{get_scoring_data_dir()}/my_testing_script.py',
          'my_arg1',
          'executable_output_file.json',
        ]
        IMPORTANT NOTE 1:
        Note the usage of 'sys.executable' above instead of 'python' - this ensures
        your script is with the correct python interpreter (in the correct virtual
        environment).

        IMPORTANT NOTE 2:
        Your executable should always take a filepath (executable_output_file.json) as its final
        argument, and then save the testing results to that file in the following format:
        { 'score' : <score>,
          'metadata': <metadata dictionary> }
        ExecutableJudge will in fact modify this path to point to a file
        within a temporary directory, so that it is deleted after judging is finished.

        Note that the values of the metadata dictionary must support .__str__()

        The task is continued if the score is greater than or equal to continue_threshold.
        """
        self.subprocess_run_args: Final = subprocess_run_args
        self.continue_threshold: Final = continue_threshold

    @override
    def evaluate(self, transcript: Transcript) -> Scoring:
        with tempfile.TemporaryDirectory() as temp_dir:
            self.subprocess_run_args[-1] = (
                f"{temp_dir}/{os.path.basename(self.subprocess_run_args[-1])}"
            )

            try:
                process = subprocess.Popen(
                    self.subprocess_run_args,
                    cwd=get_env_data_dir(),
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    text=True,
                )
                assert process.stdout is not None
                assert process.stderr is not None
                stdout_lines = []
                stderr_lines = []

                while True:
                    reads = [process.stdout.fileno(), process.stderr.fileno()]
                    ret = select.select(reads, [], [])

                    for fd in ret[0]:
                        if fd == process.stdout.fileno():
                            read = process.stdout.readline()
                            if read:
                                sys.stdout.write(read)
                                sys.stdout.flush()
                                stdout_lines.append(read)
                        if fd == process.stderr.fileno():
                            read = process.stderr.readline()
                            if read:
                                sys.stderr.write(read)
                                sys.stderr.flush()
                                stderr_lines.append(read)

                    if process.poll() is not None:
                        # Process finished, read any remaining output
                        for line in process.stdout:
                            stdout_lines.append(line)
                            sys.stdout.write(line)
                            sys.stdout.flush()
                        for line in process.stderr:
                            stderr_lines.append(line)
                            sys.stderr.write(line)
                            sys.stderr.flush()
                        break

                return_code = process.wait()
                stdout = "".join(stdout_lines)
                stderr = "".join(stderr_lines)
                if return_code != 0:
                    raise subprocess.CalledProcessError(
                        return_code, self.subprocess_run_args, stdout, stderr
                    )
            except subprocess.CalledProcessError as e:
                score = 0
                metadata = {
                    "CalledProcessError": str(e),
                    "stderr": e.stderr if hasattr(e, "stderr") else "",
                    "stdout": e.stdout if hasattr(e, "stdout") else "",
                }
                continue_task = False
                return Scoring(score, metadata, continue_task)
            except FileNotFoundError as e:
                score = 0
                metadata = {"FileNotFoundError": str(e)}
                continue_task = False
                return Scoring(score, metadata, continue_task)

            try:
                with open(self.subprocess_run_args[-1]) as f:
                    results = json.load(f)
                    continue_task = results["score"] >= self.continue_threshold
                    metadata = {k: str(v) for k, v in results["metadata"].items()}
                    metadata["stdout"] = stdout
                    metadata["stderr"] = stderr
                    return Scoring(
                        results["score"],
                        metadata,
                        continue_task,
                    )
            except (json.JSONDecodeError, UnicodeDecodeError, OSError, KeyError) as e:
                score = 0
                metadata = {"json_read_error": str(e)}
                continue_task = False
                return Scoring(score, metadata, continue_task)
