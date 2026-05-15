import json
import sys
from pathlib import Path

from pm_env.get_data_dir import get_env_data_dir

if __name__ == "__main__":
    output_path = sys.argv[1]

    answer = (Path(get_env_data_dir()) / "python_version.txt").read_text().strip()

    score = 1.0 if "3.12.11" in answer else 0.0

    Path(output_path).write_text(json.dumps({"score": score, "metadata": {}}))
