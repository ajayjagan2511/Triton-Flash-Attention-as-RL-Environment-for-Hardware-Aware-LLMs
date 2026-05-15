import os
from pathlib import Path


def get_env_data_dir() -> str:
    if "PM_CONTAINERIZED" in os.environ:
        return Path("/workdir").as_posix()
    else:
        return (Path(__file__).parent.parent.parent / "env_data").absolute().as_posix()


def get_scoring_data_dir() -> str:
    if "PM_CONTAINERIZED" in os.environ:
        return Path("/pm_env/scoring_data").as_posix()
    else:
        return (
            (Path(__file__).parent.parent.parent / "scoring_data").absolute().as_posix()
        )
