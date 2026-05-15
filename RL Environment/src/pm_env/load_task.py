from pm_env.schemas.evaluation_run_config import EvaluationRunConfig
from pm_env.task import Task
from pm_env.tasks import get_tasks


def load_task(config: EvaluationRunConfig) -> Task:
    tasks = get_tasks(config)

    for task in tasks:
        if task.id == config.task_id:
            return task

    raise ValueError(
        f"Task {config.task_id!r} not found. List existing tasks with `pm_env list-tasks`."
    )
