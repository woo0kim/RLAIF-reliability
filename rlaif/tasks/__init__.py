from .base import Problem, Task
from .countdown import CountdownTask
from .math_task import MathTask


def get_task(name: str, **kw) -> Task:
    if name == "countdown":
        return CountdownTask()
    if name == "math":
        return MathTask(**kw)
    raise ValueError(f"unknown task {name!r}")


__all__ = ["Problem", "Task", "CountdownTask", "MathTask", "get_task"]
