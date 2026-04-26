from typing import Any, Callable, Dict


class InvokableTool:
    """Small compatibility wrapper that mimics `.invoke({...})`."""

    def __init__(self, func: Callable[..., Any]) -> None:
        self._func = func
        self.__doc__ = getattr(func, "__doc__", None)
        self.__name__ = getattr(func, "__name__", self.__class__.__name__)

    def __call__(self, *args, **kwargs) -> Any:
        return self._func(*args, **kwargs)

    def invoke(self, payload: Any = None) -> Any:
        if payload is None:
            return self._func()
        if isinstance(payload, dict):
            return self._func(**payload)
        return self._func(payload)
