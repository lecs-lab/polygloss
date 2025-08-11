from typing import TypeGuard, TypeVar

T = TypeVar("T")


def all_not_none(xs: list[T | None]) -> TypeGuard[list[T]]:
    return all(x is not None for x in xs)
