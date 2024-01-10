"""Lazy properties.
"""

from typing import Generic, TypeVar, overload

T = TypeVar('T')


class InitOnAccess(Generic[T]):
    """Descriptor class that initializes an attribute on first access.
    """
    def __init__(self, klass: T, *args, **kwargs):
        self.klass = klass
        self.args = args
        self.kwargs = kwargs
        self._initialized: T | None = None

    def __repr__(self) -> str:
        return f'Lazy({self.klass}, {self.args}, {self.kwargs})'

    @overload
    def __get__(self, instance: None, owner: None) -> T:
        pass

    @overload
    def __get__(self, instance: object, owner: None) -> T:
        pass

    def __get__(self, instance, owner) -> T:
        if self._initialized is None:
            self._initialized = self.klass(*self.args, **self.kwargs)
        return self._initialized
