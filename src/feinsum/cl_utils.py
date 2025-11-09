"""
.. autoclass:: DeviceT
"""

from dataclasses import dataclass
from typing import Protocol


class DeviceT(Protocol):
    """
    Abstract type providing an interface like :class:`pyopencl.Device`.
    """

    @property
    def name(self) -> str:
        pass


@dataclass(frozen=True, repr=True, eq=True)
class FakeCLDevice:
    name: str
