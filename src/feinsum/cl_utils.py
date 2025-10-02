"""
.. autoclass:: ContextT

.. autoclass:: DeviceT
"""

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Protocol


class DeviceT(Protocol):
    """
    Abstract type providing an interface like :class:`pyopencl.Device`.
    """

    @property
    def name(self) -> str:
        pass


class ContextT(Protocol):
    """
    Abstract type providing an interface like :class:`pyopencl.Context`.
    """

    @property
    def devices(self) -> Sequence[DeviceT]:
        pass


# {{{ make_fake_cl_context


@dataclass(frozen=True, repr=True, eq=True)
class FakeDevice:
    name: str


@dataclass(frozen=True, repr=True, eq=True)
class FakeContext:
    devices: tuple[FakeDevice, ...]


def make_fake_cl_context(devices: str | Sequence[str]) -> FakeContext:
    if isinstance(devices, str):
        devices = (devices,)

    return FakeContext(tuple(FakeDevice(dev_name) for dev_name in devices))


# }}}
