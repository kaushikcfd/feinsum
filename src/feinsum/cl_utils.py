from typing import Protocol, Sequence, Tuple, Union
from dataclasses import dataclass


class DeviceT(Protocol):
    @property
    def name(self) -> str:
        pass


class ContextT(Protocol):
    @property
    def devices(self) -> Sequence[DeviceT]:
        pass


# {{{ make_fake_cl_context

@dataclass(frozen=True, repr=True, eq=True)
class FakeDevice:
    name: str


@dataclass(frozen=True, repr=True, eq=True)
class FakeContext:
    devices: Tuple[FakeDevice, ...]


def make_fake_cl_context(devices: Union[str, Sequence[str]]) -> FakeContext:
    if isinstance(devices, str):
        devices = devices,

    return FakeContext(tuple(FakeDevice(dev_name)
                             for dev_name in devices))

# }}}
