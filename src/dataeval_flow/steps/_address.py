"""Addresses: how a step names a chain input, an earlier step's output, or one element of a list."""

__all__ = ["STEP_NAME_PATTERN", "Address", "parse_address"]

import re
from dataclasses import dataclass

STEP_NAME_PATTERN = r"^[A-Za-z_][A-Za-z0-9_-]*$"
"""What a step's or an input slot's name may be: `.`, `[` and `/` mean something in addresses and result keys."""

_NAME = r"[A-Za-z_][A-Za-z0-9_-]*"
_ADDRESS = re.compile(rf"^(?P<name>{_NAME})(?:\.(?P<output>{_NAME}))?(?:\[(?P<key>[^\[\]\s]+)\])?$")


@dataclass(frozen=True)
class Address:
    """A chain input or step (`name`), one of its outputs (`output`), and one element of a list (`key`)."""

    name: str
    output: str | None = None
    key: str | None = None

    def __str__(self) -> str:
        text = self.name if self.output is None else f"{self.name}.{self.output}"
        return text if self.key is None else f"{text}[{self.key}]"

    @property
    def base(self) -> "Address":
        """This address without its element key."""
        return Address(self.name, self.output)


def parse_address(text: str) -> Address:
    """Read `text` as an address: ``name``, ``name.output``, ``name[key]`` or ``name.output[key]``.

    Raises
    ------
    ValueError
        When `text` is none of those forms.
    """
    match = _ADDRESS.match(text) if isinstance(text, str) else None
    if match is None:
        raise ValueError(
            f"{text!r} is not an address: name a chain input or an earlier step, optionally followed by "
            "`.output` and `[key]`, such as `clean`, `split.train` or `kfold.train[0]`."
        )
    return Address(match["name"], match["output"], match["key"])
