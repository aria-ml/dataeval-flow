"""Multi-file YAML/JSON configuration loader with schema validation."""

import json
import logging
import re
from pathlib import Path
from typing import Any

import yaml

from dataeval_flow._logging import LogMessage

_logger: logging.Logger = logging.getLogger(__name__)

_YAML_EXTS = frozenset({".yaml", ".yml"})
_JSON_EXTS = frozenset({".json"})
_CONFIG_EXTS = _YAML_EXTS | _JSON_EXTS

# What an address's name and output look like just before a `[`, and its key from that `[` on. `parse_address` then
# decides whether the two make an address.
_BEFORE_KEY = re.compile(r"(?:^|[\s{\[,])(?P<base>[^\s{}\[\],:'\"]+)$")
_KEY = re.compile(r"^\[[^\[\]\s]+\]")


class AddressQuotingError(yaml.MarkedYAMLError):
    """A pipeline file that fails to parse at an address with a key left unquoted inside YAML's ``{…}`` or ``[…]``."""


def load_yaml(path: Path) -> Any:
    """Parse the YAML file at `path`, saying to quote a keyed address where one breaks YAML's flow style.

    Inside ``{…}`` or ``[…]``, YAML reads a ``[`` as the start of a list, so ``{input: kfold.train[0]}`` fails with
    PyYAML's "expected ',' or '}', but got '['". Where the parse stops at such an address, the error is raised again
    as an :class:`AddressQuotingError`, with PyYAML's message and one sentence more. Any other error is raised as it
    is.
    """
    with open(path, encoding="utf-8") as f:
        try:
            return yaml.safe_load(f)
        except yaml.MarkedYAMLError as error:
            address = _unquoted_address(error, path.read_text(encoding="utf-8").splitlines())
            if address is None:
                raise
            hint = (
                f"An address with a key, such as `{address}`, must be quoted inside YAML's `{{…}}` or `[…]`: "
                f'write "{address}".'
            )
            note = hint if error.note is None else f"{error.note}\n{hint}"
            raise AddressQuotingError(
                error.context, error.context_mark, error.problem, error.problem_mark, note
            ) from error


def _unquoted_address(error: yaml.MarkedYAMLError, lines: list[str]) -> str | None:
    """The keyed address a parse inside ``{…}`` or ``[…]`` stopped at the ``[`` of; ``None`` for any other error.

    Decided from the source and the error's marks, not from PyYAML's wording: the collection the parser was inside
    starts at the context mark, and the parse stopped at the problem mark.
    """
    from dataeval_flow.steps._address import parse_address

    opened, stopped = error.context_mark, error.problem_mark
    if opened is None or stopped is None or _character(lines, opened) not in ("{", "["):
        return None
    if _character(lines, stopped) != "[":
        return None
    line = lines[stopped.line]
    before, key = _BEFORE_KEY.search(line[: stopped.column]), _KEY.match(line[stopped.column :])
    if before is None or key is None:
        return None
    text = before["base"] + key.group(0)
    try:
        parse_address(text)
    except ValueError:
        return None
    return text


def _character(lines: list[str], mark: Any) -> str | None:
    """The character of the source at `mark`, or ``None`` past its end."""
    if mark.line >= len(lines) or mark.column >= len(lines[mark.line]):
        return None
    return lines[mark.line][mark.column]


def _load_file(path: Path) -> dict[str, Any] | list[str]:
    """Load a single YAML or JSON file and return its contents as a dict."""
    if path.suffix.lower() in _JSON_EXTS:
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    return load_yaml(path) or []


def _is_config_fragment(data: dict[str, Any]) -> bool:
    """Whether *data* is a pipeline config fragment: it holds at least one top-level section.

    A file holding none, such as a JSON schema or a compose file, is unrelated and skipped. A fragment's other keys
    must all be sections too, which :func:`merge_config_folder` checks, so a misspelled section is refused rather than
    dropping its whole file.
    """
    from dataeval_flow.config._models import top_level_keys

    return not top_level_keys().isdisjoint(data)


def merge_config_folder(config_path: Path) -> dict[str, Any]:
    """Scan a folder and merge every pipeline config file in it, in name order.

    Files are loaded in sorted order (00-base.yaml before 01-datasets.yaml). Later files override earlier ones for
    duplicate keys: mappings merge, lists extend, and scalars replace.

    Skipped, with a debug log line: a file that does not parse as YAML or JSON, and a file holding no top-level
    section, such as a JSON schema or unrelated YAML.

    Refused:
    - a pipeline file whose parse stops at a keyed address left unquoted inside YAML's ``{…}`` or ``[…]``, since it is
      a pipeline file with a fixable typo, not an unrelated one;
    - a file holding a section beside a key that is none, which is most likely a misspelled section.

    Returns the raw dict; use ``load_config()`` for a validated ``PipelineConfig``.

    Raises
    ------
    AddressQuotingError
        When a file's parse stops at a keyed address to quote; the message names the address.
    ValueError
        When `config_path` is not a directory, or a file holds a key that is no section, naming the file and the key.
    FileNotFoundError
        When the folder holds no pipeline config file.
    """
    from dataeval_flow.config._models import unknown_keys_problem

    config: dict[str, Any] = {}

    if not config_path.is_dir():
        raise ValueError(f"Config path is not a directory: {config_path}")

    candidates = sorted(f for f in config_path.iterdir() if f.is_file() and f.suffix.lower() in _CONFIG_EXTS)
    _logger.debug(LogMessage(lambda: f"Found {len(candidates)} candidate file(s): {[f.name for f in candidates]}"))

    accepted: list[Path] = []
    for config_file in candidates:
        try:
            file_config = _load_file(config_file)
        except AddressQuotingError:
            raise  # a pipeline file with an address to quote, not an unrelated file: say so rather than skip it
        except (json.JSONDecodeError, yaml.YAMLError) as exc:
            _logger.debug("Skipping %s (parse error: %s)", config_file.name, exc)
            continue

        if not isinstance(file_config, dict) or not _is_config_fragment(file_config):
            _logger.debug("Skipping %s (not a pipeline config)", config_file.name)
            continue
        problem = unknown_keys_problem(file_config)
        if problem is not None:
            raise ValueError(f"{config_file.name}: {problem}")

        _deep_merge(config, file_config)
        accepted.append(config_file)

    if not accepted:
        raise FileNotFoundError(f"No valid pipeline config files found in {config_path}")

    _logger.debug(LogMessage(lambda: f"Accepted {len(accepted)} config file(s): {[f.name for f in accepted]}"))
    return config


def _deep_merge(base: dict, overlay: dict) -> None:
    """Recursively merge overlay into base.

    Rules: dicts merge recursively, lists extend, scalars replace.
    """
    for key, value in overlay.items():
        if key in base and isinstance(base[key], dict) and isinstance(value, dict):
            _deep_merge(base[key], value)
        elif key in base and isinstance(base[key], list) and isinstance(value, list):
            base[key].extend(value)
        else:
            base[key] = value
