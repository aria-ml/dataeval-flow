#!/usr/bin/env python
"""Write the Config Reference, one MyST page per top-level key, from config/params.schema.json.

conf.py runs it on every docs build, so the pages track the checked-in schema; they are not committed.

Usage:
  python docs/build_config_reference.py
"""

import json
import re
import textwrap
from pathlib import Path
from typing import Any

REPO = Path(__file__).resolve().parents[1]
SCHEMA = REPO / "config" / "params.schema.json"
OUT = REPO / "docs" / "source" / "reference" / "config"

INTRO = """\
Every key a pipeline config file takes, generated from the JSON Schema `config/params.schema.json`. Each top-level
key has its own page, listing every entry it takes, the keys of each entry, their types, defaults and constraints.
A key marked required must be set; every other key may be left out.

An editor with a YAML language server, such as VS Code with the Red Hat YAML extension, validates and completes a
config file against the same schema when the file's first line names it:

```yaml
# yaml-language-server: $schema=<path to>/params.schema.json
```

For what each evaluator, step and preset does, see the catalogs: [Evaluator Catalog](../evaluators.md),
[Transform Catalog](../transforms.md), [Combine Catalog](../combines.md), [Check Catalog](../checks.md) and
[Preset Catalog](../presets.md)."""

Node = dict[str, Any]


def _label(name: str) -> str:
    return f"config-{name.lower()}"


def _ref(node: Node) -> str:
    return node["$ref"].rsplit("/", 1)[-1]


def _refs(node: Any) -> list[str]:
    """Every definition `node` names, in the order it names them."""
    if isinstance(node, dict):
        return ([_ref(node)] if "$ref" in node else []) + [r for v in node.values() for r in _refs(v)]
    if isinstance(node, list):
        return [r for v in node for r in _refs(v)]
    return []


def _variants(node: Node) -> list[Node]:
    """A union's members, without the `null` pydantic adds for an optional key."""
    members = node.get("oneOf") or node.get("anyOf") or [node]
    return [m for m in members if m.get("type") != "null"]


class Reference:
    """The schema, and the page each of its definitions is written on."""

    def __init__(self, schema: Node) -> None:
        self.schema = schema
        self.defs: dict[str, Node] = schema["$defs"]
        # A key's own entries go on its page, even where an earlier key embeds one (a preset's `diversity:`); what
        # they reach goes on the first page that reaches it.
        entries = {key: list(dict.fromkeys(_refs(prop))) for key, prop in schema["properties"].items()}
        placed = {name for names in entries.values() for name in names}
        self.pages: dict[str, list[str]] = {
            key: [n for name in names for n in (name, *self._walk(self.defs[name], placed))]
            for key, names in entries.items()
            if names
        }

    def _walk(self, node: Node, placed: set[str]) -> list[str]:
        """The definitions `node` reaches that no earlier page holds, each followed by those it reaches."""
        names = []
        for name in _refs(node):
            if name not in placed:
                placed.add(name)
                names += [name, *self._walk(self.defs[name], placed)]
        return names

    def tag(self, name: str) -> tuple[str, Any] | None:
        """The key and value that pick this kind of entry out of a list (`type: outliers`), if it has one."""
        consts = [
            (key, prop["const"]) for key, prop in self.defs[name].get("properties", {}).items() if "const" in prop
        ]
        return consts[-1] if consts else None  # a wrap step also states its wrapper; its own key comes last

    def title(self, name: str) -> str:
        tag = self.tag(name)
        return f"{tag[0]}: {tag[1]}" if tag else name

    def link(self, name: str, text: str | None = None) -> str:
        return f"{{ref}}`{text or self.title(name)} <{_label(name)}>`"

    def type_of(self, node: Node) -> str:
        """`node`'s type in words, its definitions linked."""
        if "$ref" in node:
            return self.link(_ref(node))
        if "const" in node:
            return f"`{json.dumps(node['const'])}`"
        if "enum" in node:
            return " or ".join(f"`{v}`" for v in node["enum"])
        if "oneOf" in node or "anyOf" in node:
            return " or ".join(self._union(_variants(node)))
        kind = node.get("type")
        if kind == "array":
            items = node.get("prefixItems")
            if items:
                return "[" + ", ".join(self.type_of(item) for item in items) + "]"
            return f"list of {self.type_of(node['items'])}" if node.get("items") else "list"
        if kind == "object" or "properties" in node:
            if "properties" in node:
                return "mapping of " + ", ".join(f"`{key}`" for key in node["properties"])
            values = node.get("additionalProperties")
            return f"mapping of name to {self.type_of(values)}" if isinstance(values, dict) and values else "mapping"
        if isinstance(kind, list):
            return " or ".join(kind)
        return f"{kind} ({node['format']})" if "format" in node else kind or "any"

    def _union(self, members: list[Node]) -> list[str]:
        """The members in words; kinds told apart by one key are listed under it, as `type: a, b, c`."""
        tagged: dict[str, list[str]] = {}
        for member in members:
            tag = self.tag(_ref(member)) if "$ref" in member else None
            if tag:
                tagged.setdefault(tag[0], []).append(self.link(_ref(member), str(tag[1])))
        parts, listed = [], set()
        for member in members:
            tag = self.tag(_ref(member)) if "$ref" in member else None
            if tag is None or len(tagged[tag[0]]) == 1:
                parts.append(self.type_of(member))
            elif tag[0] not in listed:
                listed.add(tag[0])
                parts.append(f"`{tag[0]}:` " + ", ".join(tagged[tag[0]]))
        return parts

    def table(self, node: Node, tag_key: str | None = None) -> list[str]:
        """One row per key of the mapping `node`."""
        required = set(node.get("required", []))
        exclusive = {key for choice in node.get("oneOf", []) for key in choice.get("required", [])}
        rows = []
        for key, prop in node.get("properties", {}).items():
            needed = "yes" if key in required or key == tag_key else "one of" if key in exclusive else ""
            default = "" if needed == "yes" or prop.get("default") is None else f"`{json.dumps(prop['default'])}`"
            limits = _constraints(prop)
            rows.append(
                f"| `{key}` | {self.type_of(prop)}{f' ({limits})' if limits else ''} | {needed} | {default} "
                f"| {_cell(prop.get('description', ''))} |"
            )
        return _table(["Key", "Type", "Required", "Default", "Description"], rows)

    def section(self, name: str) -> list[str]:
        node = self.defs[name]
        tag = self.tag(name)
        # A mapping's `oneOf` only names keys to set one of; a definition without keys is a union (`by:`).
        variants = [node] if "properties" in node else _variants(node)
        lines = [f"({_label(name)})=", f"## {self.title(name)}", ""]
        description = node.get("description") or next((v["description"] for v in variants if "description" in v), "")
        if description:
            lines += [_markdown(description), ""]
        if tag:
            lines += [f"Model: `{name}`", ""]
        if exclusive := [c["required"][0] for c in node.get("oneOf", []) if len(c.get("required", [])) == 1]:
            lines += ["Set exactly one of " + ", ".join(f"`{key}`" for key in exclusive) + ".", ""]
        for i, variant in enumerate(variants):
            if "properties" in variant:
                lines += ["Or as a mapping:", ""] if i else []
                lines += [*self.table(variant, tag[0] if tag else None), ""]
            else:
                lines += [f"Written as {self.type_of(variant)}.", ""]
        return lines

    def page(self, key: str) -> str:
        prop = self.schema["properties"][key]
        lines = [f"(config-key-{key})=", f"# `{key}:`", "", _markdown(prop.get("description", "")), ""]
        entries = [m for v in _variants(prop) for m in _variants(v.get("items", v))]
        if len(entries) > 1:
            kinds = [_ref(member) for entry in entries for member in _variants(entry)]
            rows = [f"| {self.link(name)} | {_cell(_first_paragraph(self.defs[name]))} |" for name in kinds]
            lines += ["Each entry is one of these kinds:", "", *_table(["Kind", "Description"], rows), ""]
        for name in self.pages[key]:
            lines += self.section(name)
        return "\n".join(lines).rstrip() + "\n"

    def index(self) -> str:
        lines = ["(config-reference)=", "# Config Reference", "", INTRO, "", "## Top-level keys", ""]
        lines += [_markdown(self.schema["description"]), ""]
        rows = []
        for key, prop in self.schema["properties"].items():
            shape = _variants(prop)[0]
            brief = "list" if shape.get("type") == "array" else "mapping" if "$ref" in shape else self.type_of(prop)
            default = "" if prop.get("default") is None else f"`{json.dumps(prop['default'])}`"
            name = f"[`{key}`]({key}.md)" if key in self.pages else f"`{key}`"
            rows.append(f"| {name} | {brief} | {default} | {_cell(prop.get('description', ''))} |")
        lines += [
            *_table(["Key", "Type", "Default", "Description"], rows),
            "",
            ":::{toctree}",
            ":hidden:",
            "",
            *self.pages,
            ":::",
        ]
        return "\n".join(lines) + "\n"


def _constraints(node: Node) -> str:
    words = {
        "minimum": "≥ {}",
        "exclusiveMinimum": "> {}",
        "maximum": "≤ {}",
        "exclusiveMaximum": "< {}",
        "minItems": "length ≥ {}",
        "maxItems": "length ≤ {}",
        "minProperties": "keys ≥ {}",
        "pattern": "matching `{}`",
    }
    found = [word.format(n[key]) for n in (node, *_variants(node)) for key, word in words.items() if key in n]
    return ", ".join(dict.fromkeys(found))


_SECTIONS = re.compile(r"\n[^\n]+\n-{3,}\n")  # numpydoc "Examples" and the like: Python, not config
_LITERAL = re.compile(r"::\n\n((?:(?: {4}[^\n]*)?\n)*(?: {4}[^\n]*)?)")
_ROLE = re.compile(r":[\w:]+:`(~?)([^`]+)`")


def _prose(text: str) -> str:
    """RST inline markup as Markdown: roles and double backticks to code, a bare `<` escaped."""
    text = _ROLE.sub(lambda m: f"`{m.group(2).rsplit('.', 1)[-1] if m.group(1) else m.group(2)}`", text)
    text = text.replace("``", "`")
    return re.sub(r"(`[^`\n]*`)|<", lambda m: m.group(1) or "\\<", text)


def _markdown(text: str) -> str:
    """A docstring as Markdown, its `::` literal blocks fenced as YAML."""
    pieces = _LITERAL.split(_SECTIONS.split(text)[0])
    return "".join(
        f":\n\n```yaml\n{textwrap.dedent(piece).strip()}\n```\n\n" if i % 2 else _prose(piece)
        for i, piece in enumerate(pieces)
    ).strip()


def _first_paragraph(node: Node) -> str:
    return _markdown(node.get("description", "")).split("\n\n")[0]


def _table(columns: list[str], rows: list[str]) -> list[str]:
    """A Markdown table, classed so that custom.css keeps each key's name on one line."""
    head = ["| " + " | ".join(columns) + " |", "|" + " --- |" * len(columns)]
    return ["```{table}", ":class: config-keys", "", *head, *rows, "```"]


def _cell(text: str) -> str:
    return _markdown(text).replace("\n", " ").replace("|", "\\|")


def write_pages(schema_path: Path = SCHEMA, out: Path = OUT) -> None:
    """Write the index and one page per top-level key, leaving unchanged pages untouched."""
    reference = Reference(json.loads(schema_path.read_text(encoding="utf-8")))
    pages = {"index.md": reference.index(), **{f"{key}.md": reference.page(key) for key in reference.pages}}
    out.mkdir(parents=True, exist_ok=True)
    for stale in {p.name for p in out.glob("*.md")} - pages.keys():
        (out / stale).unlink()
    for name, text in pages.items():
        path = out / name
        if not path.exists() or path.read_text(encoding="utf-8") != text:
            path.write_text(text, encoding="utf-8")


if __name__ == "__main__":
    write_pages()
    print(f"Wrote {OUT}")
