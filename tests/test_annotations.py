"""No class annotation names something its own class body binds.

From Python 3.14 annotations are evaluated lazily, in a scope that sees the class namespace, so an
annotation that says ``type[Any]`` in a class with a ``type`` field reads the field, not the builtin.
Pydantic evaluates model annotations at class creation, so the module fails to import on 3.14 alone.
Spell the builtin ``builtins.type`` there instead.
"""

import ast
from collections.abc import Iterator
from contextlib import suppress
from pathlib import Path

SRC = Path(__file__).parents[1] / "src" / "dataeval_flow"


def _bound(cls: ast.ClassDef) -> set[str]:
    names: set[str] = set()
    for node in cls.body:
        if isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
            names.add(node.target.id)
        elif isinstance(node, ast.Assign):
            names.update(target.id for target in node.targets if isinstance(target, ast.Name))
        elif isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef | ast.ClassDef):
            names.add(node.name)
    return names


def _annotations(cls: ast.ClassDef) -> Iterator[ast.expr]:
    for node in cls.body:
        if isinstance(node, ast.AnnAssign):
            yield node.annotation
        elif isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef):
            arguments = node.args
            params = [*arguments.posonlyargs, *arguments.args, *arguments.kwonlyargs, arguments.vararg, arguments.kwarg]
            yield from (param.annotation for param in params if param is not None and param.annotation is not None)
            if node.returns is not None:
                yield node.returns


def _names(annotation: ast.expr) -> set[str]:
    names: set[str] = set()
    for node in ast.walk(annotation):
        if isinstance(node, ast.Name):
            names.add(node.id)
        elif isinstance(node, ast.Constant) and isinstance(node.value, str):
            with suppress(SyntaxError):
                names |= _names(ast.parse(node.value, mode="eval").body)
    return names


def test_no_annotation_names_a_member_of_its_own_class() -> None:
    shadowed = []
    for path in sorted(SRC.rglob("*.py")):
        for cls in (node for node in ast.walk(ast.parse(path.read_text())) if isinstance(node, ast.ClassDef)):
            bound = _bound(cls)
            for annotation in _annotations(cls):
                shadowed += [
                    f"{path.relative_to(SRC)}:{annotation.lineno} {cls.name}: {name}"
                    for name in sorted(_names(annotation) & bound)
                ]
    assert shadowed == []
