"""Tests for the shared ontology loader."""

import builtins
from pathlib import Path
from typing import Any

import pytest

from dataeval_flow.workflows._ontology import OntologyLoadError, load_ontology, synthesize_ontology

_TURTLE = """
@prefix skos: <http://www.w3.org/2004/02/skos/core#> .
@prefix ex:   <http://example.org/> .

ex:vehicle a skos:Concept ; skos:prefLabel "vehicle" .
ex:car     a skos:Concept ; skos:prefLabel "car"  ; skos:broader ex:vehicle .
ex:truck   a skos:Concept ; skos:prefLabel "truck"; skos:broader ex:vehicle .
"""


class TestLoadInline:
    def test_nested_mapping(self) -> None:
        onto, source = load_ontology({"vehicle": ["car", "truck"]})
        assert source == "inline"
        assert set(onto.leaves) == {"car", "truck"}

    def test_cycle_is_reported(self) -> None:
        # A concept that is its own ancestor makes subsumption meaningless.
        with pytest.raises(OntologyLoadError) as exc:
            load_ontology({"a": {"b": ["a"]}})
        assert "a" in str(exc.value)


@pytest.mark.optional
class TestLoadFile:
    def test_turtle(self, tmp_path: Path) -> None:
        path = tmp_path / "taxonomy.ttl"
        path.write_text(_TURTLE)
        onto, source = load_ontology(str(path))
        assert source == str(path)
        assert set(onto.leaves) == {"http://example.org/car", "http://example.org/truck"}

    def test_missing_file(self, tmp_path: Path) -> None:
        with pytest.raises(OntologyLoadError) as exc:
            load_ontology(str(tmp_path / "nope.ttl"))
        assert "nope.ttl" in str(exc.value)

    def test_non_utf8_file(self, tmp_path: Path) -> None:
        path = tmp_path / "latin1.ttl"
        # Otherwise-valid turtle, but with a non-ASCII label encoded as latin-1
        # rather than UTF-8 — decoding as UTF-8 must fail cleanly, not escape raw.
        content = """
@prefix skos: <http://www.w3.org/2004/02/skos/core#> .
@prefix ex:   <http://example.org/> .

ex:vehicle a skos:Concept ; skos:prefLabel "véhicule" .
"""
        path.write_bytes(content.encode("latin-1"))
        with pytest.raises(OntologyLoadError) as exc:
            load_ontology(str(path))
        assert str(path) in str(exc.value)

    def test_parse_error(self, tmp_path: Path) -> None:
        path = tmp_path / "broken.ttl"
        path.write_text("this is not turtle {{{")
        with pytest.raises(OntologyLoadError) as exc:
            load_ontology(str(path))
        assert "turtle" in str(exc.value)

    def test_unknown_suffix_lets_rdflib_guess(self, tmp_path: Path) -> None:
        path = tmp_path / "taxonomy.unknown"
        path.write_text(_TURTLE)
        # An unknown suffix means no format hint is passed (fmt=None), so rdflib
        # is left to guess from content. rdflib 7.x guesses turtle correctly here,
        # so this succeeds rather than erroring — verified against the installed
        # rdflib before writing this assertion.
        onto, source = load_ontology(str(path))
        assert source == str(path)
        assert set(onto.leaves) == {"http://example.org/car", "http://example.org/truck"}

    def test_rdflib_missing(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        path = tmp_path / "taxonomy.ttl"
        path.write_text(_TURTLE)

        real_import = builtins.__import__

        def fake_import(name: str, *args: Any, **kwargs: Any) -> Any:
            if name == "rdflib" or name.startswith("rdflib."):
                raise ImportError("No module named 'rdflib'")
            return real_import(name, *args, **kwargs)

        monkeypatch.setattr(builtins, "__import__", fake_import)

        with pytest.raises(OntologyLoadError) as exc:
            load_ontology(str(path))
        assert "dataeval[ontology]" in str(exc.value)


class TestSynthesize:
    def test_flat_from_index2label(self) -> None:
        onto, source = synthesize_ontology({0: "cat", 1: "dog", 2: "bird"})
        assert source == "index2label"
        assert set(onto.leaves) == {"cat", "dog", "bird"}
        # A flat vocabulary has no is-a edges: every concept is both root and leaf.
        assert set(onto.roots) == set(onto.leaves)

    def test_empty_is_an_error(self) -> None:
        with pytest.raises(OntologyLoadError):
            synthesize_ontology({})
