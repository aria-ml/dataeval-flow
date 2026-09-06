"""Tests for the shared ontology loader."""

import builtins
import logging
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


@pytest.mark.required
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

    def test_source_resolves_against_the_given_root(self, tmp_path: Path) -> None:
        # TestDataRoot proves a wrong root misses. This proves the right root hits, so
        # data_dir is genuinely honoured rather than accepted and ignored.
        (tmp_path / "config").mkdir()
        (tmp_path / "config" / "taxonomy.ttl").write_text(_TURTLE)
        onto, source = load_ontology("config/taxonomy.ttl", data_dir=tmp_path)
        assert set(onto.leaves) == {"http://example.org/car", "http://example.org/truck"}
        assert str(tmp_path) in source

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


@pytest.mark.required
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


@pytest.mark.required
class TestDeclaredConcepts:
    def test_concepts_extend_an_inline_hierarchy(self) -> None:
        onto, source = load_ontology(
            {"vehicle": ["car"]},
            concepts=[{"id": "freight_car", "label": "Freight Car", "parents": ["vehicle"]}],
        )
        assert source == "inline"
        assert "freight_car" in set(onto.ids)
        assert "car" in set(onto.ids)

    def test_a_declared_synonym_is_findable(self) -> None:
        # Alignment anchors on synonyms, so this is the field that makes a declared concept
        # actually match the dataset's own spelling.
        onto, _ = load_ontology(
            {"vehicle": ["car"]},
            concepts=[
                {
                    "id": "freight_car",
                    "label": "Freight Car",
                    "synonyms": ["freight car"],
                    "parents": ["vehicle"],
                }
            ],
        )
        assert onto.find("freight car") == ("freight_car",)

    def test_concepts_alone_build_the_space(self) -> None:
        onto, source = load_ontology(
            None,
            concepts=[
                {"id": "vehicle", "label": "vehicle"},
                {"id": "car", "label": "car", "parents": ["vehicle"]},
            ],
        )
        assert source == "concepts"
        assert set(onto.ids) == {"vehicle", "car"}

    def test_nothing_at_all_is_an_error(self) -> None:
        with pytest.raises(OntologyLoadError):
            load_ontology(None)

    def test_a_malformed_concept_is_reported(self) -> None:
        # A concept missing `label` must fail as a config error naming the ontology, not as
        # an opaque pydantic traceback from inside the loader.
        with pytest.raises(OntologyLoadError) as exc:
            load_ontology(None, concepts=[{"id": "car"}])
        assert "label" in str(exc.value)

    def test_a_duplicate_id_among_concepts_alone_is_reported(self) -> None:
        # Two declared concepts sharing an id is an ordinary config typo, not a reason to
        # let dataeval's own exception escape the loader.
        with pytest.raises(OntologyLoadError):
            load_ontology(
                None,
                concepts=[{"id": "car", "label": "Car"}, {"id": "car", "label": "Again"}],
            )

    def test_a_duplicate_id_across_the_merge_is_reported(self) -> None:
        # Same failure mode, but when the duplicate arises from merging declared concepts
        # onto each other rather than from the concepts-only branch.
        with pytest.raises(OntologyLoadError):
            load_ontology(
                {"vehicle": ["car"]},
                concepts=[{"id": "x", "label": "X"}, {"id": "x", "label": "Y"}],
            )


@pytest.mark.required
class TestDataRoot:
    def test_source_resolves_against_the_given_root(self, tmp_path: Path) -> None:
        # The orchestrator knows the data root; the loader must honour it rather than
        # falling back to the process-wide default.
        (tmp_path / "config").mkdir()
        path = tmp_path / "config" / "taxonomy.ttl"
        path.write_text(_TURTLE)
        with pytest.raises(OntologyLoadError):
            load_ontology("config/taxonomy.ttl", data_dir=tmp_path / "elsewhere")


@pytest.mark.required
class TestResolveOntology:
    @staticmethod
    def _pool() -> list[Any]:
        from dataeval_flow.config.schemas import OntologyConfig

        return [
            OntologyConfig(
                name="vehicles",
                concepts=[  # type: ignore[arg-type]
                    {"id": "vehicle", "label": "vehicle"},
                    {"id": "car", "label": "car", "parents": ["vehicle"]},
                ],
            )
        ]

    def test_a_name_resolves_from_the_pool(self) -> None:
        from dataeval_flow.workflows._ontology import resolve_ontology

        onto, source = resolve_ontology("vehicles", self._pool())
        assert source == "vehicles"
        assert set(onto.ids) == {"vehicle", "car"}

    def test_an_unmatched_string_is_still_a_path(self, tmp_path: Path) -> None:
        # Backward compatibility: a config that named a file before must keep working.
        from dataeval_flow.workflows._ontology import resolve_ontology

        with pytest.raises(OntologyLoadError) as exc:
            resolve_ontology("does/not/exist.ttl", self._pool(), data_dir=tmp_path)
        assert "could not read" in str(exc.value)

    def test_an_inline_mapping_never_consults_the_pool(self) -> None:
        from dataeval_flow.workflows._ontology import resolve_ontology

        onto, source = resolve_ontology({"animal": ["cat"]}, self._pool())
        assert source == "inline"
        assert set(onto.ids) == {"animal", "cat"}

    def test_no_pool_leaves_a_string_a_path(self, tmp_path: Path) -> None:
        from dataeval_flow.workflows._ontology import resolve_ontology

        with pytest.raises(OntologyLoadError):
            resolve_ontology("vehicles", None, data_dir=tmp_path)

    def test_a_name_that_is_also_a_file_is_refused(self, tmp_path: Path) -> None:
        # Two people have said different things about one string. Refusing names both.
        from dataeval_flow.workflows._ontology import resolve_ontology

        (tmp_path / "vehicles").write_text("not really an ontology")
        with pytest.raises(OntologyLoadError) as exc:
            resolve_ontology("vehicles", self._pool(), data_dir=tmp_path)
        message = str(exc.value)
        assert "vehicles" in message
        assert "ontologies" in message

    def test_a_pool_entry_carries_its_concepts(self) -> None:
        from dataeval_flow.config.schemas import OntologyConfig
        from dataeval_flow.workflows._ontology import resolve_ontology

        pool = [
            OntologyConfig(
                name="extended",
                concepts=[  # type: ignore[arg-type]
                    {"id": "vehicle", "label": "vehicle"},
                    {"id": "freight_car", "label": "Freight Car", "synonyms": ["freight car"]},
                ],
            )
        ]
        onto, _ = resolve_ontology("extended", pool)
        assert onto.find("freight car") == ("freight_car",)

    def test_a_permission_error_from_the_collision_check_does_not_abort(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # Path.is_file() can raise PermissionError on a non-searchable parent directory,
        # exactly like resolve_path does — but that call used to sit outside the guard.
        # Simulated directly, since chmod-based permission tests are not portable across CI.
        from dataeval_flow.workflows._ontology import resolve_ontology

        def _raise(_self: Path) -> bool:
            raise PermissionError("permission denied")

        monkeypatch.setattr(Path, "is_file", _raise)

        onto, source = resolve_ontology("vehicles", self._pool(), data_dir=tmp_path)
        assert source == "vehicles"
        assert set(onto.ids) == {"vehicle", "car"}


@pytest.mark.required
class TestReplacementReRootsTheHierarchy:
    """A declared concept replaces the artifact's whole entry, not just its label/synonyms."""

    def test_omitting_parents_detaches_the_replaced_concept(self) -> None:
        onto, _ = load_ontology(
            {"vehicle": {"car": ["sedan"]}},
            concepts=[{"id": "car", "label": "car", "synonyms": ["automobile"]}],
        )
        # `car` loses its place under `vehicle` because replacement is total: the
        # artifact's `car` (with its parent) is gone, and the declared `car` has none.
        assert set(onto.roots) == {"vehicle", "car"}
        assert set(onto.leaves) == {"vehicle", "sedan"}

    def test_replacing_a_concept_logs_a_warning(self, caplog: pytest.LogCaptureFixture) -> None:
        with caplog.at_level(logging.WARNING, logger="dataeval_flow.workflows._ontology"):
            load_ontology(
                {"vehicle": {"car": ["sedan"]}},
                concepts=[{"id": "car", "label": "car", "synonyms": ["automobile"]}],
            )
        messages = [record.getMessage() for record in caplog.records]
        assert any("car" in message and "parents" in message for message in messages)

    def test_restating_parents_keeps_the_concept_in_place(self) -> None:
        # The warning names the fix: restate `parents` to avoid the re-rooting above.
        onto, _ = load_ontology(
            {"vehicle": {"car": ["sedan"]}},
            concepts=[{"id": "car", "label": "car", "synonyms": ["automobile"], "parents": ["vehicle"]}],
        )
        assert set(onto.roots) == {"vehicle"}
        assert set(onto.leaves) == {"sedan"}
