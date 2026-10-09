"""TC-8-2 — the taxonomy preset: a dataset's labels judged against a declared ontology."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
from pydantic import ValidationError

from dataeval_flow.config import OntologyConfig, SourceConfig, ViewConfig, ViewOperation
from dataeval_flow.workflows.taxonomy import TaxonomyConfig
from verification.functional.workflows._synthetic import CLASSES, Images, by_title, run_preset

pytestmark = pytest.mark.required

ANIMALS_ONTOLOGY = {"animal": {"mammal": ["cat", "dog"], "avian": ["bird", "owl"]}}
"""Four leaves; the datasets below label three of them and never `owl`."""
TITLES = ["Leaf Coverage", "Label Conformance", "Label Mergeability", "Ontology Structure"]


ANIMALS_TURTLE = """\
@prefix ex: <http://example.org/> .
@prefix rdfs: <http://www.w3.org/2000/01/rdf-schema#> .
@prefix owl: <http://www.w3.org/2002/07/owl#> .
ex:Animal a owl:Class ; rdfs:label "animal" .
ex:Cat a owl:Class ; rdfs:label "cat" ; rdfs:subClassOf ex:Animal .
ex:Dog a owl:Class ; rdfs:label "dog" ; rdfs:subClassOf ex:Animal .
ex:Bird a owl:Class ; rdfs:label "bird" ; rdfs:subClassOf ex:Animal .
ex:Owl a owl:Class ; rdfs:label "owl" ; rdfs:subClassOf ex:Bird .
"""


def taxonomy(**settings: Any) -> dict[str, Any]:
    return {"type": "taxonomy", "ontology": ANIMALS_ONTOLOGY, **settings}


def animals_pool() -> OntologyConfig:
    """A shared `ontologies:` entry: `cat` has the synonym `kitty`."""
    return OntologyConfig.model_validate(
        {
            "name": "animals",
            "concepts": [
                {"id": "animal", "label": "animal"},
                {"id": "cat", "label": "cat", "synonyms": ["kitty"], "parents": ["animal"]},
                {"id": "dog", "label": "dog", "parents": ["animal"]},
                {"id": "bird", "label": "bird", "parents": ["animal"]},
            ],
        }
    )


class TestTaxonomyPreset:
    def test_leaf_coverage_conformance_alignment_and_structure_are_judged(self) -> None:
        result = run_preset(taxonomy(), Images(30))

        assert result.success, result.errors
        assert result.type == "taxonomy"
        assert [f.title for f in result.findings] == TITLES
        found = by_title(result)
        assert found["Leaf Coverage"].severity == "warning"  # `owl` has no examples
        assert found["Leaf Coverage"].brief.startswith("leaf coverage 75.0% · 1 to acquire")
        assert found["Label Conformance"].brief == "conforms"
        assert found["Label Mergeability"].severity == "ok"
        assert found["Ontology Structure"].brief == "7 concepts, 4 leaves, depth 2"
        rows = result.steps["representation"].output.data().to_dicts()
        assert [(r["concept"], r["action"], r["count"], r["target"]) for r in rows] == [
            ("owl", "acquire", 0, 8),  # a sanctioned class never collected
            ("bird", "augment", 6, 8),
        ]

    def test_a_class_name_the_ontology_does_not_contain_fails_conformance_and_alignment(self) -> None:
        data = Images(30, index2label={0: "cat", 1: "doge", 2: "bird"})

        result = run_preset(taxonomy(), data)

        found = by_title(result)
        assert found["Label Conformance"].severity == "warning"
        assert found["Label Conformance"].brief == "1 unmatched, 0 ambiguous"
        reconciliation = result.steps["label-reconciliation"].output.data()
        assert reconciliation["unmatched"] == ["doge"]
        assert reconciliation["matched"] == {"cat": "cat", "bird": "bird"}
        alignment = result.steps["label-alignment"].output.alignment
        assert alignment.mergeability == "partial"
        assert alignment.unaligned_source == ["doge"]
        assert found["Label Mergeability"].severity == "warning"

    def test_thresholds_in_checks_decide_whether_a_finding_warns(self) -> None:
        data = Images(30, index2label={0: "cat", 1: "doge", 2: "bird"})
        checks = {"leaf-coverage": {"coverage": None, "empty_branches": None}, "label-conformance": {"warning": 1}}

        found = by_title(run_preset(taxonomy(checks=checks), data))

        assert found["Leaf Coverage"].severity == "info"  # judged by neither criterion
        assert found["Label Conformance"].severity == "ok"  # one unmatched name is within the limit of one

    def test_expected_shares_set_each_named_class_its_own_target(self) -> None:
        result = run_preset(taxonomy(representation={"expected": {"cat": 0.9}}), Images(30))

        rows = {r["concept"]: r for r in result.steps["representation"].output.data().to_dicts()}
        assert (rows["cat"]["count"], rows["cat"]["target"]) == (15, 27)  # 90% of 30 items
        assert rows["owl"]["action"] == "acquire"

    def test_the_label_pattern_reports_concept_labels_that_break_it(self) -> None:
        ontology = {"Animal": {"Mammal": ["cat", "Dog-Breed"]}}

        result = run_preset(
            {"type": "taxonomy", "ontology": ontology, "ontology-validation": {"label_pattern": "^[a-z]+$"}},
            Images(30),
        )

        structure = result.steps["ontology-validation"].output.data()
        assert set(structure["nonconforming_labels"]) == {"Animal", "Mammal", "Dog-Breed"}
        assert by_title(result)["Ontology Structure"].brief.endswith("3 nonconforming labels")

    def test_an_ontology_can_be_read_from_an_rdf_file_under_the_data_root(self, tmp_path: Path) -> None:
        pytest.importorskip("rdflib", reason="the ontology extra is not installed")
        (tmp_path / "animals.ttl").write_text(ANIMALS_TURTLE)

        result = run_preset({"type": "taxonomy", "ontology": "animals.ttl"}, Images(30), data_dir=tmp_path)

        assert result.success, result.errors
        found = by_title(result)
        assert found["Leaf Coverage"].brief.startswith("leaf coverage 66.7% · 1 to acquire")
        assert found["Label Conformance"].brief == "conforms"
        assert found["Ontology Structure"].brief.startswith("5 concepts, 3 leaves, depth 2")
        (owl, _dog) = result.steps["representation"].output.data().to_dicts()
        assert (owl["concept"], owl["label"], owl["count"]) == ("http://example.org/Owl", "owl", 0)

    def test_an_ontologies_entry_with_concepts_and_synonyms_defines_the_label_space(self) -> None:
        data = Images(30, index2label={0: "kitty", 1: "dog", 2: "bird"})

        result = run_preset({"type": "taxonomy", "ontology": "animals"}, data, extra={"ontologies": [animals_pool()]})

        assert result.success, result.errors
        found = by_title(result)
        assert found["Label Conformance"].brief == "conforms"  # `kitty` resolves through the synonym
        assert found["Leaf Coverage"].brief.startswith("leaf coverage 100.0%")
        alignment = result.steps["label-alignment"].output.alignment
        assert alignment.class_remap == {"kitty": "cat", "dog": "dog", "bird": "bird"}
        assert alignment.mergeability == "lossless"

    def test_the_alignment_stanza_conforms_the_source_and_the_label_space_digest_joins_the_runs(self) -> None:
        data = Images(30, index2label={0: "kitty", 1: "dog", 2: "bird"})
        pool = animals_pool()
        before = run_preset({"type": "taxonomy", "ontology": "animals"}, data, extra={"ontologies": [pool]})
        alignment = before.steps["label-alignment"].output.alignment
        relabel = ViewOperation(
            type="Relabel", params={"class_remap": alignment.class_remap, "target": alignment.target_vocabulary}
        )

        after = run_preset(
            {"type": "taxonomy", "ontology": "animals"},
            data,
            extra={
                "ontologies": [pool],
                "views": [ViewConfig(name="conform", operations=[relabel])],
                "sources": [SourceConfig(name="src", dataset="src_data", view="conform")],
            },
        )

        assert after.success, after.errors
        assert before.metadata.label_space_digest == alignment.label_space_digest
        assert after.metadata.label_space_digest == before.metadata.label_space_digest
        (record,) = after.metadata.label_space
        assert (record.source, record.ontology, record.class_remap) == ("src", "animals", alignment.class_remap)
        assert by_title(after)["Label Conformance"].brief == "conforms"

    def test_an_ontology_is_required(self) -> None:
        with pytest.raises(ValidationError, match="name one with `ontology:`"):
            TaxonomyConfig.model_validate({"name": "w"})

    def test_an_ontology_file_that_cannot_be_read_fails_the_task_naming_the_file(self) -> None:
        result = run_preset({"type": "taxonomy", "ontology": "missing/vocab.ttl"}, Images(30, index2label=CLASSES))

        assert not result.success
        assert any("missing/vocab.ttl" in error for error in result.errors)
