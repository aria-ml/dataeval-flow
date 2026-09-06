"""Ontology configuration schemas — a named label space, defined once and shared.

A label space is a decision several workflows must agree on. Two coverage tasks reading
different ontologies produce worklists that cannot be compared, and two `Relabel` views
conforming to different vocabularies produce datasets that cannot be merged. Defining it once
under `ontologies:` and referencing it by name removes the class of failure where copies
drift apart — the same argument `metadata:` policies are built on.
"""

from collections.abc import Sequence
from typing import ClassVar

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from dataeval_flow.config._paths import validate_config_path

__all__ = ["OntologyConceptConfig", "OntologyConfig"]


class OntologyConceptConfig(BaseModel):
    """One concept declared in config rather than read from an artifact.

    Field names match :class:`dataeval.types.OntologyConcept` exactly, so an entry converts
    to one by name. Declaring a concept is how a dataset's class survives an ontology that
    omits it, without editing an artifact that may belong to someone else.

    YAML example::

        concepts:
          - id: http://example.org/cv#FreightCar
            label: Freight Car
            synonyms: [freight_car, freight car]
            parents: [http://example.org/cv#LandVehicle]
            definition: A railway car designed to carry freight.
    """

    model_config: ClassVar[ConfigDict] = ConfigDict(extra="forbid")

    id: str = Field(description="Concept id. An IRI for an RDF artifact; any stable string otherwise.")
    label: str = Field(description="Human-readable name for the concept.")
    synonyms: Sequence[str] = Field(
        default_factory=list,
        description=(
            "Alternative spellings this concept answers to. Alignment anchors on label and "
            "synonym matches, so a concept declared without the dataset's own spelling for it "
            "will not anchor to that class."
        ),
    )
    parents: Sequence[str] = Field(
        default_factory=list,
        description="Ids of the concepts this one is a kind of. Empty makes it a root.",
    )
    equivalent_to: Sequence[str] = Field(
        default_factory=list,
        description="Ids of concepts this one is the same as, for cross-vocabulary equivalence.",
    )
    definition: str | None = Field(default=None, description="Prose definition, carried through unread.")


class OntologyConfig(BaseModel):
    """A named label space, referenced by the workflows that share it.

    YAML example::

        ontologies:
          - name: vehicles
            source: config/label_ontology.jsonld
            concepts:
              - id: http://example.org/cv#FreightCar
                label: Freight Car
                synonyms: [freight_car, freight car]
                parents: [http://example.org/cv#LandVehicle]

        workflows:
          - name: audit
            type: data-coverage
            ontology: vehicles
    """

    model_config: ClassVar[ConfigDict] = ConfigDict(extra="forbid")

    name: str = Field(description="Identifier for this ontology")
    source: str | None = Field(
        default=None,
        description=(
            "Path, under the data root, to a serialized RDF artifact "
            "(.ttl/.rdf/.owl/.xml/.nt/.jsonld). Omit to build the label space from `concepts` "
            'alone. Reading a file needs rdflib: install it with `pip install "dataeval[ontology]"`.'
        ),
    )
    concepts: Sequence[OntologyConceptConfig] = Field(
        default_factory=list,
        description=(
            "Concepts added on top of `source`, or the whole label space when `source` is "
            "omitted. A declared concept with an id the artifact already defines replaces it."
        ),
    )

    @field_validator("source")
    @classmethod
    def _check_source_path(cls, value: str | None) -> str | None:
        """Keep the artifact path portable, like every other config path."""
        return None if value is None else validate_config_path(value)

    @model_validator(mode="after")
    def _needs_a_label_space(self) -> "OntologyConfig":
        """An entry naming neither a source nor a concept describes nothing."""
        if self.source is None and not self.concepts:
            raise ValueError(
                f"Ontology {self.name!r} declares neither `source` nor `concepts`, so it "
                "defines no label space. Give it a path to an artifact, a list of concepts, "
                "or both.",
            )
        return self
