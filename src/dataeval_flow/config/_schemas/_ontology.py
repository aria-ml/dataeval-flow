"""Ontology configuration schemas: a named label space, defined once and shared."""

from collections.abc import Sequence
from typing import ClassVar

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from dataeval_flow.config._paths import validate_config_path

__all__ = ["OntologyConceptConfig", "OntologyConfig"]


class OntologyConceptConfig(BaseModel):
    """One concept declared in config rather than read from an artifact.

    Declare a concept to keep a dataset class the ontology omits, without editing an
    artifact you may not own. Field names match :class:`dataeval.types.OntologyConcept`
    exactly, so an entry converts to one by name.

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
            "Alternative spellings this concept answers to. Alignment matches on labels and "
            "synonyms, so give a declared concept the dataset's own spelling or it will not "
            "match that class."
        ),
    )
    parents: Sequence[str] = Field(
        default_factory=list,
        description="Ids of the concepts this one is a kind of. Leave empty to make it a root.",
    )
    equivalent_to: Sequence[str] = Field(
        default_factory=list,
        description="Ids of concepts this one is the same as, for cross-vocabulary equivalence.",
    )
    definition: str | None = Field(default=None, description="Prose definition. Carried through unread.")


class OntologyConfig(BaseModel):
    """A named label space, referenced by the workflows that share it.

    Define an ontology once here and reference it by name so workflows meant to be compared
    read the same vocabulary. Two tasks reading different ontologies produce worklists you
    cannot compare.

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
            "(.ttl/.rdf/.owl/.xml/.nt/.jsonld/.json). Omit to build the label space from `concepts` "
            'alone. Reading a file needs rdflib: install it with `pip install "dataeval[ontology]"`.'
        ),
    )
    concepts: Sequence[OntologyConceptConfig] = Field(
        default_factory=list,
        description=(
            "Concepts added on top of `source`, or the whole label space when you omit "
            "`source`. A declared concept replaces one the artifact defines under the same id. "
            "Replacement is total: restate `parents` on it too, or it becomes a root."
        ),
    )

    @field_validator("source")
    @classmethod
    def _check_source_path(cls, value: str | None) -> str | None:
        """Keep the artifact path portable, as every other config path is."""
        return None if value is None else validate_config_path(value)

    @model_validator(mode="after")
    def _needs_a_label_space(self) -> "OntologyConfig":
        """Refuse an entry that names neither a source nor a concept."""
        if self.source is None and not self.concepts:
            raise ValueError(
                f"Ontology {self.name!r} declares neither `source` nor `concepts`, so it "
                "defines no label space. Give it a path to an artifact, a list of concepts, "
                "or both.",
            )
        return self
