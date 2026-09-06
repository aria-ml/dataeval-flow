"""Tests for the label-space digest."""

import pytest

from dataeval_flow.label_space import label_space_digest, ontology_digest


@pytest.mark.required
class TestOntologyDigest:
    def test_is_order_independent(self) -> None:
        # An ontology is a set of concepts; the order a parser yields them is not part of
        # its identity.
        assert ontology_digest(["car", "truck", "person"]) == ontology_digest(["person", "car", "truck"])

    def test_a_new_concept_changes_it(self) -> None:
        assert ontology_digest(["car", "truck"]) != ontology_digest(["car", "truck", "bus"])

    def test_is_twelve_hex_chars(self) -> None:
        digest = ontology_digest(["car"])
        assert len(digest) == 12
        assert all(c in "0123456789abcdef" for c in digest)


@pytest.mark.required
class TestLabelSpaceDigest:
    def test_class_remap_order_does_not_matter(self) -> None:
        # A mapping is a set of pairs. Two configs listing them differently describe one
        # rewrite and must compare equal.
        a = label_space_digest(ontology="abc", class_remap={"people": "Person", "car": "Car"}, target=["Person", "Car"])
        b = label_space_digest(ontology="abc", class_remap={"car": "Car", "people": "Person"}, target=["Person", "Car"])
        assert a == b

    def test_target_order_does_matter(self) -> None:
        # The order IS the integer indexing: reordering it changes what every label means.
        a = label_space_digest(ontology="abc", class_remap={"car": "Car"}, target=["Person", "Car"])
        b = label_space_digest(ontology="abc", class_remap={"car": "Car"}, target=["Car", "Person"])
        assert a != b

    def test_ontology_participates(self) -> None:
        a = label_space_digest(ontology="abc", class_remap={"car": "Car"}, target=["Car"])
        b = label_space_digest(ontology="def", class_remap={"car": "Car"}, target=["Car"])
        assert a != b

    def test_remap_participates(self) -> None:
        a = label_space_digest(ontology="abc", class_remap={"car": "Car"}, target=["Car", "Truck"])
        b = label_space_digest(ontology="abc", class_remap={"car": "Truck"}, target=["Car", "Truck"])
        assert a != b

    def test_is_stable_across_runs(self) -> None:
        # A golden value. The digest is a cross-artifact contract: an audit archived today
        # must still match a run next year, so a change to the payload shape is a breaking
        # change and has to be a deliberate edit to this test.
        assert (
            label_space_digest(
                ontology="0123456789ab",
                class_remap={"car": "Car", "people": "Person"},
                target=["Person", "Car", "Truck"],
            )
            == "67112b6779a4"
        )
