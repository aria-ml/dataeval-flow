"""Tests for adapting a resolved source into a datamaite dataset."""

import json
import logging
from pathlib import Path
from types import SimpleNamespace
from typing import Literal
from unittest.mock import patch

import numpy as np
import pytest
from datamaite import (
    CategoryEntry,
    DatasetMetadata,
    ImageObjectDetectionSample,
    ObjectDetectionAnnotation,
    ObjectDetectionDataset,
    Taxonomy,
)

from dataeval_flow import export as export_module
from dataeval_flow.config import PipelineConfig, SourceConfig, ViewConfig, ViewOperation
from dataeval_flow.config.schemas import (
    DatasetProtocolConfig,
    ExportConfig,
    OntologyConceptConfig,
    OntologyConfig,
)
from dataeval_flow.config.schemas._workflow import DataCoverageWorkflowConfig
from dataeval_flow.export import build_od_dataset, export_provenance, write_export, write_exports
from dataeval_flow.sources import label_space_records, resolve_source
from dataeval_flow.workflow import ResolvedOntology
from tests.test_sources import _PNG, _merge_config, _od_dataset

_CAR = ObjectDetectionAnnotation(bbox=(1.0, 2.0, 3.0, 4.0), category_id=0, category_name="Car")


def _disk_config(
    tmp_path: Path,
    *,
    operations: list[ViewOperation] | None = None,
    splits: tuple[str | None, ...] = (None,),
    declare_size: bool = True,
) -> PipelineConfig:
    """A single source whose images are PNGs on disk rather than bytes in memory."""
    samples = []
    for index, split in enumerate(splits):
        png = tmp_path / f"on_disk_{index}.png"
        png.write_bytes(_PNG)
        samples.append(
            ImageObjectDetectionSample(
                image_id=index,
                path_or_uri=str(png),
                file_name=png.name,
                width=8 if declare_size else None,
                height=8 if declare_size else None,
                split=split,
                metadata={"weather": "rain"},
                detections=(_CAR,),
            )
        )
    dataset = ObjectDetectionDataset(
        samples=tuple(samples),
        dataset_metadata=DatasetMetadata(
            taxonomy=Taxonomy(entries=(CategoryEntry(source_id=0, name="Car"),), id_density="dense")
        ),
        dataset_id="on_disk",
    )
    return PipelineConfig(
        datasets=[DatasetProtocolConfig(name="ds_disk", dataset=dataset)],
        views=[ViewConfig(name="transform", operations=operations)] if operations else None,
        sources=[SourceConfig(name="disk", dataset="ds_disk", view="transform" if operations else None)],
    )


def _built_from_disk(tmp_path: Path, **kwargs) -> ObjectDetectionDataset:
    """Export the on-disk source built with *kwargs*."""
    resolved = resolve_source("disk", _disk_config(tmp_path, **kwargs))
    return build_od_dataset(resolved, dataset_metadata=DatasetMetadata())


class _PlainOdDataset:
    """A MAITE OD dataset that is not datamaite-backed, so no source records exist."""

    @property
    def metadata(self) -> dict:
        return {"id": "plain", "index2label": {0: "Car"}}

    def __len__(self) -> int:
        return 2

    def __getitem__(self, index: int):
        target = SimpleNamespace(
            boxes=np.array([[1.0, 2.0, 4.0, 6.0]], dtype=np.float32),
            labels=np.array([0], dtype=np.int64),
        )
        return np.zeros((3, 8, 8), dtype=np.uint8), target, {"id": index, "weather": "rain"}


def _plain_config() -> PipelineConfig:
    """A source backed by a dataset datamaite never loaded."""
    return PipelineConfig(
        datasets=[DatasetProtocolConfig(name="ds_plain", dataset=_PlainOdDataset())],
        sources=[SourceConfig(name="plain", dataset="ds_plain")],
    )


def _sparse_config() -> PipelineConfig:
    """A source whose category ids are COCO-shaped: sparse, and not starting at zero."""
    names = {1: "person", 17: "cat", 90: "toothbrush"}
    dataset = ObjectDetectionDataset(
        samples=(
            ImageObjectDetectionSample(
                image_id=0,
                image_bytes=_PNG,
                file_name="img_000.png",
                width=8,
                height=8,
                detections=(ObjectDetectionAnnotation(bbox=(1.0, 2.0, 3.0, 4.0), category_id=17, category_name="cat"),),
            ),
        ),
        dataset_metadata=DatasetMetadata(
            taxonomy=Taxonomy(
                entries=tuple(CategoryEntry(source_id=code, name=name) for code, name in names.items()),
                id_density="sparse",
            )
        ),
        dataset_id="sparse",
    )
    return PipelineConfig(
        datasets=[DatasetProtocolConfig(name="ds_sparse", dataset=dataset)],
        sources=[SourceConfig(name="sparse", dataset="ds_sparse")],
    )


@pytest.mark.required
class TestBuildOdDataset:
    """build_od_dataset materializes the corpus datamaite's writers need."""

    def _built(self, source: str = "merged"):
        return build_od_dataset(resolve_source(source, _merge_config()), dataset_metadata=DatasetMetadata())

    def test_every_datum_becomes_a_sample(self):
        assert len(self._built().samples) == 4

    def test_image_ids_are_integers(self):
        """COCO drops a sample whose image_id is not an int, so allocate them."""
        assert [s.image_id for s in self._built().samples] == [0, 1, 2, 3]

    def test_the_merged_id_survives_as_source_id(self):
        assert [s.metadata["source_id"] for s in self._built().samples] == ["0:0", "0:1", "1:0", "1:1"]

    def test_file_names_are_prefixed_per_operand(self):
        """Two operands both hold img_000.png; without a prefix one overwrites the other."""
        names = [s.file_name for s in self._built().samples]
        assert names == ["0/img_000.png", "0/img_001.png", "1/img_000.png", "1/img_001.png"]

    def test_a_single_source_keeps_its_file_names(self):
        assert [s.file_name for s in self._built("a").samples] == ["img_000.png", "img_001.png"]

    def test_boxes_convert_from_xyxy_to_xywh(self):
        """The fixture's source box is xywh (1,2,3,4); MAITE hands it back as xyxy (1,2,4,6)."""
        assert self._built().samples[0].detections[0].bbox == (1.0, 2.0, 3.0, 4.0)

    def test_detections_carry_the_conformed_class(self):
        first = self._built().samples[0].detections[0]
        assert first.category_id == 1
        assert first.category_name == "Car"
        last = self._built().samples[2].detections[0]
        assert last.category_id == 2
        assert last.category_name == "Truck"

    def test_datum_metadata_passes_through_without_the_derived_keys(self):
        meta = self._built().samples[0].metadata
        assert meta["weather"] == "rain"
        assert "id" not in meta
        assert "width" not in meta

    def test_bytes_backed_source_falls_back_to_encoded_bytes(self):
        """A source with no path on disk still exports, by carrying its pixels."""
        sample = self._built().samples[0]
        assert sample.path_or_uri is None
        assert sample.image_bytes is not None

    def test_a_relabel_that_dropped_everything_yields_no_samples(self):
        config = _merge_config()
        assert config.views is not None
        config.views[0].operations[0].params = {"class_remap": {"absent": "Car"}, "target": ["Person", "Car"]}
        config.views[1].operations[0].params = {"class_remap": {"absent": "Car"}, "target": ["Person", "Car"]}
        built = build_od_dataset(resolve_source("merged", config), dataset_metadata=DatasetMetadata())
        assert built.samples == ()


@pytest.mark.required
class TestImageReference:
    """An export references the file on disk only where the file still matches the boxes."""

    def test_a_path_backed_source_is_referenced_not_re_encoded(self, tmp_path):
        """Referencing the file on disk is what makes exporting a large corpus affordable."""
        sample = _built_from_disk(tmp_path).samples[0]
        assert sample.path_or_uri == str(tmp_path / "on_disk_0.png")
        assert sample.image_bytes is None

    def test_a_view_that_leaves_the_imagery_alone_still_references_the_file(self, tmp_path):
        operations = [ViewOperation(type="Limit", params={"size": 1})]
        sample = _built_from_disk(tmp_path, operations=operations).samples[0]
        assert sample.path_or_uri == str(tmp_path / "on_disk_0.png")
        assert sample.image_bytes is None

    def test_a_view_that_transforms_the_imagery_stops_referencing_the_file(self, tmp_path):
        """A crop rewrites the boxes, so the untransformed file would hold the wrong pixels."""
        operations = [ViewOperation(type="Crop", params={"region": (0, 0, 4, 4)})]
        sample = _built_from_disk(tmp_path, operations=operations).samples[0]
        assert sample.path_or_uri is None
        assert sample.image_bytes is not None
        assert (sample.width, sample.height) == (4, 4)
        assert sample.detections[0].bbox == (1.0, 2.0, 3.0, 2.0)

    def test_a_view_that_selects_channels_stops_referencing_the_file(self, tmp_path):
        """datamaite decodes every referenced file as 3-channel RGB, so one band is not it."""
        import cv2

        operations = [ViewOperation(type="SelectChannels", params={"channels": [0]})]
        sample = _built_from_disk(tmp_path, operations=operations).samples[0]
        assert sample.path_or_uri is None
        assert sample.image_bytes is not None
        decoded = cv2.imdecode(np.frombuffer(sample.image_bytes, dtype=np.uint8), cv2.IMREAD_UNCHANGED)
        assert decoded.shape == (8, 8)

    def test_imagery_cv2_cannot_encode_is_refused(self, tmp_path):
        """A two-channel PNG raises inside OpenCV, so name the count rather than pass it on."""
        operations = [ViewOperation(type="SelectChannels", params={"channels": [0, 1]})]
        with pytest.raises(ValueError, match="2 channels"):
            _built_from_disk(tmp_path, operations=operations)

    def test_a_source_that_declares_no_size_still_references_the_file(self, tmp_path):
        """Fall back to the datum metadata, which datamaite fills in from the image itself."""
        sample = _built_from_disk(tmp_path, declare_size=False).samples[0]
        assert sample.path_or_uri == str(tmp_path / "on_disk_0.png")
        assert (sample.width, sample.height) == (8, 8)

    def test_a_source_datamaite_never_loaded_carries_its_pixels(self):
        built = build_od_dataset(resolve_source("plain", _plain_config()), dataset_metadata=DatasetMetadata())
        assert [s.file_name for s in built.samples] == ["00000000.png", "00000001.png"]
        assert all(s.path_or_uri is None and s.image_bytes is not None for s in built.samples)
        assert (built.samples[0].width, built.samples[0].height) == (8, 8)
        assert built.samples[0].detections[0].bbox == (1.0, 2.0, 3.0, 4.0)

    def test_splits_survive_the_export(self, tmp_path):
        """A writer resolves `sample.split or default_split`, so dropping it leaks val into train."""
        built = _built_from_disk(tmp_path, splits=("train", "val"))
        assert [s.split for s in built.samples] == ["train", "val"]

    def test_imagery_that_is_not_8_bit_is_refused(self, tmp_path):
        """cv2 casts silently, so a normalized view would otherwise export as black."""
        operations = [ViewOperation(type="Resize", params={"size": 16})]
        with pytest.raises(ValueError, match="8-bit"):
            _built_from_disk(tmp_path, operations=operations)


@pytest.mark.required
class TestTaxonomy:
    """The taxonomy an export attaches describes the corpus's own vocabulary."""

    def test_taxonomy_comes_from_the_conformed_vocabulary(self):
        built = build_od_dataset(resolve_source("merged", _merge_config()), dataset_metadata=DatasetMetadata())
        taxonomy = built.dataset_metadata.taxonomy
        assert taxonomy is not None
        assert list(taxonomy.ordered_names) == ["Person", "Car", "Truck"]
        assert taxonomy.id_density == "dense"

    def test_sparse_source_ids_are_reported_as_sparse(self):
        """A source read straight from COCO keeps that format's gapped ids."""
        built = build_od_dataset(resolve_source("sparse", _sparse_config()), dataset_metadata=DatasetMetadata())
        taxonomy = built.dataset_metadata.taxonomy
        assert taxonomy is not None
        assert taxonomy.id_density == "sparse"
        assert [e.source_id for e in taxonomy.entries] == [1, 17, 90]
        assert taxonomy.ordered_names == ("person", "cat", "toothbrush")
        assert built.samples[0].detections[0].category_name == "cat"

    def test_a_taxonomy_already_set_is_kept(self):
        declared = Taxonomy(entries=(CategoryEntry(source_id=7, name="Declared"),), source_dataset="upstream")
        built = build_od_dataset(
            resolve_source("merged", _merge_config()), dataset_metadata=DatasetMetadata(taxonomy=declared)
        )
        assert built.dataset_metadata.taxonomy is declared


@pytest.mark.required
class TestExportProvenance:
    """The provenance an export carries names every operand behind the corpus."""

    def test_every_operand_is_described(self):
        info = export_provenance(resolve_source("merged", _merge_config())).info
        assert info is not None
        assert [op["source"] for op in info["operands"]] == ["a", "b"]
        assert [op["view"] for op in info["operands"]] == ["conform_a", "conform_b"]

    def test_an_operand_that_conformed_nothing_carries_an_empty_remap(self):
        """Operands that already share a vocabulary are merged without any Relabel."""
        config = PipelineConfig(
            datasets=[
                DatasetProtocolConfig(name="ds_a", dataset=_od_dataset("a", ["car", "van"], 0)),
                DatasetProtocolConfig(name="ds_b", dataset=_od_dataset("b", ["car", "van"], 1)),
            ],
            sources=[
                SourceConfig(name="a", dataset="ds_a"),
                SourceConfig(name="b", dataset="ds_b"),
                SourceConfig(name="merged", merge=["a", "b"]),
            ],
        )
        info = export_provenance(resolve_source("merged", config)).info
        assert info is not None
        assert [op["class_remap"] for op in info["operands"]] == [{}, {}]
        assert [op["view"] for op in info["operands"]] == [None, None]
        assert info["label_space"] == []

    def test_a_merge_view_that_conforms_again_is_recorded_under_the_source(self):
        """A merged source's own Relabel belongs to the corpus, not to any one operand."""
        config = _merge_config()
        assert config.views is not None
        assert config.sources is not None
        config.views.append(  # type: ignore[reportAttributeAccessIssue]
            ViewConfig(
                name="coarsen",
                operations=[
                    ViewOperation(type="Relabel", params={"class_remap": {"Truck": "Car"}, "target": ["Person", "Car"]})
                ],
            )
        )
        config.sources[2] = SourceConfig(name="merged", merge=["a", "b"], view="coarsen")  # type: ignore[reportIndexIssue]
        info = export_provenance(resolve_source("merged", config)).info
        assert info is not None
        assert [record["source"] for record in info["label_space"]] == ["a", "b", "merged"]
        # The merge's own record names no operand, so it stays out of the operand entries.
        assert [op["class_remap"] for op in info["operands"]] == [{"car": "Car"}, {"lorry": "Truck"}]


@pytest.mark.required
class TestWriteExport:
    """An export writes the corpus and the provenance that produced it."""

    def _write(
        self,
        tmp_path: Path,
        mode: Literal["error", "replace", "append"] = "error",
        fmt: Literal["coco", "yolo", "huggingface_vision", "visdrone"] = "coco",
    ) -> Path:
        return write_export(
            ExportConfig(name="corpus", source="merged", format=fmt, mode=mode),
            _merge_config(),
            tmp_path,
        )

    def _info(self, tmp_path: Path) -> dict:
        return json.loads((tmp_path / "corpus" / "annotations" / "instances.json").read_text())["info"]

    def test_writes_under_the_export_name(self, tmp_path: Path):
        assert self._write(tmp_path) == tmp_path / "corpus"
        assert (tmp_path / "corpus" / "annotations" / "instances.json").is_file()

    def test_coco_holds_every_image_and_annotation(self, tmp_path: Path):
        self._write(tmp_path)
        written = json.loads((tmp_path / "corpus" / "annotations" / "instances.json").read_text())
        assert len(written["images"]) == 4
        assert len(written["annotations"]) == 4
        assert [c["name"] for c in written["categories"]] == ["Person", "Car", "Truck"]

    def test_images_do_not_collide_across_operands(self, tmp_path: Path):
        self._write(tmp_path)
        assert len(list((tmp_path / "corpus").rglob("*.png"))) == 4

    def test_provenance_lands_in_the_coco_info_block(self, tmp_path: Path):
        self._write(tmp_path)
        info = self._info(tmp_path)
        assert info["tool"] == "dataeval-flow"
        assert info["source"] == "merged"
        assert [op["dataset"] for op in info["operands"]] == ["ds_a", "ds_b"]
        assert info["operands"][0]["class_remap"] == {"car": "Car"}
        assert info["label_space"][0]["digest"]

    def test_the_digest_matches_the_envelope(self, tmp_path: Path):
        """The emitted dataset and the run that produced it carry one identity."""
        self._write(tmp_path)
        expected = label_space_records([resolve_source("merged", _merge_config())], None)
        assert [entry["digest"] for entry in self._info(tmp_path)["label_space"]] == [r.digest for r in expected]

    def test_the_declared_ontology_reaches_the_provenance(self, tmp_path: Path):
        """An export names its own ontology, so the emitted corpus can join to the audit."""
        config = _merge_config()
        config.ontologies = [OntologyConfig(name="vehicles", concepts=[OntologyConceptConfig(id="Car", label="Car")])]
        write_export(ExportConfig(name="corpus", source="merged", ontology="vehicles"), config, tmp_path)
        info = self._info(tmp_path)
        assert info["ontology"] == "vehicles"
        assert info["ontology_digest"]
        assert info["label_space"][0]["ontology_digest"] == info["ontology_digest"]

    def test_an_unreadable_ontology_leaves_the_export_written(self):
        """The orchestrator's resolver degrades to no ontology rather than losing the export."""
        config = _merge_config()
        info = export_provenance(
            resolve_source("merged", config),
            ontology=ResolvedOntology(ontology=None, source="absent", error="not found"),
        ).info
        assert info is not None
        assert info["ontology"] is None

    def test_a_second_write_is_refused_by_default(self, tmp_path: Path):
        self._write(tmp_path)
        with pytest.raises(FileExistsError):
            self._write(tmp_path)

    def test_replace_mode_overwrites(self, tmp_path: Path):
        self._write(tmp_path)
        assert self._write(tmp_path, mode="replace") == tmp_path / "corpus"

    def test_yolo_writes_images_and_labels(self, tmp_path: Path):
        self._write(tmp_path, fmt="yolo")
        assert (tmp_path / "corpus" / "data.yaml").is_file()
        assert len(list((tmp_path / "corpus").rglob("*.txt"))) == 4

    def test_an_unknown_source_raises(self, tmp_path: Path):
        with pytest.raises(ValueError, match="Unknown source"):
            write_export(ExportConfig(name="corpus", source="absent"), _merge_config(), tmp_path)


@pytest.mark.required
class TestWriteExports:
    """Every declared export is written, and a failure is counted rather than raised."""

    def test_no_exports_writes_nothing(self, tmp_path: Path):
        assert write_exports(_merge_config(), tmp_path) == 0
        assert not (tmp_path / "datasets").exists()

    def test_each_export_lands_under_its_own_name(self, tmp_path: Path):
        config = _merge_config()
        config.exports = [
            ExportConfig(name="corpus", source="merged"),
            ExportConfig(name="just_a", source="a"),
        ]
        assert write_exports(config, tmp_path) == 0
        assert (tmp_path / "datasets" / "corpus" / "annotations" / "instances.json").is_file()
        assert (tmp_path / "datasets" / "just_a" / "annotations" / "instances.json").is_file()

    def test_one_failure_does_not_cost_the_others(self, tmp_path: Path, caplog):
        config = _merge_config()
        config.exports = [
            ExportConfig(name="broken", source="absent"),
            ExportConfig(name="corpus", source="merged"),
        ]
        with caplog.at_level(logging.ERROR):
            assert write_exports(config, tmp_path) == 1
        assert "broken" in caplog.text
        assert (tmp_path / "datasets" / "corpus" / "annotations" / "instances.json").is_file()

    def test_a_refused_overwrite_is_counted_not_raised(self, tmp_path: Path):
        """FileExistsError is an OSError, so the run keeps going and reports the count."""
        config = _merge_config()
        config.exports = [ExportConfig(name="corpus", source="merged")]
        assert write_exports(config, tmp_path) == 0
        assert write_exports(config, tmp_path) == 1


@pytest.mark.required
class TestProvenanceSidecar:
    """Every format carries the provenance, not only the one that embeds it."""

    def _runs(self, tmp_path: Path) -> list:
        return json.loads((tmp_path / "corpus" / "provenance.json").read_text())["runs"]

    @pytest.mark.parametrize("fmt", ["coco", "yolo", "huggingface_vision", "visdrone"])
    def test_every_format_writes_a_readable_sidecar(self, tmp_path: Path, fmt):
        """Only COCO embeds `info`, so the sidecar is what makes the corpus joinable."""
        write_export(ExportConfig(name="corpus", source="merged", format=fmt), _merge_config(), tmp_path)
        runs = self._runs(tmp_path)
        assert len(runs) == 1
        expected = label_space_records([resolve_source("merged", _merge_config())], None)
        assert [entry["digest"] for entry in runs[0]["label_space"]] == [r.digest for r in expected]
        assert runs[0]["tool"] == "dataeval-flow"
        assert [op["dataset"] for op in runs[0]["operands"]] == ["ds_a", "ds_b"]

    def test_the_sidecar_and_the_coco_info_block_agree(self, tmp_path: Path):
        """Only the sidecar gains the wrapper. The embedded block stays flat."""
        write_export(ExportConfig(name="corpus", source="merged"), _merge_config(), tmp_path)
        written = json.loads((tmp_path / "corpus" / "annotations" / "instances.json").read_text())
        assert self._runs(tmp_path)[0] == written["info"]

    def test_append_records_every_write_in_order(self, tmp_path: Path):
        """An appended corpus holds both writes, so its provenance has to describe both."""
        write_export(ExportConfig(name="corpus", source="merged"), _merge_config(), tmp_path)
        write_export(ExportConfig(name="corpus", source="a", mode="append"), _merge_config(), tmp_path)
        assert [run["source"] for run in self._runs(tmp_path)] == ["merged", "a"]

    def test_an_unreadable_sidecar_is_replaced_with_a_warning(self, tmp_path: Path, caplog):
        """Do not fail the export over the sidecar, and do not drop history in silence."""
        write_export(ExportConfig(name="corpus", source="merged"), _merge_config(), tmp_path)
        (tmp_path / "corpus" / "provenance.json").write_text("not json")
        with caplog.at_level(logging.WARNING):
            write_export(ExportConfig(name="corpus", source="a", mode="append"), _merge_config(), tmp_path)
        assert "Could not read" in caplog.text
        assert [run["source"] for run in self._runs(tmp_path)] == ["a"]

    def test_a_sidecar_of_the_wrong_shape_is_replaced_with_a_warning(self, tmp_path: Path, caplog):
        write_export(ExportConfig(name="corpus", source="merged"), _merge_config(), tmp_path)
        (tmp_path / "corpus" / "provenance.json").write_text('{"tool": "something else"}')
        with caplog.at_level(logging.WARNING):
            write_export(ExportConfig(name="corpus", source="a", mode="append"), _merge_config(), tmp_path)
        assert "records no 'runs' list" in caplog.text
        assert [run["source"] for run in self._runs(tmp_path)] == ["a"]

    def test_a_failed_write_leaves_no_sidecar(self, tmp_path: Path):
        """Written after the corpus, so nothing describes a dataset that is not there."""
        config = _merge_config()
        with (
            patch("datamaite.write", side_effect=RuntimeError("writer said no")),
            pytest.raises(RuntimeError, match="writer said no"),
        ):
            write_export(ExportConfig(name="corpus", source="merged"), config, tmp_path)
        assert not (tmp_path / "corpus" / "provenance.json").exists()


@pytest.mark.required
class TestExportFailureHandling:
    """A config typo is a counted failure, not a traceback that costs the run its exit code."""

    def test_a_type_error_from_a_bad_relabel_is_counted(self, tmp_path: Path):
        config = _merge_config()
        assert config.views is not None
        config.views[0].operations[0].params = {"class_remap": {"car": "Car"}, "target": 5}
        config.exports = [ExportConfig(name="corpus", source="merged")]
        assert write_exports(config, tmp_path) == 1

    def test_an_attribute_error_from_the_wrong_task_is_counted(self, tmp_path: Path):
        """A classification source has no detections to write."""
        config = _merge_config()
        config.exports = [ExportConfig(name="corpus", source="merged")]
        with patch.object(export_module, "build_od_dataset", side_effect=AttributeError("no boxes")):
            assert write_exports(config, tmp_path) == 1

    def test_replace_mode_says_the_previous_corpus_is_gone(self, tmp_path: Path, caplog):
        """The destination is emptied before the write and is not restored."""
        config = _merge_config()
        config.exports = [ExportConfig(name="corpus", source="merged", mode="replace")]
        assert write_exports(config, tmp_path) == 0
        with (
            patch("datamaite.write", side_effect=RuntimeError("writer said no")),
            caplog.at_level(logging.ERROR),
        ):
            assert write_exports(config, tmp_path) == 1
        assert "does not restore it" in caplog.text

    def test_a_write_that_cleared_nothing_says_nothing(self, tmp_path: Path, caplog):
        config = _merge_config()
        config.exports = [ExportConfig(name="corpus", source="merged", mode="replace")]
        with (
            patch("datamaite.write", side_effect=RuntimeError("writer said no")),
            caplog.at_level(logging.ERROR),
        ):
            assert write_exports(config, tmp_path) == 1
        assert "does not restore it" not in caplog.text


@pytest.mark.required
class TestOccupiedDestination:
    """A destination that cannot be written to is refused before the corpus is built."""

    def test_the_refusal_comes_before_the_corpus_is_built(self, tmp_path: Path):
        """Refuse up front rather than after decoding every image the write throws away."""
        (tmp_path / "corpus").mkdir()
        (tmp_path / "corpus" / "stale.json").write_text("{}")
        with (
            patch.object(export_module, "build_od_dataset") as build,
            pytest.raises(FileExistsError, match="already exists and is not empty"),
        ):
            write_export(ExportConfig(name="corpus", source="merged"), _merge_config(), tmp_path)
        build.assert_not_called()

    def test_an_empty_destination_is_not_refused(self, tmp_path: Path):
        (tmp_path / "corpus").mkdir()
        assert write_export(ExportConfig(name="corpus", source="merged"), _merge_config(), tmp_path)

    def test_append_writes_into_an_occupied_destination(self, tmp_path: Path):
        (tmp_path / "corpus").mkdir()
        (tmp_path / "corpus" / "stale.json").write_text("{}")
        write_export(ExportConfig(name="corpus", source="merged", mode="append"), _merge_config(), tmp_path)
        assert (tmp_path / "corpus" / "provenance.json").is_file()


@pytest.mark.required
class TestOntologyDivergenceWarning:
    """An export written without the ontology a workflow declares carries a different identity."""

    def _config_with_workflow_ontology(self) -> PipelineConfig:
        config = _merge_config()
        config.ontologies = [OntologyConfig(name="vehicles", concepts=[OntologyConceptConfig(id="Car", label="Car")])]
        config.workflows = [DataCoverageWorkflowConfig(name="audit", ontology="vehicles")]
        return config

    def test_it_warns_and_names_the_workflow(self, tmp_path: Path, caplog):
        with caplog.at_level(logging.WARNING):
            write_export(ExportConfig(name="corpus", source="merged"), self._config_with_workflow_ontology(), tmp_path)
        assert "declares no ontology" in caplog.text
        assert "audit" in caplog.text

    def test_it_is_silent_when_the_export_declares_one(self, tmp_path: Path, caplog):
        config = self._config_with_workflow_ontology()
        with caplog.at_level(logging.WARNING):
            write_export(ExportConfig(name="corpus", source="merged", ontology="vehicles"), config, tmp_path)
        assert "declares no ontology" not in caplog.text

    def test_it_is_silent_when_no_workflow_declares_one(self, tmp_path: Path, caplog):
        config = _merge_config()
        config.workflows = [DataCoverageWorkflowConfig(name="plain")]
        with caplog.at_level(logging.WARNING):
            write_export(ExportConfig(name="corpus", source="merged"), config, tmp_path)
        assert "declares no ontology" not in caplog.text

    def test_declaring_the_ontology_makes_the_digests_match(self, tmp_path: Path):
        """The warning points at a real divergence: the digests differ without it."""
        config = self._config_with_workflow_ontology()
        write_export(ExportConfig(name="bare", source="merged"), config, tmp_path)
        write_export(ExportConfig(name="named", source="merged", ontology="vehicles"), config, tmp_path)
        bare = json.loads((tmp_path / "bare" / "provenance.json").read_text())["runs"][0]
        named = json.loads((tmp_path / "named" / "provenance.json").read_text())["runs"][0]
        assert bare["label_space"][0]["digest"] != named["label_space"][0]["digest"]
