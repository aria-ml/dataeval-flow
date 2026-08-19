"""Extracting a committable encoding descriptor from an archived result."""

import json
from pathlib import Path

import numpy as np
import pytest
from dataeval import Metadata

from dataeval_flow._encoding_cli import write_encoding
from dataeval_flow.binning import describe_binning, descriptor_from_record, write_descriptor

pytestmark = pytest.mark.filterwarnings("ignore::UserWarning")


def _metadata(**kwargs) -> Metadata:
    rng = np.random.default_rng(0)
    return Metadata.from_factors(
        {"temp_c": rng.normal(20.0, 3.0, 200), "weather": rng.choice(["sun", "rain"], 200)}, **kwargs
    )


def _result_file(tmp_path: Path, **tasks: dict) -> Path:
    path = tmp_path / "result.json"
    path.write_text(
        json.dumps({name: {"metadata": {"metadata_binning": record}} for name, record in tasks.items()}),
        encoding="utf-8",
    )
    return path


class TestDescriptorFromRecord:
    def test_matches_what_dataeval_writes(self, tmp_path: Path):
        """The descriptor is DataEval's, carried through the envelope untouched.

        Byte-identity is the contract: a descriptor written from a result has to be the
        same artifact `export_encoding` produces, or the loop does not close.
        """
        md = _metadata(continuous_factor_bins={"temp_c": [-np.inf, 0.0, 10.0, np.inf]})
        md.export_encoding(tmp_path / "upstream.json")
        write_descriptor(describe_binning(md), tmp_path / "flow.json")

        assert (tmp_path / "flow.json").read_bytes() == (tmp_path / "upstream.json").read_bytes()

    def test_carries_the_format_version_rather_than_assuming_one(self):
        """The number belongs to DataEval, so it is read off what DataEval just wrote."""
        record = describe_binning(_metadata())
        assert descriptor_from_record(record)["version"] == record["descriptor_version"]

    def test_a_record_with_no_encodings_is_refused(self):
        with pytest.raises(ValueError, match="records no encodings"):
            descriptor_from_record({"factors": {}})

    def test_splits_encoded_alike_give_one_descriptor(self):
        declared = {"temp_c": [-np.inf, 0.0, np.inf]}
        record = {
            "per_split": {
                "train": describe_binning(_metadata(continuous_factor_bins=declared)),
                "test": describe_binning(_metadata(continuous_factor_bins=declared)),
            }
        }
        assert "temp_c" in descriptor_from_record(record)["factors"]

    def test_a_grown_vocabulary_still_writes_one_descriptor(self):
        """The state `reference_split` produces, so refusing it makes the artifact unreachable.

        Appending leaves every shared code meaning what it meant, so the widest vocabulary
        describes the run — and it is what a user should commit.
        """
        record = {
            "per_split": {
                "train": {"factors": {"w": {"encoding": {"kind": "levels", "levels": ["a", "b"]}}}},
                "test": {"factors": {"w": {"encoding": {"kind": "levels", "levels": ["a", "b", "c"]}}}},
            }
        }
        assert descriptor_from_record(record)["factors"]["w"]["levels"] == ["a", "b", "c"]

    def test_a_record_with_no_splits_raises_value_error(self):
        """Both callers guard on ValueError, so a StopIteration escapes as a traceback."""
        with pytest.raises(ValueError, match="no splits"):
            descriptor_from_record({"per_split": {}})

    def test_splits_encoded_differently_are_refused(self):
        """Picking one would hand somebody a policy nobody chose."""
        record = {
            "per_split": {
                "train": describe_binning(_metadata(continuous_factor_bins={"temp_c": [-np.inf, 0.0, np.inf]})),
                "test": describe_binning(_metadata(continuous_factor_bins={"temp_c": [-np.inf, 5.0, np.inf]})),
            }
        }
        with pytest.raises(ValueError, match=r"encode \['temp_c'\] differently"):
            descriptor_from_record(record)


class TestSelectingAcrossTasks:
    """A result holds one entry per task, and they need not all record an encoding."""

    def test_a_task_without_an_encoding_does_not_hide_one_that_has_it(self, tmp_path: Path):
        encoded = {"factors": {"w": {"encoding": {"kind": "levels", "levels": ["a"]}}}}
        result = tmp_path / "result.json"
        result.write_text(
            json.dumps(
                {
                    "good": {"metadata": {"metadata_binning": encoded}},
                    "no_encoding": {"metadata": {"metadata_binning": {"factors": {"w": {}}}}},
                }
            ),
            encoding="utf-8",
        )
        out = tmp_path / "encoding.json"
        assert write_encoding(result, out) == 0
        assert json.loads(out.read_text())["factors"]["w"]["levels"] == ["a"]


class TestWriteEncodingCommand:
    def test_writes_a_descriptor_a_policy_can_read_back(self, tmp_path: Path):
        """The loop closes inside flow: run, extract, commit, reference."""
        from dataeval_flow.config._models import PipelineConfig
        from dataeval_flow.policy import resolve_policy
        from dataeval_flow.workflow.base import MetadataConfigMixin

        md = _metadata(continuous_factor_bins={"temp_c": [-np.inf, 0.0, 10.0, np.inf]})
        result = _result_file(tmp_path, coverage_check=describe_binning(md))

        assert write_encoding(result, tmp_path / "policy" / "bins.json") == 0

        config = PipelineConfig.model_validate({"metadata": [{"name": "std", "encoding": "policy/bins.json"}]})
        policy = resolve_policy(MetadataConfigMixin(metadata="std"), config, tmp_path)
        assert set(policy.encoding or {}) == {"temp_c", "weather"}

    def test_prints_when_no_output_is_given(self, tmp_path: Path, capsys):
        result = _result_file(tmp_path, run=describe_binning(_metadata()))
        assert write_encoding(result) == 0
        assert json.loads(capsys.readouterr().out)["factors"]

    def test_creates_the_parent_directory(self, tmp_path: Path):
        result = _result_file(tmp_path, run=describe_binning(_metadata()))
        assert write_encoding(result, tmp_path / "deep" / "nested" / "bins.json") == 0
        assert (tmp_path / "deep" / "nested" / "bins.json").exists()

    def test_one_of_several_tasks_by_name(self, tmp_path: Path):
        result = _result_file(
            tmp_path,
            a=describe_binning(_metadata(continuous_factor_bins={"temp_c": [-np.inf, 0.0, np.inf]})),
            b=describe_binning(_metadata(continuous_factor_bins={"temp_c": [-np.inf, 5.0, np.inf]})),
        )
        assert write_encoding(result, tmp_path / "bins.json", task="b") == 0
        edges = json.loads((tmp_path / "bins.json").read_text())["factors"]["temp_c"]["edges"]
        assert edges == ["-inf", 5.0, "inf"]

    def test_tasks_that_agree_need_no_task_name(self, tmp_path: Path):
        declared = {"temp_c": [-np.inf, 0.0, np.inf]}
        result = _result_file(
            tmp_path,
            a=describe_binning(_metadata(continuous_factor_bins=declared)),
            b=describe_binning(_metadata(continuous_factor_bins=declared)),
        )
        assert write_encoding(result, tmp_path / "bins.json") == 0

    def test_tasks_that_differ_ask_which(self, tmp_path: Path, caplog):
        result = _result_file(
            tmp_path,
            a=describe_binning(_metadata(continuous_factor_bins={"temp_c": [-np.inf, 0.0, np.inf]})),
            b=describe_binning(_metadata(continuous_factor_bins={"temp_c": [-np.inf, 5.0, np.inf]})),
        )
        assert write_encoding(result, tmp_path / "bins.json") == 1
        assert "--task" in caplog.text
        assert not (tmp_path / "bins.json").exists()

    def test_an_unknown_task_lists_what_is_there(self, tmp_path: Path, caplog):
        result = _result_file(tmp_path, run=describe_binning(_metadata()))
        assert write_encoding(result, tmp_path / "bins.json", task="nope") == 1
        assert "run" in caplog.text

    def test_a_result_with_no_metadata_says_so(self, tmp_path: Path, caplog):
        path = tmp_path / "result.json"
        path.write_text(json.dumps({"split": {"metadata": {}}}), encoding="utf-8")
        assert write_encoding(path) == 1
        assert "records no encodings" in caplog.text

    def test_an_unreadable_file_is_reported_not_raised(self, tmp_path: Path, caplog):
        (tmp_path / "broken.json").write_text("{not json", encoding="utf-8")
        assert write_encoding(tmp_path / "broken.json") == 1
        assert "Cannot read" in caplog.text
