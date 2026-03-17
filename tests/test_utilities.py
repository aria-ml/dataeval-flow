"""Tests for utility modules — embeddings, metadata, selection."""

from unittest.mock import MagicMock, patch

import pytest

import dataeval_flow.embeddings
import dataeval_flow.metadata
import dataeval_flow.selection  # noqa: F401
from dataeval_flow.config import (
    BoVWExtractorConfig,
    FlattenExtractorConfig,
    OnnxExtractorConfig,
    SelectionStep,
    TorchExtractorConfig,
)

# ---------------------------------------------------------------------------
# build_embeddings
# ---------------------------------------------------------------------------


class TestBuildEmbeddings:
    @patch("dataeval_flow.embeddings.Embeddings")
    @patch("dataeval_flow.embeddings.OnnxExtractor")
    def test_basic(self, mock_extractor_cls: MagicMock, mock_embed_cls: MagicMock):
        from dataeval_flow.embeddings import build_embeddings

        mock_dataset = MagicMock()
        mock_extractor = MagicMock()
        mock_extractor_cls.return_value = mock_extractor
        mock_embeddings = MagicMock()
        mock_embed_cls.return_value = mock_embeddings

        config = OnnxExtractorConfig(name="test", model_path="/model.onnx", output_name="layer4")
        result = build_embeddings(mock_dataset, config)

        mock_extractor_cls.assert_called_once_with("/model.onnx", transforms=None, output_name="layer4", flatten=True)
        mock_embed_cls.assert_called_once_with(mock_dataset, extractor=mock_extractor, batch_size=None)
        assert result is mock_embeddings

    @patch("dataeval_flow.embeddings.Embeddings")
    @patch("dataeval_flow.embeddings.OnnxExtractor")
    def test_with_transforms(self, mock_extractor_cls: MagicMock, mock_embed_cls: MagicMock):  # noqa: ARG002
        from dataeval_flow.embeddings import build_embeddings

        mock_transforms = MagicMock()
        config = OnnxExtractorConfig(name="test", model_path="/model.onnx")
        build_embeddings(MagicMock(), config, transforms=mock_transforms)

        call_kwargs = mock_extractor_cls.call_args[1]
        assert call_kwargs["transforms"] is mock_transforms

    @patch("dataeval_flow.embeddings.Embeddings")
    @patch("dataeval_flow.embeddings.FlattenExtractor")
    def test_flatten_extractor(self, mock_flatten_cls: MagicMock, mock_embed_cls: MagicMock):
        """FlattenExtractorConfig creates FlattenExtractor."""
        from dataeval_flow.embeddings import build_embeddings

        mock_flatten = MagicMock()
        mock_flatten_cls.return_value = mock_flatten

        config = FlattenExtractorConfig(name="test")
        build_embeddings(MagicMock(), config)

        mock_flatten_cls.assert_called_once_with()
        mock_embed_cls.assert_called_once()

    @patch("dataeval_flow.embeddings.Embeddings")
    @patch("dataeval_flow.embeddings.BoVWExtractor")
    def test_bovw_extractor(self, mock_bovw_cls: MagicMock, mock_embed_cls: MagicMock):
        """BoVWExtractorConfig creates BoVWExtractor and calls fit()."""
        from dataeval_flow.embeddings import build_embeddings

        mock_bovw = MagicMock()
        mock_bovw_cls.return_value = mock_bovw
        mock_dataset = MagicMock()

        config = BoVWExtractorConfig(name="test", vocab_size=1024)
        build_embeddings(mock_dataset, config)

        mock_bovw_cls.assert_called_once_with(vocab_size=1024)
        mock_embed_cls.assert_called_once()

    def test_unsupported_extractor_raises(self):
        """Unsupported extractor type raises ValueError."""
        from dataeval_flow.embeddings import build_embeddings

        config = TorchExtractorConfig(name="test", model_path="/model.pt")
        with pytest.raises(ValueError, match="not yet implemented"):
            build_embeddings(MagicMock(), config)


# ---------------------------------------------------------------------------
# build_metadata
# ---------------------------------------------------------------------------


class TestBuildMetadata:
    @patch("dataeval_flow.metadata.Metadata")
    def test_basic(self, mock_meta_cls: MagicMock):
        from dataeval_flow.metadata import build_metadata

        mock_dataset = MagicMock()
        mock_metadata = MagicMock()
        mock_meta_cls.return_value = mock_metadata

        result = build_metadata(mock_dataset)

        mock_meta_cls.assert_called_once_with(mock_dataset)
        assert result is mock_metadata

    @patch("dataeval_flow.metadata.Metadata")
    def test_with_all_kwargs(self, mock_meta_cls: MagicMock):
        from dataeval_flow.metadata import build_metadata

        build_metadata(
            MagicMock(),
            auto_bin_method="uniform_width",
            exclude=["col_a"],
            continuous_factor_bins={"col_b": [0.0, 0.5, 1.0]},
        )

        call_kwargs = mock_meta_cls.call_args[1]
        assert call_kwargs["auto_bin_method"] == "uniform_width"
        assert call_kwargs["exclude"] == ["col_a"]
        assert call_kwargs["continuous_factor_bins"] == {"col_b": [0.0, 0.5, 1.0]}

    @patch("dataeval_flow.metadata.Metadata")
    def test_skips_none_kwargs(self, mock_meta_cls: MagicMock):
        from dataeval_flow.metadata import build_metadata

        build_metadata(MagicMock(), auto_bin_method=None, exclude=None)

        call_kwargs = mock_meta_cls.call_args[1]
        assert "auto_bin_method" not in call_kwargs
        assert "exclude" not in call_kwargs


# ---------------------------------------------------------------------------
# build_selection
# ---------------------------------------------------------------------------


class TestBuildSelection:
    @patch("dataeval_flow.selection.Select")
    @patch("dataeval_flow.selection.sel")
    def test_single_step(self, mock_sel_module: MagicMock, mock_select_cls: MagicMock):
        from dataeval_flow.selection import build_selection

        mock_limit_cls = MagicMock()
        mock_limit_instance = MagicMock()
        mock_limit_cls.return_value = mock_limit_instance
        mock_sel_module.Limit = mock_limit_cls

        mock_dataset = MagicMock()
        steps = [SelectionStep(type="Limit", params={"size": 100})]

        build_selection(mock_dataset, steps)

        mock_limit_cls.assert_called_once_with(size=100)
        mock_select_cls.assert_called_once_with(mock_dataset, selections=[mock_limit_instance])

    @patch("dataeval_flow.selection.Select")
    @patch("dataeval_flow.selection.sel")
    def test_multiple_steps(self, mock_sel_module: MagicMock, mock_select_cls: MagicMock):
        from dataeval_flow.selection import build_selection

        mock_limit = MagicMock()
        mock_shuffle = MagicMock()
        mock_sel_module.Limit.return_value = mock_limit
        mock_sel_module.Shuffle.return_value = mock_shuffle

        steps = [
            SelectionStep(type="Limit", params={"size": 50}),
            SelectionStep(type="Shuffle", params={}),
        ]

        build_selection(MagicMock(), steps)

        selections = mock_select_cls.call_args[1]["selections"]
        assert len(selections) == 2

    @patch("dataeval_flow.selection.Select")
    @patch("dataeval_flow.selection.sel")
    def test_step_without_params(self, mock_sel_module: MagicMock, mock_select_cls: MagicMock):  # noqa: ARG002
        from dataeval_flow.selection import build_selection

        mock_reverse = MagicMock()
        mock_sel_module.Reverse.return_value = mock_reverse

        steps = [SelectionStep(type="Reverse")]
        build_selection(MagicMock(), steps)

        mock_sel_module.Reverse.assert_called_once_with()

    def test_invalid_selection_type_raises(self):
        import pytest

        from dataeval_flow.selection import build_selection

        steps = [SelectionStep(type="NonexistentSelector")]
        with pytest.raises(ValueError, match="Unknown selection type"):
            build_selection(MagicMock(), steps)
