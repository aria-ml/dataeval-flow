"""Tests for custom named preprocessors."""

import torch

from dataeval_flow.preprocessors import CUSTOM_PREPROCESSORS, ToRGB, resolve_custom


class TestToRGB:
    """Test the ToRGB channel-coercion transform."""

    def test_hw_to_three_channel(self):
        """2D HW image expands to 3 channels."""
        out = ToRGB()(torch.zeros(8, 8))
        assert out.shape == (3, 8, 8)

    def test_grayscale_repeats_to_rgb(self):
        """Single-channel grayscale repeats luma across 3 channels."""
        img = torch.arange(8 * 8, dtype=torch.float32).reshape(1, 8, 8)
        out = ToRGB()(img)
        assert out.shape == (3, 8, 8)
        assert torch.equal(out[0], out[1])
        assert torch.equal(out[1], out[2])

    def test_gray_alpha_to_rgb(self):
        """2-channel (gray+alpha) coerces to 3 channels by repeating luma."""
        img = torch.stack([torch.arange(16, dtype=torch.float32).reshape(4, 4), torch.ones(4, 4)])
        out = ToRGB()(img)
        assert out.shape == (3, 4, 4)
        assert torch.equal(out[0], out[1])
        assert torch.equal(out[1], out[2])
        assert torch.equal(out[0], img[0])  # luma repeated, alpha dropped

    def test_three_channel_passthrough(self):
        """3-channel image is returned unchanged."""
        img = torch.rand(3, 8, 8)
        out = ToRGB()(img)
        assert out.shape == (3, 8, 8)
        assert torch.equal(out, img)

    def test_rgba_drops_alpha(self):
        """4-channel RGBA keeps the first three channels."""
        img = torch.rand(4, 8, 8)
        out = ToRGB()(img)
        assert out.shape == (3, 8, 8)
        assert torch.equal(out, img[:3])

    def test_repr_is_stable(self):
        """repr is deterministic for cache-key stability."""
        assert repr(ToRGB()) == "ToRGB()"


class TestRegistry:
    """Test the custom preprocessor registry and resolver."""

    def test_torgb_registered(self):
        """ToRGB is registered under its name."""
        assert CUSTOM_PREPROCESSORS["ToRGB"] is ToRGB

    def test_resolve_known_name(self):
        """resolve_custom returns the class for a registered name."""
        assert resolve_custom("ToRGB") is ToRGB

    def test_resolve_unknown_returns_none(self):
        """resolve_custom returns None for an unregistered name."""
        assert resolve_custom("Resize") is None
