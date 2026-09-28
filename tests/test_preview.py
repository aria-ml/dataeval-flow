"""An item's thumbnail: its pixels scaled by what their type or dataset says they span, cropped to a box, shrunk."""

import base64
import io
from types import SimpleNamespace

import numpy as np
import pytest
from PIL import Image

from dataeval_flow._blocks import ItemRef
from dataeval_flow._preview import NotAnImageError, _crop, _to_uint8, preview

pytestmark = pytest.mark.required

_REF = ItemRef(source="train", index=0)


def _decoded(image, *, ref: ItemRef = _REF, target=None, value_range=None) -> Image.Image:
    asset = preview(ref, image, target, value_range)
    assert asset.media_type == "image/webp"
    picture = Image.open(io.BytesIO(base64.b64decode(asset.data)))
    assert picture.format == "WEBP"
    assert (picture.width, picture.height) == (asset.width, asset.height)
    return picture


class TestRange:
    """Each value lands where its type or its dataset says it sits, never where the largest value would guess."""

    def test_uint8_is_read_as_it_is(self):
        array = np.array([[0, 64, 255]], dtype=np.uint8)
        assert _to_uint8(array, None) is array

    def test_an_integer_image_spans_its_type(self):
        """A dark uint16 image stays dark: 4096 of 65535 is 16 of 255, not white."""
        assert _to_uint8(np.array([0, 4096, 65535], dtype=np.uint16), None).tolist() == [0, 16, 255]
        assert _to_uint8(np.array([-128, 0, 127], dtype=np.int8), None).tolist() == [0, 128, 255]

    def test_a_float_image_spans_its_dataset_s_declared_range(self):
        assert _to_uint8(np.array([0.0, 0.5, 1.0]), (0.0, 1.0)).tolist() == [0, 128, 255]
        assert _to_uint8(np.array([0.0, 50.0, 300.0]), (0.0, 255.0)).tolist() == [0, 50, 255]

    def test_a_float_image_without_a_declared_range_spans_its_own(self):
        assert _to_uint8(np.array([0.2, 0.3, 0.4]), None).tolist() == [0, 128, 255]

    def test_a_constant_image_is_mid_grey(self):
        assert _to_uint8(np.full((2, 2), 0.7), None).tolist() == [[128, 128], [128, 128]]

    def test_a_value_that_is_not_a_number_draws_black(self):
        assert _to_uint8(np.array([0.0, np.nan, 1.0]), None).tolist() == [0, 0, 255]


class TestChannels:
    @pytest.mark.parametrize(("channels", "mode"), [(1, "L"), (3, "RGB"), (4, "RGB"), (5, "L"), (2, "L")])
    def test_one_is_grey_three_are_colour_four_drop_the_fourth_and_others_show_the_first(self, channels, mode):
        image = np.zeros((channels, 8, 8), dtype=np.uint8)
        image[0] = 200
        assert preview(_REF, image, None, None).width == 8
        picture = _decoded(image)
        # WebP stores grey as colour: a grey preview has equal channels, a colour one here a red first.
        red, green, blue = np.asarray(picture.convert("RGB"), dtype=int).mean(axis=(0, 1))
        assert red > 150
        assert (abs(green - red) < 8) == (mode == "L")

    def test_a_two_dimensional_image_is_one_channel(self):
        assert _decoded(np.full((6, 10), 255, dtype=np.uint8)).size == (10, 6)

    def test_a_tensor_like_array_reads_through_numpy(self):
        class Tensor:
            def __array__(self, dtype=None, copy=None):
                return np.zeros((3, 4, 4), dtype=np.float32)

        assert _decoded(Tensor()).size == (4, 4)


class TestSize:
    def test_the_long_side_shrinks_to_192_keeping_the_shape(self):
        assert _decoded(np.zeros((3, 480, 640), dtype=np.uint8)).size == (192, 144)

    def test_a_small_image_is_never_enlarged(self):
        assert _decoded(np.zeros((3, 16, 12), dtype=np.uint8)).size == (12, 16)


class TestNotAnImageError:
    """Only images have a previewer: anything else is refused as such, so capture can name it instead."""

    @pytest.mark.parametrize(
        "item",
        [np.zeros(128), np.zeros((1, 4, 8, 8)), {"image": 1}, "an item", np.array([["a", "b"], ["c", "d"]])],
        ids=["vector", "volume", "dict", "string", "text array"],
    )
    def test_it_is_refused_as_no_image(self, item):
        with pytest.raises(NotAnImageError):
            preview(_REF, item, None, None)


class TestBoxes:
    _IMAGE = np.zeros((3, 480, 640), dtype=np.uint8)

    @staticmethod
    def _target(*boxes: tuple[float, float, float, float]) -> SimpleNamespace:
        return SimpleNamespace(boxes=np.array(boxes, dtype=np.float32).reshape(-1, 4))

    def test_a_box_is_cropped_with_a_margin_rounded_outward(self):
        """(100.5, 50.2)–(300.7, 150.9) widens by 10% each side, to whole pixels outside it: x 80–321, y 40–161."""
        target = self._target((0, 0, 1, 1), (100.5, 50.2, 300.7, 150.9))
        assert _crop(self._IMAGE, target, 1).shape == (3, 121, 241)
        assert _decoded(self._IMAGE, ref=ItemRef(source="train", index=0, target=1), target=target).size == (192, 96)

    def test_a_box_past_the_image_s_edge_is_clamped_to_it(self):
        assert _crop(self._IMAGE, self._target((600, 400, 700, 500)), 0).shape == (3, 90, 50)

    @pytest.mark.parametrize(
        ("box", "error"),
        [((100, 50, 100, 150), "no area"), ((700, 500, 800, 600), "outside the image")],
        ids=["degenerate", "outside"],
    )
    def test_a_box_with_nothing_to_show_has_no_thumbnail(self, box, error):
        """Not the whole image instead, which would mislead about which object was flagged."""
        with pytest.raises(ValueError, match=error):
            _crop(self._IMAGE, self._target(box), 0)

    @pytest.mark.parametrize(
        ("target", "error"),
        [(SimpleNamespace(), "no boxes"), (None, "no boxes"), (SimpleNamespace(boxes=np.zeros((2, 4))), "no box 2")],
        ids=["no boxes", "no target", "out of range"],
    )
    def test_a_box_that_cannot_be_found_has_no_thumbnail(self, target, error):
        with pytest.raises(LookupError, match=error):
            preview(ItemRef(source="train", index=0, target=2), self._IMAGE, target, None)
