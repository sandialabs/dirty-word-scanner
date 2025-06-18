import unittest

import numpy as np

from src.opencsp_sensitive_strings.image_conversion import numpy_to_image


class TestImageConversion(unittest.TestCase):
    def test_numpy_to_image_truncate(self) -> None:
        arr8i = np.array([[0, 125, 255]]).astype(np.int8)
        arr16i = np.array([[0, 8192, 16384]]).astype(np.int16)
        arr8f = arr8i.astype(np.float16)
        arr16f = arr16i.astype(np.float16)
        im8i = numpy_to_image(arr8i, rescale_or_clip="truncate")
        im16i = numpy_to_image(arr16i, rescale_or_clip="truncate")
        im8f = numpy_to_image(arr8f, rescale_or_clip="truncate")
        im16f = numpy_to_image(arr16f, rescale_or_clip="truncate")
        np.testing.assert_array_equal(
            np.asarray(im8i), np.array([[0, 125, 255]])
        )
        np.testing.assert_array_equal(
            np.asarray(im16i), np.array([[0, 255, 255]])
        )
        np.testing.assert_array_equal(
            np.asarray(im8f), np.array([[0, 125, 255]])
        )
        np.testing.assert_array_equal(
            np.asarray(im16f), np.array([[0, 255, 255]])
        )

    def test_numpy_to_image_rescale(self) -> None:
        arr8i = np.array([[0, 125, 255]]).astype(np.int8)
        arr16i = np.array([[0, 8192, 16384]]).astype(np.int16)
        arr8f = arr8i.astype(np.float16)
        arr16f = arr16i.astype(np.float16)
        im8i = numpy_to_image(arr8i, rescale_or_clip="rescale")
        im16i = numpy_to_image(arr16i, rescale_or_clip="rescale")
        im8f = numpy_to_image(arr8f, rescale_or_clip="rescale")
        im16f = numpy_to_image(arr16f, rescale_or_clip="rescale")
        np.testing.assert_array_equal(
            np.asarray(im8i), np.array([[0, 125, 255]])
        )
        np.testing.assert_array_equal(
            np.asarray(im16i), np.array([[0, 127, 255]])
        )
        np.testing.assert_array_equal(
            np.asarray(im8f), np.array([[0, 125, 255]])
        )
        np.testing.assert_array_equal(
            np.asarray(im16f), np.array([[0, 127, 255]])
        )
