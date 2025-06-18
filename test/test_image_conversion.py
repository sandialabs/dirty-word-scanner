import numpy as np
import pytest

from src.opencsp_sensitive_strings.image_conversion import numpy_to_image


@pytest.mark.parametrize(
    ("mode", "array", "expected"),
    [
        (
            "truncate",
            np.array([[0, 125, 255]]).astype(np.int8),
            np.array([[0, 125, 255]]),
        ),
        (
            "truncate",
            np.array([[0, 8192, 16384]]).astype(np.int16),
            np.array([[0, 255, 255]]),
        ),
        (
            "truncate",
            np.array([[0, 125, 255]]).astype(np.float16),
            np.array([[0, 125, 255]]),
        ),
        (
            "truncate",
            np.array([[0, 8192, 16384]]).astype(np.float16),
            np.array([[0, 255, 255]]),
        ),
        (
            "rescale",
            np.array([[0, 125, 255]]).astype(np.int8),
            np.array([[0, 125, 255]]),
        ),
        (
            "rescale",
            np.array([[0, 8192, 16384]]).astype(np.int16),
            np.array([[0, 127, 255]]),
        ),
        (
            "rescale",
            np.array([[0, 125, 255]]).astype(np.float16),
            np.array([[0, 125, 255]]),
        ),
        (
            "rescale",
            np.array([[0, 8192, 16384]]).astype(np.float16),
            np.array([[0, 127, 255]]),
        ),
    ],
)
def test_numpy_to_image(
    mode: str, array: np.ndarray, expected: np.ndarray
) -> None:
    im = numpy_to_image(array, rescale_or_clip=mode)
    np.testing.assert_array_equal(np.asarray(im), expected)
