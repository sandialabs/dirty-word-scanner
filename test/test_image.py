import numpy as np
import pytest

from opencsp_sensitive_strings.image import numpy_to_image


@pytest.mark.parametrize(
    ("rescale", "array", "expected"),
    [
        (
            False,
            np.array([[0, 125, 255]]).astype(np.int8),
            np.array([[0, 125, 255]]),
        ),
        (
            False,
            np.array([[0, 8192, 16384]]).astype(np.int16),
            np.array([[0, 255, 255]]),
        ),
        (
            False,
            np.array([[0, 125, 255]]).astype(np.float16),
            np.array([[0, 125, 255]]),
        ),
        (
            False,
            np.array([[0, 8192, 16384]]).astype(np.float16),
            np.array([[0, 255, 255]]),
        ),
        (
            True,
            np.array([[0, 125, 255]]).astype(np.int8),
            np.array([[0, 125, 255]]),
        ),
        (
            True,
            np.array([[0, 8192, 16384]]).astype(np.int16),
            np.array([[0, 127, 255]]),
        ),
        (
            True,
            np.array([[0, 125, 255]]).astype(np.float16),
            np.array([[0, 125, 255]]),
        ),
        (
            True,
            np.array([[0, 8192, 16384]]).astype(np.float16),
            np.array([[0, 127, 255]]),
        ),
    ],
)
def test_numpy_to_image(
    rescale: bool,  # noqa: FBT001
    array: np.ndarray,
    expected: np.ndarray,
) -> None:
    im = numpy_to_image(array, rescale=rescale)
    np.testing.assert_array_equal(np.asarray(im), expected)
