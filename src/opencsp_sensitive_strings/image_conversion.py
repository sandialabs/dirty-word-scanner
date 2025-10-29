"""
Handles converting NumPy arrays to Pillow Images.
"""

import numpy as np
from PIL.Image import Image, fromarray

EIGHT_BIT_DEPTH = 255


def numpy_to_image(array: np.ndarray, *, rescale: bool = True) -> Image:
    """
    Convert the NumPy representation of an image to a Pillow Image.

    The array is converted to an integer type, as necessary.  The color
    information is then optionally rescaled and then clipped to fit
    within an 8-bit color depth.

    Note:
        In theory, images can be saved with higher bit-depth information
        using OpenCV's ``imwrite('12-bit-image.png', array)``, but we
        haven't tried very hard and haven't had any luck getting this to
        work.

    Args:
        array:  The array to be converted.
        rescale:  Whether to rescale the value in the array to fit
            within 0-255.

    Returns:
        The image representation of the input array.
    """
    allowed_int_types: list[type] = [
        np.int8,
        np.uint8,
        np.int16,
        np.uint16,
        np.int32,
        np.uint32,
        np.int64,
        np.uint64,
    ]

    # get the current integer size, and convert to integer type
    if not np.issubdtype(array.dtype, np.integer):
        maximum = np.max(array)
        for int_type in allowed_int_types:
            if np.iinfo(int_type).max >= maximum:
                break
        array = array.astype(int_type)
    else:
        int_type = array.dtype

    # rescale down to 8-bit if bit depth is too large
    if np.iinfo(int_type).max > EIGHT_BIT_DEPTH:
        if rescale:
            scale = 255 / np.max(array)
            array = array * scale
        array = np.clip(array, 0, 255)
        array = array.astype(np.uint8)
    return fromarray(array)
