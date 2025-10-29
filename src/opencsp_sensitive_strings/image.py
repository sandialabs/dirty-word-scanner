"""
Handles converting NumPy arrays to Pillow Images.
"""

from __future__ import annotations

import time
from typing import TYPE_CHECKING

import cv2
import numpy as np
from PIL import UnidentifiedImageError
from PIL.Image import Image, fromarray
from PIL.Image import open as open_image

from opencsp_sensitive_strings.user_interaction import user

if TYPE_CHECKING:
    from pathlib import Path

EIGHT_BIT_DEPTH = 255
MAX_WIDTH, MAX_HEIGHT = 1920, 1080


def is_image(file: Path) -> bool:
    """
    Is the file an image?

    Args:
        file:  The file in question.

    Returns:
        Whether the file can be handled by the Python Imaging Library
        (PIL), based on its extension.
    """
    return file.suffix.lower().lstrip(".") in [
        "apng",
        "blp",
        "bmp",
        "dds",
        "dib",
        "eps",
        "gif",
        "icns",
        "ico",
        "im",
        "jpg",
        "jpeg",
        "msp",
        "pbm",
        "pcx",
        "pgm",
        "ppm",
        "png",
        "pnm",
        "sgi",
        "spider",
        "tga",
        "tiff",
        "webp",
        "xbm",
    ]


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


def get_image(file: Path) -> Image | None:
    try:
        with open_image(file) as contents:
            np_image = np.array(contents.convert("RGB")).copy()
    except (
        FileNotFoundError,
        UnidentifiedImageError,
        ValueError,
        TypeError,
    ):
        return None
    else:
        return numpy_to_image(np_image)


def show_image(file: Path, image: Image) -> bool:
    rescaled = ""
    if image.size[0] > MAX_WIDTH or image.size[1] > MAX_HEIGHT:
        scale = min(MAX_WIDTH / image.size[0], MAX_HEIGHT / image.size[1])
        image = image.resize(
            (int(scale * image.size[0]), int(scale * image.size[1]))
        )
        rescaled = " (downscaled)"
    cv2.imshow(f"{file}{rescaled}", np.array(image))
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    time.sleep(0.1)  # small delay to prevent accidental double-bounces
    return user.approved(file)
