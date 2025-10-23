"""
Handles extracting the contents of HDF5 files.
"""

from __future__ import annotations

from pathlib import Path

import h5py
import numpy as np

from opencsp_sensitive_strings.image import numpy_to_image

MAX_ASPECT_RATIO = 10
MIN_PIXELS = 10


def extract_hdf5_to_directory(hdf5_file: Path, destination: Path) -> Path:
    """
    Extract the given HDF5 file into the given destination directory.

    A new directory is created in the destination with the same name as
    the HDF5 file.  String values are extracted as ``.txt`` files, and
    images are extracted as ``.png`` files.  Everything else is saved
    with NumPy as ``.npy`` files.

    Parameters:
        hdf5_file:  The HDF5 file to extract.
        destination:  The directory in which to create a directory for
            the HDF5 file.

    Returns:
        The path to the newly created directory into which the HDF5
        files were extracted.
    """
    hdf5_directory = destination / hdf5_file.name
    hdf5_directory.mkdir(exist_ok=True, parents=True)
    extraction_functions = [
        _extract_string,
        _extract_images,
        _extract_other_dataset,
    ]
    for dataset in _get_datasets(hdf5_file):
        dataset.parent.mkdir(exist_ok=True, parents=True)
        value = _load_dataset_from_file(dataset, hdf5_file)
        for extract in extraction_functions:
            if extract(value, hdf5_directory / dataset):
                break
    return hdf5_directory


def _extract_string(
    value: str | float | np.ndarray | None,
    dataset: Path,
) -> bool:
    """
    Extract a string value and save it to a text file.

    Args:
        value:  The value to extract.
        dataset:  The location of the dataset.

    Returns:
        ``True`` if the value was processed as a string, otherwise
        ``False``.
    """
    if isinstance(value, str):
        with dataset.with_suffix(".txt").open("w") as output_file:
            output_file.write(value)
        return True
    return False


def _extract_images(
    value: str | float | np.ndarray | None,
    dataset: Path,
) -> bool:
    """
    Extract image data and save it as one or more PNG files.

    Args:
        value:  The value to extract, expected to be an image array.
        dataset:  The location of the dataset.

    Returns:
        ``True`` if the value was processed as an image, otherwise
        ``False``.
    """
    if not isinstance(value, np.ndarray):
        return False
    shape = value.shape
    # we assume images have 2 or 3 dimensions
    if len(shape) not in {2, 3}:
        return False
    # We assume shapes are at least 10x10 pixels and have an
    # aspect ratio of at least 10:1.
    aspect_ratio = max(shape[0], shape[1]) / min(shape[0], shape[1])
    if (shape[0] < MIN_PIXELS or shape[1] < MIN_PIXELS) or (
        aspect_ratio >= MAX_ASPECT_RATIO + 1e-3
    ):
        return False
    dataset_path = dataset.with_suffix(".png")
    np_value = np.array(value).squeeze()
    # assumed grayscale or RGB
    if len(shape) == 2 or shape[2] in [1, 3]:  # noqa: PLR2004
        image = numpy_to_image(np_value)
        image.save(dataset_path)
    else:  # assumed multiple images
        for index in range(shape[2]):
            np_single_image = np_value[:, :, index].squeeze()
            image = numpy_to_image(np_single_image)
            image.save(
                dataset_path.parent
                / f"{dataset_path.stem}_{index}{dataset_path.suffix}"
            )
    return True


def _extract_other_dataset(
    value: str | float | np.ndarray | None,
    dataset: Path,
) -> bool:
    """
    Extract other types of datasets and save them as NumPy files.

    Args:
        value:  The value to extract.
        dataset:  The location of the dataset.

    Returns:
        ``True``, because if a dataset gets this far, it will always be
        processed as an arbitrary NumPy file.
    """
    np.save(dataset.with_suffix(""), np.array(value), allow_pickle=False)
    return True


def _get_datasets(hdf5_file: Path) -> list[Path]:
    """
    Get the dataset names from a HDF5 file.

    Args:
        hdf5_file:  The HDF5 file to read.

    Returns:
        The names of (i.e., paths to) the dataset objects within the
        file.
    """
    datasets: list[Path] = []
    with h5py.File(hdf5_file, "r") as input_file:
        input_file.visititems(
            lambda name, entity: datasets.append(Path(name))
            if isinstance(entity, h5py.Dataset)
            else None
        )
    return datasets


def _load_dataset_from_file(
    dataset: Path, file: Path
) -> str | float | np.ndarray | None:
    """
    Load the requested dataset from a HDF5 file.

    Args:
        dataset:  The path to the dataset within the HDF5 file.
        file:  The HDF5 file from which to load the dataset.

    Returns:
        The loaded dataset value.
    """
    with h5py.File(file, "r") as input_file:
        data: h5py.Dataset = input_file[f"{dataset}"]
        is_scalar = np.ndim(data) == 0 and np.size(data) == 1
        is_single_element_array = np.ndim(data) > 0 and np.size(data) == 1
        is_non_empty_array = np.size(data) > 0
        return _decode_if_bytes(
            _get_scalar_value(data)
            if is_scalar
            else _get_single_element_value(data)
            if is_single_element_array
            else _get_non_empty_array(data)
            if is_non_empty_array
            else None
        )


def _decode_if_bytes(
    value: str | float | np.bytes_ | bytes | np.ndarray | None,
) -> str | float | np.ndarray | None:
    """
    Decode byte data to string if necessary.

    Args:
        value:  The value to decode.

    Returns:
        The decoded string, if the input was of a bytes type; otherwise
        the input value, unaltered.
    """
    return value.decode() if isinstance(value, (np.bytes_, bytes)) else value


def _get_scalar_value(
    data: h5py.Dataset,
) -> str | float | np.bytes_ | bytes:
    """
    Retrieve the scalar value from the dataset.

    Args:
        data:  The dataset in question.

    Returns:
        The scalar value.
    """
    return data[()]


def _get_single_element_value(
    data: h5py.Dataset,
) -> str | float | np.bytes_ | bytes:
    """
    Flatten the dataset and return the first element.

    Args:
        data:  The dataset in question.

    Returns:
        The first element.
    """
    return data[...].flatten()[0]


def _get_non_empty_array(data: h5py.Dataset) -> np.ndarray:
    """
    Squeeze the dataset to remove singleton dimensions.

    Args:
        data:  The dataset in question.

    Returns:
        The corresponding NumPy array with any extraneous dimensions
        removed.
    """
    return data[...].squeeze()
