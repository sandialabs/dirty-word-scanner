from pathlib import Path
from typing import Optional, Union

import h5py
import numpy as np

from opencsp_sensitive_strings.image_conversion import numpy_to_image

MAX_ASPECT_RATIO = 10
MIN_PIXELS = 10


def _get_datasets(hdf5_file: Path) -> list[Path]:
    """Get the dataset names from a HDF5 file."""
    datasets: list[Path] = []
    with h5py.File(hdf5_file, "r") as input_file:
        input_file.visititems(
            lambda name, entity: datasets.append(Path(name))
            if isinstance(entity, h5py.Dataset)
            else None
        )
    return datasets


def _get_scalar_value(
    data: h5py.Dataset,
) -> Union[str, float, int, np.bytes_, bytes]:
    """Retrieve the scalar value from the dataset."""
    return data[()]


def _get_single_element_value(
    data: h5py.Dataset,
) -> Union[str, float, int, np.bytes_, bytes]:
    """Flatten the dataset and return the first element."""
    return data[...].flatten()[0]


def _get_non_empty_array(data: h5py.Dataset) -> np.ndarray:
    """Squeeze the dataset to remove singleton dimensions."""
    return data[...].squeeze()


def _decode_if_bytes(
    values: Optional[Union[str, float, int, np.bytes_, bytes, np.ndarray]],
) -> Optional[Union[str, float, int, np.ndarray]]:
    """Decode byte data to string if necessary."""
    return value.decode() if isinstance(value, (np.bytes_, bytes)) else value


def _load_dataset_from_file(
    dataset: Path, file: Path
) -> Optional[Union[str, float, int, np.ndarray]]:
    """Loads the requested dataset from a HDF5 file."""
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


def _extract_string(
    value: Optional[Union[str, float, int, np.ndarray]],
    dataset: Path,
) -> bool:
    if isinstance(value, str):
        with dataset.with_suffix(".txt").open("w") as output_file:
            output_file.write(value)
        return True
    return False


def _extract_images(
    value: Optional[Union[str, float, int, np.ndarray]],
    dataset: Path,
) -> bool:
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
    value: Optional[Union[str, float, int, np.ndarray]],
    dataset: Path,
) -> bool:
    np.save(dataset.with_suffix(""), np.array(value), allow_pickle=False)
    return True


def extract_hdf5_to_directory(hdf5_file: Path, destination: Path) -> Path:
    """Unpacks the given HDF5 file into the given destination directory.

    Unpacks the given HDF5 file into the given destination directory.  A
    new directory is created in the destination with the same name as
    the HDF5 file.  String values are extracted as ``.txt`` files, and
    images are extracted as ``.png`` files.  Everything else is saved
    with numpy as ``.npy`` files.

    Parameters:
        hdf5_path_name_ext:  The HDF5 file to unpack.
        destination_dir:  The directory in which to create a directory
            for the HDF5 file.

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
