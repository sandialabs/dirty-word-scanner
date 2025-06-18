from pathlib import Path
from typing import Union

import h5py
import numpy as np

from opencsp_sensitive_strings.image_conversion import numpy_to_image

MAX_ASPECT_RATIO = 10
MIN_PIXELS = 10


def is_dataset_and_shape(
    entity: Union[h5py.Group, h5py.Dataset],
) -> tuple[bool, tuple[int, ...]]:
    """Returns whether the given object is an hdf5 dataset and, if it is, then
    also what it's shape is.

    Parameters
    ----------
    object : Union[h5py.Group, h5py.Dataset]
        The object to check the type of.

    Returns
    -------
    is_dataset: bool
        True if object is a dataset, False otherwise
    shape: tuple[int, ...]
        The shape of the dataset. Empty tuple() object if not a dataset.
    """
    if not isinstance(entity, h5py.Group):
        if isinstance(entity, h5py.Dataset):
            dset: h5py.Dataset = entity
            return True, dset.shape
        return True, ()
    return False, ()


def get_groups_and_datasets(
    hdf5_path_name_ext: Path,
) -> tuple[list[str], list[tuple[Path, tuple[int, ...]]]]:
    """
    Get the structure of an HDF5 file.

    Including all group and dataset names, and the dataset shapes.

    Parameters:
        hdf5_path_name_ext:  The HDF5 file to parse the structure of.

    Returns:
        group_names
            The absolute names of all the groups in the file.  For
            example:  ``"foo/bar"``.
        file_names_and_shapes
            The absolute names of all the datasets in the file, and
            their shapes.  For example:  ``"foo/bar/baz",
            (1920, 1080)``.
    """
    group_names: list[str] = []
    file_names_and_shapes: list[tuple[Path, tuple[int, ...]]] = []
    visited: list[tuple[str, bool, tuple]] = []

    def visitor(name: str, entity: Union[h5py.Group, h5py.Dataset]) -> None:
        visited.append((name, *is_dataset_and_shape(entity)))

    with h5py.File(hdf5_path_name_ext, "r") as fin:
        fin.visititems(visitor)
    for name, is_dataset, shape in visited:
        # Add to the file or group names list.
        # If a dataset, then include its shape.
        if not is_dataset:
            group_names.append(name)
        if is_dataset:
            file_names_and_shapes.append((Path(name), shape))
    return group_names, file_names_and_shapes


def load_hdf5_datasets(
    datasets: list[Path], file: Path
) -> dict[str, Union[str, h5py.Dataset]]:
    """Loads datasets from HDF5 file"""
    with h5py.File(file, "r") as f:
        kwargs: dict[str, Union[str, h5py.Dataset]] = {}
        # Loop through fields to retreive
        for dataset in datasets:
            # Get data and get dataset name
            data = f[f"{dataset}"]
            name = dataset.name

            # Format data shape
            if np.ndim(data) == 0 and np.size(data) == 1:
                data = data[()]
            elif np.ndim(data) > 0 and np.size(data) == 1:
                data = data[...].flatten()[0]
            elif np.size(data) > 0:
                data = data[...].squeeze()

            # Format strings
            if type(data) is np.bytes_ or type(data) is bytes:
                data = data.decode()

            # Save in dictionary
            kwargs[name] = data
    return kwargs


def _create_dataset_path(dataset: Path, extension: str = ".txt") -> Path:
    dataset.parent.mkdir(exist_ok=True, parents=True)
    return dataset.with_suffix(extension)


def extract_hdf5_to_directory(
    hdf5_path_name_ext: Path, destination_dir: Path
) -> Path:
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
    hdf5_dir = destination_dir / hdf5_path_name_ext.name

    # Create the HDF5 output directory
    hdf5_dir.mkdir(exist_ok=True, parents=True)

    # Get all of what may be strings or images from the h5 file
    _, dataset_names_and_shapes = get_groups_and_datasets(hdf5_path_name_ext)
    possible_strings: list[tuple[Path, tuple[int, ...]]] = []
    possible_strings.extend(
        (dataset_name, shape)
        for dataset_name, shape in dataset_names_and_shapes
    )
    # Extract strings into .txt files, and collect possible images
    possible_images = extract_strings(
        possible_strings, hdf5_path_name_ext, hdf5_dir
    )
    # Extract images into .png files, and collect other datasets
    other_datasets = extract_images(
        possible_images, hdf5_path_name_ext, hdf5_dir
    )
    # Extract everything else into numpy arrays
    extract_other_datasets(other_datasets, hdf5_path_name_ext, hdf5_dir)
    return hdf5_dir


def extract_strings(
    possible_strings: list[tuple[Path, tuple[int, ...]]],
    norm_path: Path,
    hdf5_dir: Path,
) -> list[tuple[Path, tuple[int, ...]]]:
    possible_images: list[tuple[Path, tuple[int, ...]]] = []
    for i, (possible_string, _) in enumerate(possible_strings):
        h5_val = load_hdf5_datasets([possible_string], norm_path)[
            possible_string.name
        ]
        if (
            isinstance(h5_val, np.ndarray)
            and h5_val.ndim <= 1
            and isinstance(h5_val.tolist()[0], str)
        ):
            h5_val = h5_val.tolist()[0]
        if isinstance(h5_val, str):
            dataset_path = _create_dataset_path(
                hdf5_dir / possible_strings[i][0], ".txt"
            )
            with dataset_path.open("w") as fout:
                fout.write(h5_val)
        else:
            possible_images.append(possible_strings[i])
    return possible_images


def extract_images(
    possible_images: list[tuple[Path, tuple[int, ...]]],
    norm_path: Path,
    hdf5_dir: Path,
) -> list[tuple[Path, tuple[int, ...]]]:
    other_datasets: list[tuple[Path, tuple[int, ...]]] = []
    for i, (possible_image, shape) in enumerate(possible_images):
        h5_val = load_hdf5_datasets([possible_image], norm_path)[
            possible_image.name
        ]
        if not isinstance(h5_val, (h5py.Dataset, np.ndarray)):
            other_datasets.append(possible_images[i])
            continue
        np_image = np.array(h5_val).squeeze()
        # we assume images have 2 or 3 dimensions
        if len(shape) not in {2, 3}:
            other_datasets.append(possible_images[i])
            continue
        # We assume shapes are at least 10x10 pixels and have an
        # aspect ratio of at least 10:1.
        aspect_ratio = max(shape[0], shape[1]) / min(shape[0], shape[1])
        if (shape[0] < MIN_PIXELS or shape[1] < MIN_PIXELS) or (
            aspect_ratio >= MAX_ASPECT_RATIO + 1e-3
        ):
            other_datasets.append(possible_images[i])
            continue
        dataset_path = _create_dataset_path(
            hdf5_dir / possible_images[i][0], ".png"
        )
        # assumed grayscale or RGB
        if len(shape) == 2 or shape[2] in [1, 3]:  # noqa: PLR2004
            img = numpy_to_image(np_image)
            img.save(dataset_path)
        else:  # assumed multiple images
            for index in range(shape[2]):
                dataset_path_i = (
                    dataset_path.parent
                    / f"{dataset_path.stem}_{index}{dataset_path.suffix}"
                )
                np_single_image = np_image[:, :, index].squeeze()
                img = numpy_to_image(np_single_image)
                img.save(dataset_path_i)
    return other_datasets


def extract_other_datasets(
    other_datasets: list[tuple[Path, tuple[int, ...]]],
    norm_path: Path,
    hdf5_dir: Path,
) -> None:
    for i, (other_dataset_name, _) in enumerate(other_datasets):
        h5_val = load_hdf5_datasets([other_dataset_name], norm_path)[
            other_dataset_name.name
        ]
        np_val = np.array(h5_val)
        dataset_path_name = _create_dataset_path(
            hdf5_dir / other_datasets[i][0], ""
        )
        # save as a numpy file
        np.save(dataset_path_name, np_val, allow_pickle=False)
