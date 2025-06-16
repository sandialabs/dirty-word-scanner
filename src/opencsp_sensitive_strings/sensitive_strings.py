from __future__ import annotations

import argparse
import copy
import csv
import logging
import os
import shlex
import shutil
import subprocess
import sys
import tempfile
import time
from contextlib import suppress
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional, Union

import cv2
import h5py
import numpy as np
from PIL import Image, UnidentifiedImageError

import opencsp_sensitive_strings.file_cache as fc
import opencsp_sensitive_strings.file_fingerprint as ff
import opencsp_sensitive_strings.sensitive_string_matcher as ssm

logger = logging.getLogger(__name__)
EIGHT_BIT_DEPTH = 255
FULL_DEFINITION_SIZE = (1920, 1080)
MAX_ASPECT_RATIO = 10
MAX_FILE_SIZE = 1e6
MIN_PIXELS = 10
PIL_IMAGE_FORMATS_RW = [
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
    "pcx",
    "png",
    "apng",
    "pbm",
    "pgm",
    "ppm",
    "pnm",
    "sgi",
    "spider",
    "tga",
    "tiff",
    "webp",
    "xbm",
]
"""Image formats that can be handled by the Python Imaging Library (PIL)."""


def numpy_to_image(
    arr: np.ndarray, rescale_or_clip: str = "rescale", rescale_max: int = -1
) -> Image.Image:
    """Convert the numpy representation of an image to a Pillow Image.

    Coverts the given arr to an Image. The array is converted to an integer
    type, as necessary. The color information is then rescaled/clipd to fit
    within an 8-bit color depth.

    In theory, images can be saved with higher bit-depth information using
    opencv imwrite('12bitimage.png', arr), but I (BGB) haven't tried very hard
    and haven't had any luck getting this to work.

    Parameters
    ----------
    arr : np.ndarray
        The array to be converted.
    rescale_or_clip : str, optional
        Whether to rescale the value in the array to fit within 0-255, or to
        clip the values so that anything over 255 is set to 255. By default
        'rescale'.
    rescale_max : int, optional
        The maximum value expected in the input arr, which will be set to 255.
        When less than 0, the maximum of the input array is used. Only
        applicable when rescale_or_clip='rescale'. By default -1.

    Returns
    -------
    image: PIL.Image
        The image representation of the input array.
    """
    allowed_int_types = [
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
    if not np.issubdtype(arr.dtype, np.integer):
        maxval = np.max(arr)
        for int_type in allowed_int_types:
            if np.iinfo(int_type).max >= maxval:
                break
        arr = arr.astype(int_type)
    else:
        int_type = arr.dtype

    # rescale down to 8-bit if bitdepth is too large
    if np.iinfo(int_type).max > EIGHT_BIT_DEPTH:
        if rescale_or_clip == "rescale":
            if rescale_max < 0:
                rescale_max = np.max(arr)
            scale = 255 / rescale_max
            arr = arr * scale
        arr = np.clip(arr, 0, 255)
        arr = arr.astype(np.uint8)
    return Image.fromarray(arr)


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


def _create_dataset_path(
    base_dir: Path, h5_dataset_path_name: Path, dataset_ext: str = ".txt"
) -> Path:
    dataset_location = h5_dataset_path_name.parent
    dataset_name = h5_dataset_path_name.stem
    dataset_path = base_dir / dataset_location
    dataset_path.mkdir(exist_ok=True)
    return dataset_path / (dataset_name + dataset_ext)


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
    for i, (possible_string_name, _) in enumerate(possible_strings):
        dataset_name = possible_string_name.name
        h5_val = load_hdf5_datasets([possible_string_name], norm_path)[
            dataset_name
        ]
        if (
            isinstance(h5_val, np.ndarray)
            and h5_val.ndim <= 1
            and isinstance(h5_val.tolist()[0], str)
        ):
            h5_val = h5_val.tolist()[0]
        if isinstance(h5_val, str):
            dataset_path_name_ext = _create_dataset_path(
                hdf5_dir, possible_strings[i][0], ".txt"
            )
            with dataset_path_name_ext.open("w") as fout:
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
    for i, (possible_image_name, shape) in enumerate(possible_images):
        dataset_name = possible_image_name.name
        h5_val = load_hdf5_datasets([possible_image_name], norm_path)[
            dataset_name
        ]
        if isinstance(h5_val, (h5py.Dataset, np.ndarray)):
            np_image = np.array(h5_val).squeeze()
            # we assume images have 2 or 3 dimensions
            if len(shape) in {2, 3}:
                # We assume shapes are at least 10x10 pixels and have an
                # aspect ratio of at least 10:1.
                aspect_ratio = max(shape[0], shape[1]) / min(
                    shape[0], shape[1]
                )
                if (shape[0] >= MIN_PIXELS and shape[1] >= MIN_PIXELS) and (
                    aspect_ratio < MAX_ASPECT_RATIO + 1e-3
                ):
                    dataset_path_name_ext = _create_dataset_path(
                        hdf5_dir, possible_images[i][0], ".png"
                    )
                    # assumed grayscale or RGB
                    if (
                        len(shape) == 2  # noqa: PLR2004
                        or shape[2] in [1, 3]
                    ):
                        img = numpy_to_image(np_image)
                        img.save(dataset_path_name_ext)
                    else:  # assumed multiple images
                        for index in range(shape[2]):
                            dataset_path_name_ext_i = (
                                dataset_path_name_ext.parent
                                / (
                                    f"{dataset_path_name_ext.stem}_{index}"
                                    + dataset_path_name_ext.suffix
                                )
                            )
                            np_single_image = np_image[:, :, index].squeeze()
                            img = numpy_to_image(np_single_image)
                            img.save(dataset_path_name_ext_i)
                else:
                    other_datasets.append(possible_images[i])
            else:
                other_datasets.append(possible_images[i])
        else:
            other_datasets.append(possible_images[i])
    return other_datasets


def extract_other_datasets(
    other_datasets: list[tuple[Path, tuple[int, ...]]],
    norm_path: Path,
    hdf5_dir: Path,
) -> None:
    for i, (other_dataset_name, _) in enumerate(other_datasets):
        dataset_name = other_dataset_name.name
        h5_val = load_hdf5_datasets([other_dataset_name], norm_path)[
            dataset_name
        ]
        np_val = np.array(h5_val)
        dataset_path_name = _create_dataset_path(
            hdf5_dir, other_datasets[i][0], ""
        )
        # save as a numpy file
        np.save(dataset_path_name, np_val, allow_pickle=False)


class SensitiveStringsSearcher:
    _text_file_extensions = (".txt", ".csv", ".py", ".md", ".rst")
    _text_file_path_name_exts = (".coverageac",)

    def __init__(
        self,
        root_search_dir: Path,
        sensitive_strings_csv: Path,
        allowed_binary_files_csv: Path,
        cache_file_csv: Optional[Path] = None,
    ) -> None:
        self.root_search_dir = root_search_dir
        self.sensitive_strings_csv = sensitive_strings_csv
        self.allowed_binary_files_csv = allowed_binary_files_csv
        self.cache_file_csv = cache_file_csv
        self.verbose = False
        self._interactive = False
        self.verify_all_on_behalf_of_user = False
        self.remove_unfound_binaries = False
        self.date_time_str = datetime.now(tz=timezone.utc).strftime(
            "%Y%m%d_%H%M%S"
        )
        self.tmp_dir_base = (
            Path(tempfile.gettempdir()) / "SensitiveStringSearcher"
        )
        self.git_files_only = True
        self.is_hdf5_searcher = False
        self.has_backed_up_allowed_binaries_csv = False

        self.matchers = self.build_matchers()
        self.matches: dict[Path, list[ssm.Match]] = {}
        self.allowed_binary_files: list[ff.FileFingerprint] = []
        self.accepted_binary_files: list[ff.FileFingerprint] = []
        self.unknown_binary_files: list[ff.FileFingerprint] = []
        self.unfound_allowed_binary_files: list[ff.FileFingerprint] = []
        self.cached_cleared_files: list[fc.FileCache] = []
        self.new_cached_cleared_files: list[fc.FileCache] = []

    @property
    def interactive(self) -> bool:
        return self._interactive or self.verify_all_on_behalf_of_user

    @interactive.setter
    def interactive(self, val: bool) -> None:
        self._interactive = val

    def __del__(self) -> None:
        for root, dirs, _ in os.walk(self.tmp_dir_base):
            for dir_name in dirs:
                if dir_name.startswith("tmp_"):
                    shutil.rmtree(Path(root) / dir_name)

    def build_matchers(self) -> list[ssm.SensitiveStringMatcher]:
        matchers: list[ssm.SensitiveStringMatcher] = []
        with self.sensitive_strings_csv.open(newline="") as csv_file:
            reader = csv.reader(csv_file)
            next(reader)
            for row in reader:
                if row:
                    name = row[0]
                    patterns = row[1:]
                    matchers.append(
                        ssm.SensitiveStringMatcher(name, *patterns)
                    )
        return matchers

    def norm_path(self, file_path: Path, file_name_ext: Path) -> Path:
        return self.root_search_dir / file_path / file_name_ext

    def _is_file_in_cleared_cache(
        self, file_path: Path, file_name_ext: Path
    ) -> bool:
        cache_entry = fc.FileCache.for_file(
            self.root_search_dir, file_path, file_name_ext
        )
        return cache_entry in self.cached_cleared_files

    def _register_file_in_cleared_cache(
        self, file_path: Path, file_name_ext: Path
    ) -> None:
        cache_entry = fc.FileCache.for_file(
            self.root_search_dir, file_path, file_name_ext
        )
        self.new_cached_cleared_files.append(cache_entry)

    def _is_binary_file(
        self, rel_file_path: Path, file_name_ext: Path
    ) -> bool:
        is_binary_file = False

        # check if a binary file
        ext = file_name_ext.suffix.lower()
        if ext == ".ipynb":
            is_binary_file = True
        elif self._is_img_ext(ext):
            is_binary_file = (
                (ext not in self._text_file_extensions)
                and (
                    f"{rel_file_path / file_name_ext}"
                    not in self._text_file_path_name_exts
                )
                and (file_name_ext not in self._text_file_path_name_exts)
            )
        elif (
            self.norm_path(rel_file_path, file_name_ext).stat().st_size
            > MAX_FILE_SIZE
        ):
            # assume any file > 1MB is a binary file, in order to prevent
            # sensitive_strings from taking hours to check these files
            # needlessly
            is_binary_file = True
        if not is_binary_file:
            # attempt to parse the file as a text file
            try:
                file_path_norm: Path = self.norm_path(
                    rel_file_path, file_name_ext
                )
                with file_path_norm.open(newline="") as input_stream:
                    input_stream.readlines()
            except UnicodeDecodeError:
                is_binary_file = True

        return is_binary_file

    def _enqueue_unknown_binary_files_for_later_processing(
        self, rel_file_path: Path, file_name_ext: Path
    ) -> None:
        """
        Determine what to do with binary files.

        If the given file is recognized as an allowed file, and its
        fingerprint matches the allowed file, then we can dismiss it
        from the list of unfound files and add it to the list of the
        accepted files.

        However, if the given file isn't recognized or its fingerprint
        is different, then add it to the unknown list, to be dealt with
        later.
        """
        file_ff = ff.FileFingerprint.for_file(
            self.root_search_dir, rel_file_path, file_name_ext
        )

        if file_ff in self.allowed_binary_files:
            # we already know and trust this binary file
            with suppress(ValueError):
                self.unfound_allowed_binary_files.remove(file_ff)
            self.accepted_binary_files.append(file_ff)
        else:
            # we'll deal with unknown files as a group later
            self.unknown_binary_files.append(file_ff)

    def parse_file(
        self, rel_file_path: Path, file_name_ext: Path
    ) -> list[str]:
        file_path_norm: Path = self.norm_path(rel_file_path, file_name_ext)
        logger.debug(file_path_norm)

        if self._is_binary_file(rel_file_path, file_name_ext):
            return []
        with file_path_norm.open(newline="") as input_stream:
            return input_stream.readlines()

    def search_lines(self, lines: list[str]) -> list[ssm.Match]:
        matches: list[ssm.Match] = []

        for matcher in self.matchers:
            matches += matcher.check_lines(lines)

        return matches

    def search_file(
        self, file_path: Path, file_name_ext: Path
    ) -> list[ssm.Match]:
        lines = self.parse_file(file_path, file_name_ext)

        matches: list[ssm.Match] = []
        matches += self.search_lines([f"{file_path}/{file_name_ext}"])
        matches += self.search_lines(lines)

        return matches

    def get_tmp_dir(self) -> Path:
        i = 0
        while True:
            ret = self.tmp_dir_base / f"tmp_{i}"
            if ret.is_dir():
                i += 1
            else:
                return ret

    def _handle_hdf5_error(
        self,
        matches: list[ssm.Match],
        hdf5_matches: dict[Path, list[ssm.Match]],
        relative_path_name_ext: str,
    ) -> None:
        # There was an error, but the user may want to sign off on
        # the file anyway.
        if len(hdf5_matches) > 0:
            # Describe the issues with the HDF5 file
            logger.warning(
                "Found possible issues with the HDF5 file:",
                extra={
                    "number_of_issues": len(hdf5_matches),
                    "file": relative_path_name_ext,
                },
            )
            prev_relpath_name_ext = None
            for file_relpath_name_ext, file_matches in hdf5_matches.items():
                if prev_relpath_name_ext != file_relpath_name_ext:
                    logger.warning("    %s:", file_relpath_name_ext)
                    prev_relpath_name_ext = file_relpath_name_ext
                for match in file_matches:
                    logger.warning(
                        "        %s (line %d, col %d)",
                        match.msg,
                        match.lineno,
                        match.colno,
                    )

            # Ask the user about signing off
            if self.interactive:
                if not self.verify_interactively(file_relpath_name_ext):
                    matches.append(
                        ssm.Match(
                            0,
                            0,
                            0,
                            "",
                            "",
                            None,
                            "HDF5 file denied by user",
                        )
                    )
            else:  # if self.interactive
                for (
                    file_relpath_name_ext,
                    file_matches,
                ) in hdf5_matches.items():
                    for match in file_matches:
                        dataset_name = (
                            f"{file_relpath_name_ext.parent}/"
                            + file_relpath_name_ext.stem
                        )
                        match.msg = f"{dataset_name}::{match.msg}"
                        matches.append(match)
        else:  # if len(hdf5_matches) > 0:
            message = (
                "Programmer error in SensitiveStringsSearcher."
                "search_hdf5_files(): Errors were returned for file "
                f"{relative_path_name_ext} but there were 0 matches "
                "found.",
            )
            logger.error(message)
            raise RuntimeError(message)

    def search_hdf5_file(
        self, hdf5_file: ff.FileFingerprint
    ) -> list[ssm.Match]:
        norm_path = self.norm_path(hdf5_file.relative_path, hdf5_file.name_ext)
        relative_path_name_ext = (
            f"{hdf5_file.relative_path}/{hdf5_file.name_ext}"
        )
        matches: list[ssm.Match] = []

        # Extract the contents from the HDF5 file
        unzip_dir = self.get_tmp_dir()
        logger.info("")
        logger.info("**Extracting HDF5 file**", extra={"directory": unzip_dir})
        h5_dir = extract_hdf5_to_directory(norm_path, unzip_dir)

        # Create a temporary allowed binary strings file
        with self.allowed_binary_files_csv.open() as fin:
            allowed_binary_files_lines = fin.readlines()
        fd, tmp_allowed_binary_csv = tempfile.mkstemp(
            dir=self.tmp_dir_base,
            suffix=".csv",
            text=True,
        )
        tmp_allowed_binary_csv_path = Path(tmp_allowed_binary_csv)
        with os.fdopen(fd, "w") as fout:
            fout.writelines(allowed_binary_files_lines)

        # Create a searcher for the unzipped directory
        hdf5_searcher = SensitiveStringsSearcher(
            h5_dir, self.sensitive_strings_csv, tmp_allowed_binary_csv_path
        )
        hdf5_searcher.interactive = self.interactive
        hdf5_searcher.verify_all_on_behalf_of_user = (
            self.verify_all_on_behalf_of_user
        )
        hdf5_searcher.date_time_str = self.date_time_str
        hdf5_searcher.tmp_dir_base = self.tmp_dir_base
        hdf5_searcher.git_files_only = False
        hdf5_searcher.is_hdf5_searcher = True

        # Validate all of the unzipped files
        error = hdf5_searcher.search_files()
        hdf5_matches = hdf5_searcher.matches
        if error != 0:
            self._handle_hdf5_error(
                matches, hdf5_matches, relative_path_name_ext
            )
        elif len(hdf5_matches) > 0:
            message = (
                "Programmer error in SensitiveStringsSearcher."
                "search_hdf5_files(): No errors were returned for file "
                f"{relative_path_name_ext} but there were "
                f"{len(hdf5_matches)} > 0 matches found.",
            )
            logger.error(message)
            raise RuntimeError(message)

        # Remove the temporary files created for the searcher.
        # Files created by the searcher should be removed in its
        # __del__() method.
        tmp_allowed_binary_csv_path.unlink()

        return matches

    def verify_interactively(
        self,
        relative_path_name_ext: Path,
        cv_img: Optional[Image.Image] = None,
        cv_title: Optional[str] = None,
    ) -> bool:
        if cv_img is None:
            logger.info("")
            logger.info("Unknown binary file:")
            logger.info("    %s", relative_path_name_ext)
            logger.info(
                "Is this unknown binary file safe to add, and doesn't contain "
                "any sensitive information (y/n)?"
            )
            if self.verify_all_on_behalf_of_user:
                val = "y"
            else:
                resp = input("").strip()
                val = "n" if len(resp) == 0 else resp[0]
            logger.info("    User responded '%s'", val)

        else:
            logger.info("")
            logger.info(
                "Is this image safe to add, and doesn't contain any sensitive "
                "information (y/n)?"
            )
            if self.verify_all_on_behalf_of_user:
                val = "y"
            else:
                cv2.imshow(cv_title or "", cv_img)
                key = cv2.waitKey(0)
                cv2.destroyAllWindows()
                time.sleep(
                    0.1
                )  # small delay to prevent accidental double-bounces

                # Check for 'y' or 'n'
                if key in [ord("y"), ord("Y")]:
                    val = "y"
                elif key in [ord("n"), ord("N")]:
                    val = "n"
                else:
                    val = "?"
            if val.lower() in ["y", "n"]:
                logger.info("    User responded '%s'", val)
            else:
                logger.error(
                    "Did not respond with either 'y' or 'n'. Assuming 'n'."
                )
                val = "n"

        return val.lower() == "y"

    def search_binary_file(
        self, binary_file: ff.FileFingerprint
    ) -> list[ssm.Match]:
        norm_path = self.norm_path(
            binary_file.relative_path, binary_file.name_ext
        )
        ext = norm_path.suffix
        relative_path_name_ext = (
            binary_file.relative_path / binary_file.name_ext
        )
        matches: list[ssm.Match] = []

        if ext.lower().lstrip(".") in PIL_IMAGE_FORMATS_RW:
            if self.interactive:
                if self.interactive_image_sign_off(
                    norm_path, file_ff=binary_file
                ):
                    return []
                matches.append(
                    ssm.Match(0, 0, 0, "", "", None, "File denied by user")
                )
            else:
                matches.append(
                    ssm.Match(0, 0, 0, "", "", None, "Unknown image file")
                )

        elif ext.lower() == ".h5":
            matches += self.search_hdf5_file(binary_file)

        elif not self.verify_interactively(relative_path_name_ext):
            matches.append(
                ssm.Match(0, 0, 0, "", "", None, "Unknown binary file")
            )

        return matches

    def _is_img_ext(self, ext: str) -> bool:
        return ext.lower().lstrip(".") in PIL_IMAGE_FORMATS_RW

    def interactive_image_sign_off(
        self,
        description: Path,
        np_image: Optional[np.ndarray] = None,
        file_ff: Optional[ff.FileFingerprint] = None,
    ) -> bool:
        if (np_image is None) and (file_ff is not None):
            file_norm_path = self.norm_path(
                file_ff.relative_path, file_ff.name_ext
            )
            if self._is_img_ext(file_norm_path.suffix):
                try:
                    img = Image.open(file_norm_path).convert("RGB")
                except (
                    FileNotFoundError,
                    UnidentifiedImageError,
                    ValueError,
                    TypeError,
                ):
                    img = None
                if img is not None:
                    np_image = np.copy(np.array(img))
                    img.close()
                    return self.interactive_image_sign_off(
                        file_ff.relative_path / file_ff.name_ext,
                        np_image=np_image,
                    )
                return self.verify_interactively(file_ff.relative_path)
                # if img is not None
            return False

        if np_image is None:
            return False

        # rescale the image for easier viewing
        img = numpy_to_image(np_image)
        rescaled = ""
        if img.size[0] > FULL_DEFINITION_SIZE[0]:
            scale = FULL_DEFINITION_SIZE[0] / img.size[0]
            img = img.resize(
                (int(scale * img.size[0]), int(scale * img.size[1]))
            )
            np_image = np.array(img)
            rescaled = " (downscaled)"
        if img.size[1] > FULL_DEFINITION_SIZE[1]:
            scale = FULL_DEFINITION_SIZE[1] / img.size[1]
            img = img.resize(
                (int(scale * img.size[0]), int(scale * img.size[1]))
            )
            np_image = np.array(img)
            rescaled = " (downscaled)"

        # Show the image and prompt the user
        return self.verify_interactively(
            description, np_image, f"{description}{rescaled}"
        )

    def _init_files_lists(self) -> None:
        self.matches.clear()

        if self.is_hdf5_searcher:
            # HDF5 searchers shouldn't be aware of what files are
            # contained in the HDF5 file.
            self.allowed_binary_files.clear()
        else:
            self.allowed_binary_files = [
                inst
                for inst, _ in ff.FileFingerprint.from_csv(
                    self.allowed_binary_files_csv.parent,
                    Path(self.allowed_binary_files_csv.name),
                )
            ]
        self.accepted_binary_files.clear()
        self.unknown_binary_files.clear()
        self.unfound_allowed_binary_files = copy.copy(
            self.allowed_binary_files
        )

        self.cached_cleared_files.clear()
        self.new_cached_cleared_files.clear()
        sensitive_strings_cache = fc.FileCache.for_file(
            self.sensitive_strings_csv.parent,
            Path(),
            Path(self.sensitive_strings_csv.name),
        )
        if self.cache_file_csv is not None and self.cache_file_csv.is_file():
            self.cached_cleared_files = [
                inst for inst, _ in fc.FileCache.from_csv(self.cache_file_csv)
            ]
            if sensitive_strings_cache not in self.cached_cleared_files:
                self.cached_cleared_files.clear()
        self.new_cached_cleared_files.append(sensitive_strings_cache)

    def create_backup_allowed_binaries_csv(self) -> None:
        backup_name_ext = (
            f"{self.allowed_binary_files_csv.stem}_backup_{self.date_time_str}"
            f"{self.allowed_binary_files_csv.suffix}"
        )
        backup_path_name_ext = (
            self.allowed_binary_files_csv.parent / backup_name_ext
        )
        if backup_path_name_ext.is_file():
            backup_path_name_ext.unlink()
        shutil.copyfile(
            self.allowed_binary_files_csv,
            self.allowed_binary_files_csv.parent / backup_name_ext,
        )
        self.has_backed_up_allowed_binaries_csv = True

    def update_allowed_binaries_csv(self) -> None:
        # Overwrite the allowed list CSV file with the updated
        # allowed_binary_files.
        if not self.has_backed_up_allowed_binaries_csv:
            self.create_backup_allowed_binaries_csv()
        self.allowed_binary_files = sorted(self.allowed_binary_files)
        self.allowed_binary_files[0].to_csv(
            self.allowed_binary_files_csv.parent,
            self.allowed_binary_files_csv.stem,
            self.allowed_binary_files,
        )

    def get_tracked_files(self) -> list[Path]:
        # If this script is evaluated form MobaXTerm, then the
        # built-in 16-bit version of git will fail.
        if (git := shutil.which("git")) is None:
            message = "'git' executable not found."
            raise RuntimeError(message)
        if "mobaxterm" in git:
            git = "git"
        files: list[Path] = []
        for command in [
            f"{git} ls-tree --full-tree --name-only -r HEAD",
            f"{git} diff --name-only --cached --diff-filter=A",
        ]:
            completed_process = subprocess.run(  # noqa: S603
                shlex.split(command),
                check=True,
                cwd=self.root_search_dir,
                stdout=subprocess.PIPE,
                text=True,
            )
            files.extend(
                Path(file)
                for file in completed_process.stdout.splitlines()
                if (self.root_search_dir / file).is_file()
            )
        logger.info(
            "Searching for sensitive strings in tracked files",
            extra={"number_of_files": len(files)},
        )
        return files

    def get_files_in_directory(self) -> list[Path]:
        files: list[Path] = []
        for directory, _, files_in_directory in os.walk(self.root_search_dir):
            for file_name in files_in_directory:
                full_path = Path(directory) / file_name
                relative_path = os.path.relpath(
                    full_path, self.root_search_dir
                )
                files.append(Path(relative_path))
        logger.info(
            "Searching for sensitive strings in files",
            extra={"number_of_files": len(files)},
        )
        return files

    def handle_files(self, files: list[Path]) -> dict[Path, list[ssm.Match]]:
        # Search for sensitive strings in files
        matches: dict[Path, list[ssm.Match]] = {}
        for file_path_name_ext in files:
            if self.verbose:
                logger.info("Searching file %s", file_path_name_ext)
            rel_file_path = file_path_name_ext.parent
            file_name_ext = Path(file_path_name_ext.name)
            if self._is_file_in_cleared_cache(rel_file_path, file_name_ext):
                # file cleared in a previous run, don't need to check again
                self._register_file_in_cleared_cache(
                    rel_file_path, file_name_ext
                )
            # need to check this file
            elif self._is_binary_file(rel_file_path, file_name_ext):
                # deal with non-parseable binary files as a group, below
                self._enqueue_unknown_binary_files_for_later_processing(
                    rel_file_path, file_name_ext
                )
            else:
                # check text files for sensitive strings
                file_matches = self.search_file(rel_file_path, file_name_ext)
                if len(file_matches) > 0:
                    matches[file_path_name_ext] = file_matches
                else:
                    self._register_file_in_cleared_cache(
                        rel_file_path, file_name_ext
                    )

        # Potentially remove unfound binary files
        if (
            len(self.unfound_allowed_binary_files) > 0
            and self.remove_unfound_binaries
        ):
            for file in self.unfound_allowed_binary_files:
                self.allowed_binary_files.remove(file)
            self.unfound_allowed_binary_files.clear()
            self.update_allowed_binaries_csv()
        return matches

    def print_match_information(
        self, matches: dict[Path, list[ssm.Match]]
    ) -> None:
        # Print initial information about matching files and problematic
        # binary files.
        if matches:
            logger.error(
                "Found files containing sensitive strings:",
                extra={"number_of_files": len(matches)},
            )
            for file, file_matches in matches.items():
                logger.error("    File %s:", file)
                for match in file_matches:
                    logger.error("        %s", match.msg)
        if len(self.unfound_allowed_binary_files) > 0:
            logger.error(
                "Expected binary files that can't be found:",
                extra={
                    "number_of_files": len(self.unfound_allowed_binary_files)
                },
            )
            for file_ff in self.unfound_allowed_binary_files:
                logger.info("")
                logger.error(file_ff.relpath_name_ext)
        if len(self.unknown_binary_files) > 0:
            logger.warning(
                "Found unexpected binary files:",
                extra={"number_of_files": len(self.unknown_binary_files)},
            )

    def handle_unknown_binary_files(self) -> None:
        # Deal with unknown binary files
        if len(self.unknown_binary_files) == 0:
            return
        unknowns_copy = copy.copy(self.unknown_binary_files)
        for file_ff in unknowns_copy:
            if self.verbose:
                logger.info(
                    "Searching binary file %s", file_ff.relpath_name_ext
                )
            logger.info("")
            logger.info(file_ff.relpath_name_ext)
            num_signed_binary_files = 0

            if parsable_matches := self.search_binary_file(file_ff):
                # This file is not ok. Tell the user why.
                logger.error(
                    "    Found possible sensitive issues in file.",
                    extra={
                        "number_of_issues": len(parsable_matches),
                        "file": self.norm_path(
                            file_ff.relative_path, file_ff.name_ext
                        ),
                    },
                )
                for _match in parsable_matches:
                    match: ssm.Match = _match
                    logger.error(
                        "    %s (line %d, col %d)",
                        match.msg,
                        match.lineno,
                        match.colno,
                    )

            else:
                # No matches: this file is ok.
                # Add the validated and/or signed off file to the
                # allowed binary files CSV.
                self.unknown_binary_files.remove(file_ff)
                self.allowed_binary_files.append(file_ff)

                # Overwrite the allowed list CSV file with the updated
                # allowed_binary_files and make a backup as necessary.
                self.update_allowed_binaries_csv()

                num_signed_binary_files += 1

            # Date+time stamp the new allowed list csv files
            if num_signed_binary_files > 0:
                abfc_stamped_name_ext = (
                    f"{self.allowed_binary_files_csv.stem}_"
                    + self.date_time_str
                    + self.allowed_binary_files_csv.suffix
                )
                abfc_stamped_path_name_ext = (
                    self.allowed_binary_files_csv.parent
                    / abfc_stamped_name_ext
                )
                if abfc_stamped_path_name_ext.is_file():
                    abfc_stamped_path_name_ext.unlink()
                shutil.copyfile(
                    self.allowed_binary_files_csv,
                    self.allowed_binary_files_csv.parent
                    / abfc_stamped_name_ext,
                )
        # for file_ff in unknowns_copy

    def check_cache(self) -> None:
        # Make sure we didn't accidentally add any binary files to the cache
        for file_ff in (
            self.allowed_binary_files + self.unfound_allowed_binary_files
        ):
            for file_cf in self.new_cached_cleared_files:
                if file_ff == file_cf:
                    message = (
                        "Programmer error in sensitive_strings."
                        "search_files(): No binary files should be in the "
                        "cache, but at least 1 such file was found: "
                        f'"{file_cf.relative_path}/{file_cf.name_ext}"',
                    )
                    logger.error(message)
                    raise RuntimeError(message)

        # Save the cleared files cache
        for file_ff in self.unknown_binary_files:
            for file_cf in self.new_cached_cleared_files:
                if file_ff == file_cf:
                    self.new_cached_cleared_files.remove(file_cf)
                    break
        if (
            self.cache_file_csv is not None
            and len(self.new_cached_cleared_files) > 0
        ):
            self.cache_file_csv.parent.mkdir(parents=True, exist_ok=True)
            self.new_cached_cleared_files[0].to_csv(
                self.cache_file_csv.parent,
                self.cache_file_csv.stem,
                self.new_cached_cleared_files,
            )

    def number_of_findings(self, matches: dict[Path, list[ssm.Match]]) -> int:
        ret = (
            len(matches)
            + len(self.unfound_allowed_binary_files)
            + len(self.unknown_binary_files)
        )
        info_or_warn = logger.warning if ret > 0 else logger.info
        info_or_warn("Summary:")
        info_or_warn("<<<PASS>>>" if ret == 0 else "<<<FAIL>>>")
        info_or_warn(f"Found {len(matches)} sensitive string matches")
        if len(self.unfound_allowed_binary_files) > 0:
            info_or_warn(
                f"Did not find {len(self.unfound_allowed_binary_files)} "
                "expected binary files"
            )
        else:
            info_or_warn(
                f"Found {len(self.allowed_binary_files)} expected binary files"
            )
        info_or_warn(
            f"Found {len(self.unknown_binary_files)} unexpected binary files"
        )

        # Add a 'match' for any unfound or unknown binary files
        if not self.is_hdf5_searcher:
            for file_ff in self.unfound_allowed_binary_files:
                fpne = file_ff.relative_path / file_ff.name_ext
                matches[fpne] = matches.get(fpne, [])
                matches[fpne].append(
                    ssm.Match(
                        0, 0, 0, "", "", None, f"Unfound binary file {fpne}"
                    )
                )
        for file_ff in self.unknown_binary_files:
            fpne = file_ff.relative_path / file_ff.name_ext
            matches[fpne] = matches.get(fpne, [])
            matches[fpne].append(
                ssm.Match(0, 0, 0, "", "", None, f"Unknown binary file {fpne}")
            )
        self.matches = matches
        return ret

    def search_files(self) -> int:
        self._init_files_lists()
        files = sorted(
            set(
                self.get_tracked_files()
                if self.git_files_only
                else self.get_files_in_directory()
            )
        )
        matches = self.handle_files(files)
        self.print_match_information(matches)
        self.handle_unknown_binary_files()
        self.check_cache()
        return self.number_of_findings(matches)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        prog=__file__.rstrip(".py"), description="Sensitive strings searcher"
    )
    parser.add_argument(
        "--no-interactive",
        action="store_true",
        dest="ninteractive",
        help="Don't interactively ask the user about unknown binary files. "
        "Simply fail instead.",
    )
    parser.add_argument(
        "--accept-all",
        action="store_true",
        dest="acceptall",
        help="Don't interactively ask the user about unknown binary files. "
        "Simply accept all as verified on the user's behalf. This can be "
        "useful when you're confident that the only changes have been that "
        "the binary files have moved but not changed.",
    )
    parser.add_argument(
        "--accept-unfound",
        action="store_true",
        dest="acceptunfound",
        help="Don't fail because of unfound expected binary files. Instead "
        "remove the expected files from the list of allowed binaries. This "
        "can be useful when you're confident that the only changes have been "
        "that the binary files have moved but not changed.",
    )
    parser.add_argument(
        "--log-dir",
        default=Path(tempfile.gettempdir()) / "sensitive_strings",
        help="The directory in which to store all logs.",
        type=Path,
    )
    parser.add_argument(
        "--sensitive-strings",
        help="The CSV file defining the sensitive string patterns to search "
        "for.",
        required=True,
        type=Path,
    )
    parser.add_argument(
        "--allowed-binaries",
        help="The CSV file defining the allowed binary files.",
        required=True,
        type=Path,
    )
    parser.add_argument(
        "--cache-file",
        default=None,
        help="The directory in which to store all logs.",
        type=Path,
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        dest="verbose",
        help="Print more information while running",
    )
    args = parser.parse_args()
    not_interactive: bool = args.ninteractive
    accept_all: bool = args.acceptall
    remove_unfound_binaries: bool = args.acceptunfound
    verbose: bool = args.verbose

    log_path: Path = args.log_dir / "sensitive_strings_log.txt"
    sensitive_strings_csv = args.sensitive_strings
    allowed_binary_files_csv = args.allowed_binaries
    ss_cache_file: Path = (
        args.cache_file if args.cache_file else log_path / "cache.csv"
    )
    date_time_str = datetime.now(tz=timezone.utc).strftime("%Y%m%d_%H%M%S")

    log_path = (
        log_path.parent / f"{log_path.stem}_{date_time_str}{log_path.suffix}"
    )
    logging.basicConfig(filename=log_path, level=logging.INFO)

    root_search_dir = Path(__file__).parents[2]
    searcher = SensitiveStringsSearcher(
        root_search_dir,
        sensitive_strings_csv,
        allowed_binary_files_csv,
        ss_cache_file,
    )
    searcher.interactive = not not_interactive
    searcher.verify_all_on_behalf_of_user = accept_all
    searcher.remove_unfound_binaries = remove_unfound_binaries
    searcher.verbose = verbose
    searcher.date_time_str = date_time_str
    num_errors = searcher.search_files()

    if num_errors > 0:
        sys.exit(1)
    else:
        sys.exit(0)
