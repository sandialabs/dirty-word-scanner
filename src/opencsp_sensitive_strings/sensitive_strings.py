import argparse
import copy
import csv
import h5py
import logging
import os
import shutil
import subprocess
import sys
import tempfile
import time
from contextlib import suppress
from datetime import datetime
from typing import Union

import cv2
import numpy as np
from PIL import Image

import src.opencsp_sensitive_strings.FileCache as fc
import src.opencsp_sensitive_strings.FileFingerprint as ff
import src.opencsp_sensitive_strings.SensitiveStringMatcher as ssm


logger = logging.getLogger(__name__)
pil_image_formats_rw = [
    "blp", "bmp", "dds", "dib", "eps", "gif", "icns", "ico", "im", "jpg",
    "jpeg", "msp", "pcx", "png", "apng", "pbm", "pgm", "ppm", "pnm", "sgi",
    "spider", "tga", "tiff", "webp", "xbm",
]
"""Image formats that can be handled by the Python Imaging Library (PIL)."""


def numpy_to_image(arr: np.ndarray, rescale_or_clip="rescale", rescale_max=-1):
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
    allowed_int_types = [np.int8, np.uint8, np.int16, np.uint16, np.int32, np.uint32, np.int64, np.uint64]

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
    if np.iinfo(int_type).max > 255:
        if rescale_or_clip == "rescale":
            if rescale_max < 0:
                rescale_max = np.max(arr)
            scale = 255 / rescale_max
            arr = arr * scale
            arr = np.clip(arr, 0, 255)
            arr = arr.astype(np.uint8)
        else:
            arr = np.clip(arr, 0, 255)
            arr = arr.astype(np.uint8)

    img = Image.fromarray(arr)
    return img


def is_dataset_and_shape(object: Union[h5py.Group, h5py.Dataset]) -> tuple[bool, tuple]:
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
    shape: tuple[int]
        The shape of the dataset. Empty tuple() object if not a dataset.
    """
    if not isinstance(object, h5py.Group):
        if isinstance(object, h5py.Dataset):
            dset: h5py.Dataset = object
            return True, dset.shape
        else:
            return True, tuple()
    else:
        return False, tuple()


def get_groups_and_datasets(hdf5_path_name_ext: Union[str, h5py.File]):
    """Get the structure of an HDF5 file, including all group and dataset names, and the dataset shapes.

    Parameters
    ----------
    hdf5_path_name_ext : Union[str, h5py.File]
        The HDF5 file to parse the structure of.

    Returns
    -------
    group_names: list[str]
        The absolute names of all the groups in the file. For example: "foo/bar"
    file_names_and_shapes: list[ tuple[str,tuple[int]] ]
        The absolute names of all the datasets in the file, and their shapes.
        For example: "foo/bar/baz", (1920,1080)
    """
    group_names: list[str] = []
    file_names_and_shapes: list[tuple[str, tuple[int]]] = []
    visited: list[tuple[str, bool, tuple]] = []

    def visitor(name: str, object: Union[h5py.Group, h5py.Dataset]):
        visited.append(tuple([name, *is_dataset_and_shape(object)]))
        return None

    if isinstance(hdf5_path_name_ext, str):
        hdf5_path_name_ext = os.path.normpath(hdf5_path_name_ext)
        with h5py.File(hdf5_path_name_ext, "r") as fin:
            fin.visititems(visitor)
    else:
        fin: h5py.File = hdf5_path_name_ext
        fin.visititems(visitor)

    for name, is_dataset, shape in visited:
        # Add to the file or group names list.
        # If a dataset, then include its shape.
        if not is_dataset:
            group_names.append(name)
        if is_dataset:
            file_names_and_shapes.append(tuple([name, shape]))

    return group_names, file_names_and_shapes


def load_hdf5_datasets(datasets: list[str], file: str):
    """Loads datasets from HDF5 file"""
    with h5py.File(file, "r") as f:
        kwargs: dict[str, Union[str, h5py.Dataset]] = {}
        # Loop through fields to retreive
        for dataset in datasets:
            # Get data and get dataset name
            data = f[dataset]
            name = dataset.split("/")[-1]

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
            kwargs.update({name: data})

    return kwargs


def _create_dataset_path(base_dir: str, h5_dataset_path_name: str, dataset_ext: str = ".txt"):
    dataset_location = os.path.dirname(h5_dataset_path_name)
    dataset_name = os.path.splitext(os.path.basename(h5_dataset_path_name))[0]
    dataset_path = os.path.normpath(os.path.join(base_dir, dataset_location))
    os.makedirs(dataset_path, exist_ok=True)
    return os.path.normpath(os.path.join(dataset_path, dataset_name + dataset_ext))


def extract_hdf5_to_directory(hdf5_path_name_ext: str, destination_dir: str) -> str:
    """Unpacks the given HDF5 file into the given destination directory.

    Unpacks the given HDF5 file into the given destination directory. A new
    directory is created in the destination with the same name as the hdf5 file.
    String values are extracted as .txt files, and images are extracted as .png
    files. Everything else is saved with numpy as .npy files.

    Parameters
    ----------
    hdf5_path_name_ext : str
        The HDF5 file to unpack.
    destination_dir : str
        The directory in which to create a directory for the HDF5 file.

    Returns
    -------
    output_dir: str
        The path to the newly created directory into which the HDF5 files were
        extracted into.
    """
    norm_path = os.path.normpath(hdf5_path_name_ext)
    name = os.path.splitext(norm_path)[1]
    hdf5_dir = os.path.normpath(os.path.join(destination_dir, name))

    # Create the HDF5 output directory
    os.makedirs(hdf5_dir, exist_ok=True)

    # Get all of what may be strings or images from the h5 file
    _, dataset_names_and_shapes = get_groups_and_datasets(norm_path)
    possible_strings: list[tuple[str, tuple[int]]] = []
    possible_images: list[tuple[str, tuple[int]]] = []
    other_datasets: list[tuple[str, tuple[int]]] = []
    for dataset_name, shape in dataset_names_and_shapes:
        possible_strings.append(tuple([dataset_name, shape]))

    # Extract strings into .txt files
    possible_strings_names = [t[0] for t in possible_strings]
    for i, possible_string_name in enumerate(possible_strings_names):
        dataset_name = possible_string_name.split("/")[-1]
        h5_val = load_hdf5_datasets([possible_string_name], norm_path)[dataset_name]
        if isinstance(h5_val, np.ndarray) and h5_val.ndim <= 1 and isinstance(h5_val.tolist()[0], str):
            h5_val = h5_val.tolist()[0]
        if isinstance(h5_val, str):
            dataset_path_name_ext = _create_dataset_path(hdf5_dir, possible_strings[i][0], ".txt")
            with open(dataset_path_name_ext, "w") as fout:
                fout.write(h5_val)
        else:
            possible_images.append(possible_strings[i])

    # Extract images into .png files
    possible_images_names = [t[0] for t in possible_images]
    for i, possible_image_name in enumerate(possible_images_names):
        dataset_name = possible_image_name.split("/")[-1]
        h5_val = load_hdf5_datasets([possible_image_name], norm_path)[dataset_name]
        shape = possible_images[i][1]
        if isinstance(h5_val, (h5py.Dataset, np.ndarray)):
            np_image = np.array(h5_val).squeeze()

            # we assume images have 2 or 3 dimensions
            if (len(shape) == 2) or (len(shape) == 3):
                # we assume shapes are at least 10x10 pixels and have an aspect ratio of at least 10:1
                aspect_ratio = max(shape[0], shape[1]) / min(shape[0], shape[1])
                if (shape[0] >= 10 and shape[1] >= 10) and (aspect_ratio < 10.001):
                    dataset_path_name_ext = _create_dataset_path(hdf5_dir, possible_images[i][0], ".png")
                    # assumed grayscale or RGB
                    if (len(shape) == 2) or (shape[2] in [1, 3]):
                        img = numpy_to_image(np_image)
                        img.save(dataset_path_name_ext)
                    else:  # assumed multiple images
                        dp = os.path.dirname(dataset_path_name_ext)
                        dn, de = os.path.splitext(
                            os.path.basename(dataset_path_name_ext)
                        )
                        for i in range(shape[2]):
                            dataset_path_name_ext_i = os.path.join(dp, f"{dn}_{i}{de}")
                            np_single_image = np_image[:, :, i].squeeze()
                            img = numpy_to_image(np_single_image)
                            img.save(dataset_path_name_ext_i)
                else:
                    other_datasets.append(possible_images[i])
            else:
                other_datasets.append(possible_images[i])
        else:
            other_datasets.append(possible_images[i])

    # Extract everything else into numpy or csv arrays
    other_dataset_names = [t[0] for t in other_datasets]
    for i, other_dataset_name in enumerate(other_dataset_names):
        dataset_name = other_dataset_name.split("/")[-1]
        h5_val = load_hdf5_datasets([other_dataset_name], norm_path)[dataset_name]
        np_val = np.array(h5_val)
        dataset_path_name = _create_dataset_path(hdf5_dir, other_datasets[i][0], "")

        # save as a numpy file
        np.save(dataset_path_name, np_val, allow_pickle=False)

    return hdf5_dir


class SensitiveStringsSearcher:
    _text_file_extensions = [".txt", ".csv", ".py", ".md", ".rst"]
    _text_file_path_name_exts = [".coverageac"]

    def __init__(
        self,
        root_search_dir: str,
        sensitive_strings_csv: str,
        allowed_binary_files_csv: str,
        cache_file_csv: str = None,
    ):
        self.root_search_dir = root_search_dir
        self.sensitive_strings_csv = sensitive_strings_csv
        self.allowed_binary_files_csv = allowed_binary_files_csv
        self.cache_file_csv = cache_file_csv
        self.verbose = False
        self._interactive = False
        self.verify_all_on_behalf_of_user = False
        self.remove_unfound_binaries = False
        self.date_time_str = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.tmp_dir_base = os.path.join(tempfile.gettempdir(), "SensitiveStringSearcher")
        self.git_files_only = True
        self.is_hdf5_searcher = False
        self.has_backed_up_allowed_binaries_csv = False

        self.matchers = self.build_matchers()
        self.matches: dict[str, list[ssm.Match]] = {}
        self.allowed_binary_files: list[ff.FileFingerprint] = []
        self.accepted_binary_files: list[ff.FileFingerprint] = []
        self.unknown_binary_files: list[ff.FileFingerprint] = []
        self.unfound_allowed_binary_files: list[ff.FileFingerprint] = []
        self.cached_cleared_files: list[fc.FileCache] = []
        self.new_cached_cleared_files: list[fc.FileCache] = []

    @property
    def interactive(self):
        return self._interactive or self.verify_all_on_behalf_of_user

    @interactive.setter
    def interactive(self, val: bool):
        self._interactive = val

    def __del__(self):
        for root, dirs, _ in os.walk(self.tmp_dir_base):
            for dir_name in dirs:
                if dir_name.startswith("tmp_"):
                    shutil.rmtree(os.path.join(root, dir_name))

    def build_matchers(self):
        matchers: list[ssm.SensitiveStringMatcher] = []
        with open(self.sensitive_strings_csv, newline="") as csvfile:
            reader = csv.reader(csvfile)
            next(reader)
            for row in reader:
                if row:
                    name = row[0]
                    patterns = row[1:]
                    matchers.append(
                        ssm.SensitiveStringMatcher(name, *patterns)
                    )
        return matchers

    def norm_path(self, file_path, file_name_ext: str):
        return os.path.normpath(os.path.join(self.root_search_dir, file_path, file_name_ext))

    def _is_file_in_cleared_cache(self, file_path: str, file_name_ext: str):
        cache_entry = fc.FileCache.for_file(self.root_search_dir, file_path, file_name_ext)
        if cache_entry in self.cached_cleared_files:
            return True
        return False

    def _register_file_in_cleared_cache(self, file_path: str, file_name_ext: str):
        cache_entry = fc.FileCache.for_file(self.root_search_dir, file_path, file_name_ext)
        self.new_cached_cleared_files.append(cache_entry)

    def _is_binary_file(self, rel_file_path: str, file_name_ext: str):
        is_binary_file = False

        # check if a binary file
        ext = os.path.splitext(file_name_ext)[1]
        ext = ext.lower()
        if ext == ".ipynb":
            is_binary_file = True
        elif self._is_img_ext(ext):
            if ext in self._text_file_extensions:
                is_binary_file = False
            elif (f"{rel_file_path}/{file_name_ext}" in self._text_file_path_name_exts) or (
                file_name_ext in self._text_file_path_name_exts
            ):
                is_binary_file = False
            else:
                is_binary_file = True
        elif os.path.getsize(self.norm_path(rel_file_path, file_name_ext)) > 1e6:
            # assume any file > 1MB is a binary file, in order to prevent
            # sensitive_strings from taking hours to check these files
            # needlessly
            is_binary_file = True
        if not is_binary_file:
            # attempt to parse the file as a text file
            try:
                file_path_norm: str = self.norm_path(rel_file_path, file_name_ext)
                with open(file_path_norm, newline="") as input_stream:
                    input_stream.readlines()
            except UnicodeDecodeError:
                is_binary_file = True

        return is_binary_file

    def _enqueue_unknown_binary_files_for_later_processing(self, rel_file_path: str, file_name_ext: str):
        """If the given file is recognized as an allowed file, and it's fingerprint matches the allowed file, then we
        can dismiss it from the list of unfound files and add it to the list of the accepted files.

        However, if the given file isn't recognized or it's fingerprint is different, then add it to the unknown list,
        to be dealt with later."""
        file_ff = ff.FileFingerprint.for_file(self.root_search_dir, rel_file_path, file_name_ext)

        if file_ff in self.allowed_binary_files:
            # we already know and trust this binary file
            with suppress(ValueError):
                self.unfound_allowed_binary_files.remove(file_ff)
            self.accepted_binary_files.append(file_ff)
        else:
            # we'll deal with unknown files as a group later
            self.unknown_binary_files.append(file_ff)

    def parse_file(self, rel_file_path: str, file_name_ext: str) -> list[str]:
        file_path_norm: str = self.norm_path(rel_file_path, file_name_ext)
        logger.debug(file_path_norm)

        if self._is_binary_file(rel_file_path, file_name_ext):
            return []
        else:
            with open(file_path_norm, newline="") as input_stream:
                return input_stream.readlines()

    def search_lines(self, lines: list[str]):
        matches: list[ssm.Match] = []

        for matcher in self.matchers:
            matches += matcher.check_lines(lines)

        return matches

    def search_file(self, file_path: str, file_name_ext: str):
        lines = self.parse_file(file_path, file_name_ext)

        matches: list[ssm.Match] = []
        matches += self.search_lines([file_path + "/" + file_name_ext])
        matches += self.search_lines(lines)

        return matches

    def get_tmp_dir(self):
        i = 0
        while True:
            ret = os.path.normpath(os.path.join(self.tmp_dir_base, f"tmp_{i}"))
            if os.path.isdir(ret):
                i += 1
            else:
                return ret

    def search_hdf5_file(self, hdf5_file: ff.FileFingerprint):
        norm_path = self.norm_path(hdf5_file.relative_path, hdf5_file.name_ext)
        relative_path_name_ext = f"{hdf5_file.relative_path}/{hdf5_file.name_ext}"
        matches: list[ssm.Match] = []

        # Extract the contents from the HDF5 file
        unzip_dir = self.get_tmp_dir()
        logger.info("")
        logger.info(f"**Extracting HDF5 file to {unzip_dir}**")
        h5_dir = extract_hdf5_to_directory(norm_path, unzip_dir)

        # Create a temporary allowed binary strings file
        fd, tmp_allowed_binary_csv = tempfile.mkstemp(
            dir=self.tmp_dir_base,
            suffix=".csv",
            text=True,
        )
        with open(self.allowed_binary_files_csv, "r") as fin:
            allowed_binary_files_lines = fin.readlines()
        with open(fd, "w") as fout:
            fout.writelines(allowed_binary_files_lines)

        # Create a searcher for the unzipped directory
        hdf5_searcher = SensitiveStringsSearcher(h5_dir, self.sensitive_strings_csv, tmp_allowed_binary_csv)
        hdf5_searcher.interactive = self.interactive
        hdf5_searcher.verify_all_on_behalf_of_user = self.verify_all_on_behalf_of_user
        hdf5_searcher.date_time_str = self.date_time_str
        hdf5_searcher.tmp_dir_base = self.tmp_dir_base
        hdf5_searcher.git_files_only = False
        hdf5_searcher.is_hdf5_searcher = True

        # Validate all of the unzipped files
        error = hdf5_searcher.search_files()
        hdf5_matches = hdf5_searcher.matches
        if error != 0:
            # There was an error, but the user may want to sign off on the file anyways.
            if len(hdf5_matches) > 0:
                # Describe the issues with the HDF5 file
                logger.warning(f"Found {len(hdf5_matches)} possible issues with the HDF5 file '{relative_path_name_ext}':")
                prev_relpath_name_ext = None
                for file_relpath_name_ext in hdf5_matches:
                    if prev_relpath_name_ext != file_relpath_name_ext:
                        logger.warning(f"    {file_relpath_name_ext}:")
                        prev_relpath_name_ext = file_relpath_name_ext
                    for match in hdf5_matches[file_relpath_name_ext]:
                        logger.warning(f"        {match.msg} (line {match.lineno}, col {match.colno})")

                # Ask the user about signing off
                if self.interactive:
                    if not self.verify_interactively(file_relpath_name_ext):
                        matches.append(ssm.Match(0, 0, 0, "", "", None, "HDF5 file denied by user"))
                else:  # if self.interactive
                    for file_relpath_name_ext in hdf5_matches:
                        for match in hdf5_matches[file_relpath_name_ext]:
                            path = os.path.basename(file_relpath_name_ext)
                            name = os.path.splitext(file_relpath_name_ext)[0]
                            dataset_name = path.replace("\\", "/") + "/" + name
                            match.msg = dataset_name + "::" + match.msg
                            matches.append(match)
            else:  # if len(hdf5_matches) > 0:
                message = (
                    "Programmer error in SensitiveStringsSearcher.search_hdf5_files(): "
                    + f"Errors were returned for file {relative_path_name_ext} but there were 0 matches found.",
                )
                logger.error(message)
                raise RuntimeError(message)

        else:
            # There were no errors, matches should be empty
            if len(hdf5_matches) > 0:
                message = (
                    "Programmer error in SensitiveStringsSearcher.search_hdf5_files(): "
                    + f"No errors were returned for file {relative_path_name_ext} but there were {len(hdf5_matches)} > 0 matches found.",
                )
                logger.error(message)
                raise RuntimeError(message)

        # Remove the temporary files created for the searcher.
        # Files created by the searcher should be removed in its __del__() method.
        os.remove(tmp_allowed_binary_csv)

        return matches

    def verify_interactively(self, relative_path_name_ext: str, cv_img: Image.Image = None, cv_title: str = None):
        if cv_img is None:
            logger.info("")
            logger.info("Unknown binary file:")
            logger.info("    " + relative_path_name_ext)
            logger.info("Is this unknown binary file safe to add, and doesn't contain any sensitive information (y/n)?")
            if self.verify_all_on_behalf_of_user:
                val = "y"
            else:
                resp = input("").strip()
                val = "n" if len(resp) == 0 else resp[0]
            logger.info(f"    User responded '{val}'")

        else:
            logger.info("")
            logger.info("Is this image safe to add, and doesn't contain any sensitive information (y/n)?")
            if self.verify_all_on_behalf_of_user:
                val = "y"
            else:
                cv2.imshow(cv_title, cv_img)
                key = cv2.waitKey(0)
                cv2.destroyAllWindows()
                time.sleep(0.1)  # small delay to prevent accidental double-bounces

                # Check for 'y' or 'n'
                if key == ord("y") or key == ord("Y"):
                    val = "y"
                elif key == ord("n") or key == ord("N"):
                    val = "n"
                else:
                    val = "?"
            if val.lower() in ["y", "n"]:
                logger.info(f"    User responded '{val}'")
            else:
                logger.error("Did not respond with either 'y' or 'n'. Assuming 'n'.")
                val = "n"

        return val.lower() == "y"

    def search_binary_file(self, binary_file: ff.FileFingerprint) -> list[ssm.Match]:
        norm_path = self.norm_path(binary_file.relative_path, binary_file.name_ext)
        ext = os.path.splitext(norm_path)[1]
        relative_path_name_ext = f"{binary_file.relative_path}/{binary_file.name_ext}"
        matches: list[ssm.Match] = []

        if ext.lower().lstrip(".") in pil_image_formats_rw:
            if self.interactive:
                if self.interactive_image_sign_off(file_ff=binary_file):
                    return []
                else:
                    matches.append(ssm.Match(0, 0, 0, "", "", None, "File denied by user"))
            else:
                matches.append(ssm.Match(0, 0, 0, "", "", None, "Unknown image file"))

        elif ext.lower() == ".h5":
            matches += self.search_hdf5_file(binary_file)

        else:
            if not self.verify_interactively(relative_path_name_ext):
                matches.append(ssm.Match(0, 0, 0, "", "", None, "Unknown binary file"))

        return matches

    def _is_img_ext(self, ext: str):
        return ext.lower().lstrip(".") in pil_image_formats_rw

    @staticmethod
    def numpy_to_image(arr: np.ndarray, rescale_or_clip="rescale", rescale_max=-1):
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
        allowed_int_types = [np.int8, np.uint8, np.int16, np.uint16, np.int32, np.uint32, np.int64, np.uint64]

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
        if np.iinfo(int_type).max > 255:
            if rescale_or_clip == "rescale":
                if rescale_max < 0:
                    rescale_max = np.max(arr)
                scale = 255 / rescale_max
                arr = arr * scale
                arr = np.clip(arr, 0, 255)
                arr = arr.astype(np.uint8)
            else:
                arr = np.clip(arr, 0, 255)
                arr = arr.astype(np.uint8)

        img = Image.fromarray(arr)
        return img

    def interactive_image_sign_off(
        self, np_image: np.ndarray = None, description: str = None, file_ff: ff.FileFingerprint = None
    ) -> bool:
        if (np_image is None) and (file_ff is not None):
            file_norm_path = self.norm_path(file_ff.relative_path, file_ff.name_ext)
            ext = os.path.splitext(file_norm_path)[1]
            if self._is_img_ext(ext):
                try:
                    img = Image.open(file_norm_path).convert("RGB")
                except:
                    img = None
                if img is not None:
                    np_image = np.copy(np.array(img))
                    img.close()
                    return self.interactive_image_sign_off(
                        np_image=np_image, description=f"{file_ff.relative_path}/{file_ff.name_ext}"
                    )
                else:
                    return self.verify_interactively(file_ff.relative_path)
                # if img is not None
            else:
                return False

        else:
            # rescale the image for easier viewing
            img = self.numpy_to_image(np_image)
            rescaled = ""
            if img.size[0] > 1920:
                scale = 1920 / img.size[0]
                img = img.resize((int(scale * img.size[0]), int(scale * img.size[1])))
                np_image = np.array(img)
                rescaled = " (downscaled)"
            if img.size[0] > 1080:
                scale = 1080 / img.size[1]
                img = img.resize((int(scale * img.size[0]), int(scale * img.size[1])))
                np_image = np.array(img)
                rescaled = " (downscaled)"

            # Show the image and prompt the user
            ret = self.verify_interactively(description, np_image, description + rescaled)
            return ret

    def _init_files_lists(self):
        self.matches.clear()

        if self.is_hdf5_searcher:
            # hdf5 searchers shouldn't be aware of what files are contained in the hdf5 file
            self.allowed_binary_files.clear()
        else:
            self.allowed_binary_files = [
                inst for inst, _ in ff.FileFingerprint.from_csv(
                    os.path.dirname(self.allowed_binary_files_csv),
                    os.path.basename(self.allowed_binary_files_csv),
                )
            ]
        self.accepted_binary_files.clear()
        self.unknown_binary_files.clear()
        self.unfound_allowed_binary_files = copy.copy(self.allowed_binary_files)

        self.cached_cleared_files.clear()
        self.new_cached_cleared_files.clear()
        sensitive_strings_cache = fc.FileCache.for_file(
            os.path.dirname(self.sensitive_strings_csv),
            "",
            os.path.basename(self.sensitive_strings_csv),
        )
        if self.cache_file_csv != None and os.path.isfile(self.cache_file_csv):
            self.cached_cleared_files = [
                inst for inst, _ in fc.FileCache.from_csv(
                    os.path.dirname(self.cache_file_csv),
                    os.path.basename(self.cache_file_csv),
                )
            ]
            if not sensitive_strings_cache in self.cached_cleared_files:
                self.cached_cleared_files.clear()
        self.new_cached_cleared_files.append(sensitive_strings_cache)

    def create_backup_allowed_binaries_csv(self):
        path = os.path.dirname(self.allowed_binary_files_csv)
        name, ext = os.path.splitext(os.path.basename(self.allowed_binary_files_csv))
        backup_name_ext = f"{name}_backup_{self.date_time_str}{ext}"
        backup_path_name_ext = os.path.join(path, backup_name_ext)
        if os.path.isfile(backup_path_name_ext):
            os.remove(backup_path_name_ext)
        shutil.copyfile(
            self.allowed_binary_files_csv,
            os.path.join(path, backup_name_ext),
        )
        self.has_backed_up_allowed_binaries_csv = True

    def update_allowed_binaries_csv(self):
        # Overwrite the allowed list csv file with the updated allowed_binary_files
        if not self.has_backed_up_allowed_binaries_csv:
            self.create_backup_allowed_binaries_csv()
        path = os.path.dirname(self.allowed_binary_files_csv)
        name = os.path.splitext(os.path.basename(self.allowed_binary_files_csv))[0]
        self.allowed_binary_files = sorted(self.allowed_binary_files)

        self.allowed_binary_files[0].to_csv(
            "Allowed Binary Files", path, name, rows=self.allowed_binary_files, overwrite=True
        )

    def search_files(self):
        self._init_files_lists()
        if self.git_files_only:
            # If this script is evaluated form MobaXTerm, then the built-in
            # 16-bit version of git will fail.
            git = shutil.which("git")
            if "mobaxterm" in git:
                git = "git"
            git_committed = subprocess.run(
                [git, "ls-tree", "--full-tree", "--name-only", "-r", "HEAD"],
                cwd=self.root_search_dir,
                stdout=subprocess.PIPE,
                text=True,
            )
            git_added = subprocess.run(
                [git, "diff", "--name-only", "--cached", "--diff-filter=A"],
                cwd=self.root_search_dir,
                stdout=subprocess.PIPE,
                text=True,
            )
            files = [line.val for line in git_committed + git_added]
            # don't include "git rm"'d files
            files = list(filter(lambda file: os.path.isfile(os.path.join(self.root_search_dir, file)), files))
            logger.info(f"Searching for sensitive strings in {len(files)} tracked files")
        else:
            files = []
            for directory, _, files_in_directory in os.walk(self.root_search_dir):
                for file_name in files_in_directory:
                    full_path = os.path.join(directory, file_name)
                    relative_path = os.path.relpath(full_path, self.root_search_dir)
                    files.append(relative_path)
            logger.info(f"Searching for sensitive strings in {len(files)} files")
        files = sorted(list(set(files)))

        # Search for sensitive strings in files
        matches: dict[str, list[ssm.Match]] = {}
        for file_path_name_ext in files:
            if self.verbose:
                logger.info(f"Searching file {file_path_name_ext}")
            rel_file_path = os.path.dirname(file_path_name_ext)
            file_name_ext = os.path.basename(file_path_name_ext)
            if self._is_file_in_cleared_cache(rel_file_path, file_name_ext):
                # file cleared in a previous run, don't need to check again
                self._register_file_in_cleared_cache(rel_file_path, file_name_ext)
            else:
                # need to check this file
                if self._is_binary_file(rel_file_path, file_name_ext):
                    # deal with non-parseable binary files as a group, below
                    self._enqueue_unknown_binary_files_for_later_processing(rel_file_path, file_name_ext)
                else:
                    # check text files for sensitive strings
                    file_matches = self.search_file(rel_file_path, file_name_ext)
                    if len(file_matches) > 0:
                        matches[file_path_name_ext] = file_matches
                    else:
                        self._register_file_in_cleared_cache(rel_file_path, file_name_ext)

        # Potentially remove unfound binary files
        if len(self.unfound_allowed_binary_files) > 0 and self.remove_unfound_binaries:
            for file in self.unfound_allowed_binary_files:
                self.allowed_binary_files.remove(file)
            self.unfound_allowed_binary_files.clear()
            self.update_allowed_binaries_csv()

        # Print initial information about matching files and problematic binary files
        if len(matches) > 0:
            logger.error(f"Found {len(matches)} files containing sensitive strings:")
            for file in matches:
                logger.error(f"    File {file}:")
                for match in matches[file]:
                    logger.error(f"        {match.msg}")
        if len(self.unfound_allowed_binary_files) > 0:
            logger.error(f"Expected {len(self.unfound_allowed_binary_files)} binary files that can't be found:")
            for file_ff in self.unfound_allowed_binary_files:
                logger.info("")
                logger.error(os.path.join(file_ff.relative_path, file_ff.name_ext))
        if len(self.unknown_binary_files) > 0:
            logger.warning(f"Found {len(self.unknown_binary_files)} unexpected binary files:")

        # Deal with unknown binary files
        if len(self.unknown_binary_files) > 0:
            unknowns_copy = copy.copy(self.unknown_binary_files)
            for file_ff in unknowns_copy:
                if self.verbose:
                    logger.info(f"Searching binary file {file_ff.relpath_name_ext}")
                logger.info("")
                logger.info(os.path.join(file_ff.relative_path, file_ff.name_ext))
                num_signed_binary_files = 0

                # Search for sensitive string matches, and interactively ask the user
                # about unparseable binary files.
                parsable_matches: list[ssm.Match] = self.search_binary_file(file_ff)

                if len(parsable_matches) == 0:
                    # No matches: this file is ok.
                    # Add the validated and/or signed off file to the allowed binary files csv
                    self.unknown_binary_files.remove(file_ff)
                    self.allowed_binary_files.append(file_ff)

                    # Overwrite the allowed list csv file with the updated allowed_binary_files
                    # and make a backup as necessary.
                    self.update_allowed_binaries_csv()

                    num_signed_binary_files += 1

                else:  # if len(parsable_matches) == 0:
                    # This file is not ok. Tell the user why.
                    logger.error(
                        f"    Found {len(parsable_matches)} possible sensitive issues in file {self.norm_path(file_ff.relative_path, file_ff.name_ext)}."
                    )
                    for _match in parsable_matches:
                        match: ssm.Match = _match
                        logger.error("    " + match.msg + f" (line {match.lineno}, col {match.colno})")

                # Date+time stamp the new allowed list csv files
                if num_signed_binary_files > 0:
                    path = os.path.dirname(self.allowed_binary_files_csv)
                    name, ext = os.path.splitext(os.path.basename(self.allowed_binary_files_csv))
                    abfc_stamped_name_ext = f"{name}_{self.date_time_str}{ext}"
                    abfc_stamped_path_name_ext = os.path.join(path, abfc_stamped_name_ext)
                    if os.path.isfile(abfc_stamped_path_name_ext):
                        os.remove(abfc_stamped_path_name_ext)
                    shutil.copyfile(
                        self.allowed_binary_files_csv,
                        os.path.join(path, abfc_stamped_name_ext),
                    )
            # for file_ff in unknowns_copy
        # if len(self.unknown_binary_files) > 0:

        # Make sure we didn't accidentally add any binary files to the cache
        for file_ff in self.allowed_binary_files + self.unfound_allowed_binary_files:
            for file_cf in self.new_cached_cleared_files:
                if file_ff.eq_aff(file_cf):
                    message = (
                        "Programmer error in sensitive_strings.search_files(): "
                        + "No binary files should be in the cache, but at least 1 such file was found: "
                        + f'"{file_cf.relative_path}/{file_cf.name_ext}"',
                    )
                    logger.error(message)
                    raise RuntimeError(message)

        # Save the cleared files cache
        for file_ff in self.unknown_binary_files:
            for file_cf in self.new_cached_cleared_files:
                if file_ff.eq_aff(file_cf):
                    self.new_cached_cleared_files.remove(file_cf)
                    break
        if self.cache_file_csv != None and len(self.new_cached_cleared_files) > 0:
            path = os.path.dirname(self.cache_file_csv)
            name = os.path.splitext(os.path.basename(self.cache_file_csv))[0]
            os.makedirs(path)
            self.new_cached_cleared_files[0].to_csv(
                "Cleared Files Cache", path, name, rows=self.new_cached_cleared_files, overwrite=True
            )

        # Executive summary
        info_or_warn = logger.info
        ret = len(matches) + len(self.unfound_allowed_binary_files) + len(self.unknown_binary_files)
        if ret > 0:
            info_or_warn = logger.warning
        info_or_warn("Summary:")
        info_or_warn("<<<PASS>>>" if ret == 0 else "<<<FAIL>>>")
        info_or_warn(f"Found {len(matches)} sensitive string matches")
        if len(self.unfound_allowed_binary_files) > 0:
            info_or_warn(f"Did not find {len(self.unfound_allowed_binary_files)} expected binary files")
        else:
            info_or_warn(f"Found {len(self.allowed_binary_files)} expected binary files")
        info_or_warn(f"Found {len(self.unknown_binary_files)} unexpected binary files")

        # Add a 'match' for any unfound or unknown binary files
        if not self.is_hdf5_searcher:
            for file_ff in self.unfound_allowed_binary_files:
                fpne = f"{file_ff.relative_path}/{file_ff.name_ext}"
                matches[fpne] = [] if (fpne not in matches) else matches[fpne]
                matches[fpne].append(ssm.Match(0, 0, 0, "", "", None, f"Unfound binary file {fpne}"))
        for file_ff in self.unknown_binary_files:
            fpne = f"{file_ff.relative_path}/{file_ff.name_ext}"
            matches[fpne] = [] if (fpne not in matches) else matches[fpne]
            matches[fpne].append(ssm.Match(0, 0, 0, "", "", None, f"Unknown binary file {fpne}"))

        self.matches = matches
        return ret


if __name__ == "__main__":
    parser = argparse.ArgumentParser(prog=__file__.rstrip(".py"), description="Sensitive strings searcher")
    parser.add_argument(
        "--no-interactive",
        action="store_true",
        dest="ninteractive",
        help="Don't interactively ask the user about unknown binary files. Simply fail instead.",
    )
    parser.add_argument(
        "--accept-all",
        action="store_true",
        dest="acceptall",
        help="Don't interactively ask the user about unknown binary files. Simply accept all as verified on the user's behalf. "
        + "This can be useful when you're confident that the only changes have been that the binary files have moved but not changed.",
    )
    parser.add_argument(
        "--accept-unfound",
        action="store_true",
        dest="acceptunfound",
        help="Don't fail because of unfound expected binary files. Instead remove the expected files from the list of allowed binaries. "
        + "This can be useful when you're confident that the only changes have been that the binary files have moved but not changed.",
    )
    parser.add_argument(
        "--log-dir",
        default=os.path.join(tempfile.gettempdir(), "sensitive_strings"),
        help="The directory in which to store all logs.",
    )
    parser.add_argument(
        "--sensitive-strings",
        help="The CSV file defining the sensitive string patterns to search for.",
        required=True,
    )
    parser.add_argument(
        "--allowed-binaries",
        help="The CSV file defining the allowed binary files.",
        required=True,
    )
    parser.add_argument(
        "--cache-file",
        default=None,
        help="The directory in which to store all logs.",
    )
    parser.add_argument('--verbose', action='store_true', dest="verbose", help="Print more information while running")
    args = parser.parse_args()
    not_interactive: bool = args.ninteractive
    accept_all: bool = args.acceptall
    remove_unfound_binaries: bool = args.acceptunfound
    verbose: bool = args.verbose

    ss_log_dir = os.path.normpath(args.log_dir)
    log_path = os.path.normpath(os.path.join(ss_log_dir, "sensitive_strings_log.txt"))
    sensitive_strings_csv = os.path.normpath(args.sensitive_strings)
    allowed_binary_files_csv = os.path.normpath(args.allowed_binaries)
    ss_cache_file = (
        os.path.normpath(args.cache_file)
        if args.cache_file
        else os.path.join(log_path, "cache.csv")
    )
    date_time_str = datetime.now().strftime("%Y%m%d_%H%M%S")

    log_already_exists = os.path.exists(log_path)
    path = os.path.dirname(log_path)
    name, ext = os.path.splitext(os.path.basename(log_path))
    log_path = os.path.join(path, f"{name}_{date_time_str}{ext}")
    logging.basicConfig(filename=log_path, level=logging.INFO)

    root_search_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..")
    searcher = SensitiveStringsSearcher(root_search_dir, sensitive_strings_csv, allowed_binary_files_csv, ss_cache_file)
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
