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
from typing import Optional

import cv2
import numpy as np
from PIL import UnidentifiedImageError
from PIL.Image import Image
from PIL.Image import open as open_image

from opencsp_sensitive_strings.csv_interface import write_to_csv
from opencsp_sensitive_strings.file_cache import FileCache
from opencsp_sensitive_strings.file_fingerprint import FileFingerprint
from opencsp_sensitive_strings.hdf5_extraction import extract_hdf5_to_directory
from opencsp_sensitive_strings.image_conversion import numpy_to_image
from opencsp_sensitive_strings.sensitive_string_matcher import (
    Match,
    SensitiveStringMatcher,
)

logger = logging.getLogger(__name__)
FULL_DEFINITION_SIZE = (1920, 1080)
MAX_FILE_SIZE = 1e6
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
        self.matches: dict[Path, list[Match]] = {}
        self.allowed_binary_files: list[FileFingerprint] = []
        self.accepted_binary_files: list[FileFingerprint] = []
        self.unknown_binary_files: list[FileFingerprint] = []
        self.unfound_allowed_binary_files: list[FileFingerprint] = []
        self.cached_cleared_files: list[FileCache] = []
        self.new_cached_cleared_files: list[FileCache] = []

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

    def build_matchers(self) -> list[SensitiveStringMatcher]:
        matchers: list[SensitiveStringMatcher] = []
        with self.sensitive_strings_csv.open(newline="") as csv_file:
            reader = csv.reader(csv_file)
            next(reader)
            for row in reader:
                if row:
                    name = row[0]
                    patterns = row[1:]
                    matchers.append(SensitiveStringMatcher(name, *patterns))
        return matchers

    def norm_path(self, file_path: Path) -> Path:
        return self.root_search_dir / file_path

    def _is_file_in_cleared_cache(self, file_path: Path) -> bool:
        cache_entry = FileCache.for_file(self.root_search_dir, file_path)
        return cache_entry in self.cached_cleared_files

    def _register_file_in_cleared_cache(self, file_path: Path) -> None:
        cache_entry = FileCache.for_file(self.root_search_dir, file_path)
        self.new_cached_cleared_files.append(cache_entry)

    def _is_binary_file(self, rel_file_path: Path) -> bool:
        is_binary_file = False

        # check if a binary file
        ext = rel_file_path.suffix.lower()
        if ext == ".ipynb":
            is_binary_file = True
        elif self._is_img_ext(rel_file_path):
            is_binary_file = (
                (ext not in self._text_file_extensions)
                and (f"{rel_file_path}" not in self._text_file_path_name_exts)
                and (rel_file_path.name not in self._text_file_path_name_exts)
            )
        elif self.norm_path(rel_file_path).stat().st_size > MAX_FILE_SIZE:
            # assume any file > 1MB is a binary file, in order to prevent
            # sensitive_strings from taking hours to check these files
            # needlessly
            is_binary_file = True
        if not is_binary_file:
            # attempt to parse the file as a text file
            try:
                file_path_norm: Path = self.norm_path(rel_file_path)
                with file_path_norm.open(newline="") as input_stream:
                    input_stream.readlines()
            except UnicodeDecodeError:
                is_binary_file = True
        return is_binary_file

    def _enqueue_unknown_binary_files_for_later_processing(
        self, rel_file_path: Path
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
        file_ff = FileFingerprint.for_file(self.root_search_dir, rel_file_path)
        if file_ff in self.allowed_binary_files:
            # we already know and trust this binary file
            with suppress(ValueError):
                self.unfound_allowed_binary_files.remove(file_ff)
            self.accepted_binary_files.append(file_ff)
        else:
            # we'll deal with unknown files as a group later
            self.unknown_binary_files.append(file_ff)

    def parse_file(self, rel_file_path: Path) -> list[str]:
        file_path_norm: Path = self.norm_path(rel_file_path)
        logger.debug(file_path_norm)
        if self._is_binary_file(rel_file_path):
            return []
        with file_path_norm.open(newline="") as input_stream:
            return input_stream.readlines()

    def search_lines(self, lines: list[str]) -> list[Match]:
        matches: list[Match] = []
        for matcher in self.matchers:
            matches += matcher.check_lines(lines)
        return matches

    def search_file(self, file_path: Path) -> list[Match]:
        lines = self.parse_file(file_path)
        matches: list[Match] = []
        matches += self.search_lines([f"{file_path}"])
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
        matches: list[Match],
        hdf5_matches: dict[Path, list[Match]],
        relative_path: Path,
    ) -> None:
        # There was an error, but the user may want to sign off on
        # the file anyway.
        if len(hdf5_matches) == 0:
            message = (
                f"Errors were returned for file {relative_path} but there "
                "were 0 matches found."
            )
            logger.error(message)
            raise RuntimeError(message)

        # Describe the issues with the HDF5 file
        logger.warning(
            "Found possible issues with the HDF5 file:",
            extra={
                "number_of_issues": len(hdf5_matches),
                "file": relative_path,
            },
        )
        previous_file = None
        for file, file_matches in hdf5_matches.items():
            if previous_file != file:
                logger.warning("    %s:", file)
                previous_file = file
            for match in file_matches:
                logger.warning(
                    "        %s (line %d, col %d)",
                    match.msg,
                    match.lineno,
                    match.colno,
                )

        # Ask the user about signing off
        if self.interactive:
            if not self.verify_interactively(file):
                matches.append(
                    Match(0, 0, 0, "", "", None, "HDF5 file denied by user")
                )
        else:  # if self.interactive
            for file, file_matches in hdf5_matches.items():
                for match in file_matches:
                    dataset_name = file.with_suffix("")
                    match.msg = f"{dataset_name}::{match.msg}"
                    matches.append(match)

    def search_hdf5_file(self, hdf5_file: FileFingerprint) -> list[Match]:
        norm_path = self.norm_path(hdf5_file.relative_path)
        matches: list[Match] = []

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
                matches, hdf5_matches, hdf5_file.relative_path
            )
        elif len(hdf5_matches) > 0:
            message = (
                f"No errors were returned for file {hdf5_file.relative_path} "
                f"but there were {len(hdf5_matches)} > 0 matches found."
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
        file: Path,
        cv_img: Optional[Image] = None,
        cv_title: Optional[str] = None,
    ) -> bool:
        if cv_img is None:
            logger.info("")
            logger.info("Unknown binary file:")
            logger.info("    %s", file)
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

    def _is_img_ext(self, file: Path) -> bool:
        return file.suffix.lower().lstrip(".") in PIL_IMAGE_FORMATS_RW

    def search_binary_file(self, binary_file: FileFingerprint) -> list[Match]:
        norm_path = self.norm_path(binary_file.relative_path)
        matches: list[Match] = []
        if self._is_img_ext(norm_path):
            if self.interactive:
                if self.interactive_image_sign_off(
                    norm_path, file_ff=binary_file
                ):
                    return []
                matches.append(
                    Match(0, 0, 0, "", "", None, "File denied by user")
                )
            else:
                matches.append(
                    Match(0, 0, 0, "", "", None, "Unknown image file")
                )
        elif norm_path.suffix.lower() == ".h5":
            matches += self.search_hdf5_file(binary_file)
        elif not self.verify_interactively(binary_file.relative_path):
            matches.append(Match(0, 0, 0, "", "", None, "Unknown binary file"))
        return matches

    def interactive_image_sign_off(
        self,
        description: Path,
        np_image: Optional[np.ndarray] = None,
        file_ff: Optional[FileFingerprint] = None,
    ) -> bool:
        if (np_image is None) and (file_ff is not None):
            file_norm_path = self.norm_path(file_ff.relative_path)
            if self._is_img_ext(file_norm_path):
                try:
                    img = open_image(file_norm_path).convert("RGB")
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
                        file_ff.relative_path,
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
                for inst, _ in FileFingerprint.from_csv(
                    self.allowed_binary_files_csv
                )
            ]
        self.accepted_binary_files.clear()
        self.unknown_binary_files.clear()
        self.unfound_allowed_binary_files = copy.copy(
            self.allowed_binary_files
        )
        self.cached_cleared_files.clear()
        self.new_cached_cleared_files.clear()
        sensitive_strings_cache = FileCache.for_file(
            self.sensitive_strings_csv.parent,
            Path(self.sensitive_strings_csv.name),
        )
        if self.cache_file_csv is not None and self.cache_file_csv.is_file():
            self.cached_cleared_files = [
                inst for inst, _ in FileCache.from_csv(self.cache_file_csv)
            ]
            if sensitive_strings_cache not in self.cached_cleared_files:
                self.cached_cleared_files.clear()
        self.new_cached_cleared_files.append(sensitive_strings_cache)

    def create_backup_allowed_binaries_csv(self) -> None:
        backup = self.allowed_binary_files_csv.parent / (
            f"{self.allowed_binary_files_csv.stem}_backup_"
            f"{self.date_time_str}{self.allowed_binary_files_csv.suffix}"
        )
        if backup.is_file():
            backup.unlink()
        shutil.copyfile(self.allowed_binary_files_csv, backup)
        self.has_backed_up_allowed_binaries_csv = True

    def update_allowed_binaries_csv(self) -> None:
        # Overwrite the allowed list CSV file with the updated
        # allowed_binary_files.
        if not self.has_backed_up_allowed_binaries_csv:
            self.create_backup_allowed_binaries_csv()
        self.allowed_binary_files = sorted(self.allowed_binary_files)
        write_to_csv(
            self.allowed_binary_files_csv,
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
                relative_path = os.path.relpath(
                    Path(directory) / file_name, self.root_search_dir
                )
                files.append(Path(relative_path))
        logger.info(
            "Searching for sensitive strings in files",
            extra={"number_of_files": len(files)},
        )
        return files

    def handle_files(self, files: list[Path]) -> dict[Path, list[Match]]:
        # Search for sensitive strings in files
        matches: dict[Path, list[Match]] = {}
        for file in files:
            if self.verbose:
                logger.info("Searching file %s", file)
            if self._is_file_in_cleared_cache(file):
                # file cleared in a previous run, don't need to check again
                self._register_file_in_cleared_cache(file)
            # need to check this file
            elif self._is_binary_file(file):
                # deal with non-parsable binary files as a group, below
                self._enqueue_unknown_binary_files_for_later_processing(file)
            else:
                # check text files for sensitive strings
                file_matches = self.search_file(file)
                if len(file_matches) > 0:
                    matches[file] = file_matches
                else:
                    self._register_file_in_cleared_cache(file)

        # Potentially remove unfound binary files
        if (
            len(self.unfound_allowed_binary_files) > 0
            and self.remove_unfound_binaries
        ):
            self.allowed_binary_files = [
                _
                for _ in self.allowed_binary_files
                if _ not in self.unfound_allowed_binary_files
            ]
            self.unfound_allowed_binary_files.clear()
            self.update_allowed_binaries_csv()
        return matches

    def print_match_information(
        self, matches: dict[Path, list[Match]]
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
                logger.error(file_ff.relative_path)
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
                logger.info("Searching binary file %s", file_ff.relative_path)
            logger.info("")
            logger.info(file_ff.relative_path)
            num_signed_binary_files = 0
            if parsable_matches := self.search_binary_file(file_ff):
                # This file is not ok. Tell the user why.
                logger.error(
                    "    Found possible sensitive issues in file.",
                    extra={
                        "number_of_issues": len(parsable_matches),
                        "file": self.norm_path(file_ff.relative_path),
                    },
                )
                for match in parsable_matches:
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
                stamped = self.allowed_binary_files_csv.parent / (
                    f"{self.allowed_binary_files_csv.stem}_"
                    + self.date_time_str
                    + self.allowed_binary_files_csv.suffix
                )
                if stamped.is_file():
                    stamped.unlink()
                shutil.copyfile(self.allowed_binary_files_csv, stamped)
        # for file_ff in unknowns_copy

    def check_cache(self) -> None:
        # Make sure we didn't accidentally add any binary files to the cache
        for file_ff in (
            self.allowed_binary_files + self.unfound_allowed_binary_files
        ):
            for file_cf in self.new_cached_cleared_files:
                if file_ff == file_cf:
                    message = (
                        "No binary files should be in the cache, but at least "
                        f'1 such file was found:  "{file_cf.relative_path}"'
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
            write_to_csv(
                self.cache_file_csv,
                self.new_cached_cleared_files,
            )

    def number_of_findings(self, matches: dict[Path, list[Match]]) -> int:
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
                matches[file_ff.relative_path] = matches.get(
                    file_ff.relative_path, []
                )
                matches[file_ff.relative_path].append(
                    Match(
                        0,
                        0,
                        0,
                        "",
                        "",
                        None,
                        f"Unfound binary file {file_ff.relative_path}",
                    )
                )
        for file_ff in self.unknown_binary_files:
            matches[file_ff.relative_path] = matches.get(
                file_ff.relative_path, []
            )
            matches[file_ff.relative_path].append(
                Match(
                    0,
                    0,
                    0,
                    "",
                    "",
                    None,
                    f"Unknown binary file {file_ff.relative_path}",
                )
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
