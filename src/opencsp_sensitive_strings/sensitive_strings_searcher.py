from __future__ import annotations

import csv
import logging
import shutil
import subprocess
import sys
import time
from contextlib import suppress
from datetime import datetime, timezone
from pathlib import Path
from tempfile import TemporaryDirectory, gettempdir
from uuid import uuid4

import cv2
import numpy as np
from PIL import UnidentifiedImageError
from PIL.Image import Image
from PIL.Image import open as open_image
from rich.prompt import Confirm

from opencsp_sensitive_strings.config import Config
from opencsp_sensitive_strings.csv_interface import write_to_csv
from opencsp_sensitive_strings.file_cache import FileCache
from opencsp_sensitive_strings.file_fingerprint import FileFingerprint
from opencsp_sensitive_strings.hdf5_extraction import extract_hdf5_to_directory
from opencsp_sensitive_strings.image import is_image, numpy_to_image
from opencsp_sensitive_strings.sensitive_string_matcher import (
    Match,
    SensitiveStringMatcher,
)

logger = logging.getLogger(__name__)
MAX_WIDTH, MAX_HEIGHT = 1920, 1080
MAX_FILE_SIZE = 1e6


class SensitiveStringsSearcher:
    """
    TODO:  Fill this out.
    """

    def __init__(self) -> None:
        self.accepted_binary_files: list[FileFingerprint] = []
        self.allowed_binary_files: list[FileFingerprint] = []
        self.cached_cleared_files: list[FileCache] = []
        self.config = Config()
        self.date_time_str = datetime.now(tz=timezone.utc).strftime(
            "%Y%m%d_%H%M%S"
        )
        self.is_hdf5_searcher = False
        self.matchers: list[SensitiveStringMatcher] = []
        self.matches: dict[Path, list[Match]] = {}
        """
        A mapping from files to the sensitive string matches found therein.
        """
        self.new_cached_cleared_files: list[FileCache] = []
        self.tmp_dir_base = Path(gettempdir()) / "SensitiveStringSearcher"
        self.unfound_allowed_binary_files: list[FileFingerprint] = []
        """
        Binary files given via ``--allowed-binaries`` that don't appear
        to be present in the root search directory.
        """
        self.unknown_binary_files: list[FileFingerprint] = []
        """
        Binary files discovered that the tool doesn't know about via
        ``--allowed-binaries``.
        """
        self.tmp_dir_base.mkdir(parents=True, exist_ok=True)

    def run(self, *, git_files_only: bool) -> int:
        self.build_matchers()
        self.populate_file_lists()
        files = sorted(
            set(
                self.get_tracked_files()
                if git_files_only
                else self.get_files_in_directory()
            )
        )
        self.process_files(files)
        self.print_match_information()
        self.process_unknown_binary_files()
        self.ensure_no_binaries_in_cache()
        self.save_cache()
        return self.number_of_findings()

    def build_matchers(self) -> None:
        """
        Build matchers from the sensitive strings CSV file.
        """
        with self.config.sensitive_strings_csv.open(newline="") as csv_file:
            reader = csv.reader(csv_file)
            next(reader)
            self.matchers = [
                SensitiveStringMatcher(row[0], *row[1:])
                for row in reader
                if row
            ]

    def populate_file_lists(self) -> None:
        if not self.is_hdf5_searcher:
            self.allowed_binary_files = [
                file
                for file, _ in FileFingerprint.from_csv(
                    self.config.allowed_binary_files_csv
                )
            ]
        self.unfound_allowed_binary_files = self.allowed_binary_files.copy()
        sensitive_strings_cache = FileCache.for_file(
            self.config.sensitive_strings_csv.parent,
            Path(self.config.sensitive_strings_csv.name),
        )
        if self.config.cache_file_csv and self.config.cache_file_csv.is_file():
            self.cached_cleared_files = [
                file
                for file, _ in FileCache.from_csv(self.config.cache_file_csv)
            ]
            if sensitive_strings_cache not in self.cached_cleared_files:
                self.cached_cleared_files.clear()
        self.new_cached_cleared_files.append(sensitive_strings_cache)

    def get_tracked_files(self) -> list[Path]:
        files: list[Path] = []
        git = self.get_git_command()
        for command in [
            [git, "ls-tree", "--full-tree", "--name-only", "-r", "HEAD"],
            [git, "diff", "--name-only", "--cached", "--diff-filter=A"],
        ]:
            completed_process = subprocess.run(  # noqa: S603
                command,
                check=True,
                cwd=self.config.root_search_dir,
                stdout=subprocess.PIPE,
                text=True,
            )
            files.extend(
                Path(file)
                for file in completed_process.stdout.splitlines()
                if (self.config.root_search_dir / file).is_file()
            )
        logger.info(
            "Searching for sensitive strings in %d tracked files", len(files)
        )
        return files

    @staticmethod
    def get_git_command() -> str:
        """
        Determine the git command to use for getting tracked files.

        Note:
            If this script is evaluated from MobaXTerm, then the
            built-in 16-bit version of git will fail.

        Returns:
            The git executable.
        """
        if (git := shutil.which("git")) is None:
            message = "'git' executable not found."
            raise RuntimeError(message)
        if "mobaxterm" in git:
            git = "git"
        return git

    def get_files_in_directory(self) -> list[Path]:
        files = [
            file.relative_to(self.config.root_search_dir)
            for file in self.config.root_search_dir.rglob("*")
            if file.is_file()
        ]
        logger.info("Searching for sensitive strings in %d files", len(files))
        return files

    def process_files(self, files: list[Path]) -> None:
        """
        Search for sensitive strings in files.

        TODO:  Rename unfound_allowed_binary_files.

        Args:
            files:  The files in which to search.
        """
        for file in files:
            logger.debug("Searching file %s", file)
            cache_entry = FileCache.for_file(self.config.root_search_dir, file)
            if cache_entry in self.cached_cleared_files:
                self.new_cached_cleared_files.append(cache_entry)
            elif self.is_binary(file):
                self.save_binary_for_later_processing(file)
            elif file_matches := self.search_text_file(file):
                self.matches[file] = file_matches
            else:
                self.new_cached_cleared_files.append(cache_entry)
        if (
            self.unfound_allowed_binary_files
            and self.config.remove_unfound_binaries
        ):
            self.remove_unfound_binary_files()

    def is_binary(self, file: Path) -> bool:
        """
        Determine whether the given file is a binary file.

        Note:
            Any file over 1MB in size is assumed to be a binary file to
            prevent the tool from taking hours to check these files
            needlessly.

        Args:
            file:  The file to check.

        Returns:
            Whether it's binary.
        """
        file = self.full_path(file)
        if (
            file.suffix.lower() == ".ipynb"
            or is_image(file)
            or file.stat().st_size > MAX_FILE_SIZE
        ):
            return True
        try:
            with file.open(newline="") as input_stream:
                input_stream.readlines()
                return False
        except UnicodeDecodeError:
            return True

    def full_path(self, relative_path: Path) -> Path:
        """
        Get the full path to the given file.

        Args:
            relative_path:  The path relative to the root search
                directory.

        Returns:
            The absolute path to the file.
        """
        return (self.config.root_search_dir / relative_path).resolve()

    def save_binary_for_later_processing(self, file: Path) -> None:
        """
        Possibly save a binary file to process later.

        If the given file is recognized as an allowed file, then we can
        dismiss it from the list of unfound files and add it to the list
        of the accepted files.  However, if the given file isn't
        recognized, then add it to the unknown list to be dealt with
        later.

        Args:
            file:  The binary file in question.
        """
        fingerprint = FileFingerprint.for_file(
            self.config.root_search_dir, file
        )
        if fingerprint in self.allowed_binary_files:
            with suppress(ValueError):
                self.unfound_allowed_binary_files.remove(fingerprint)
            self.accepted_binary_files.append(fingerprint)
        else:
            self.unknown_binary_files.append(fingerprint)

    def search_text_file(self, file: Path) -> list[Match]:
        """
        Find any sensitive string matches in the given text file.

        Also check the file name/path itself.

        Args:
            file:  The file to search.

        Returns:
            A (possibly empty) list of sensitive string matches.
        """
        file = self.full_path(file)
        logger.debug(file)
        with file.open(newline="") as input_stream:
            lines = input_stream.readlines()
        return [
            match
            for matcher in self.matchers
            for match in (
                matcher.check_lines([f"{file}"]) + matcher.check_lines(lines)
            )
        ]

    def remove_unfound_binary_files(self) -> None:
        self.allowed_binary_files = [
            _
            for _ in self.allowed_binary_files
            if _ not in self.unfound_allowed_binary_files
        ]
        self.unfound_allowed_binary_files.clear()
        self.update_allowed_binaries_csv()

    def update_allowed_binaries_csv(self) -> None:
        """Overwrite the allowed binaries CSV file with the current list."""
        self.create_backup_allowed_binaries_csv()
        self.allowed_binary_files = sorted(self.allowed_binary_files)
        write_to_csv(
            self.config.allowed_binary_files_csv, self.allowed_binary_files
        )

    def create_backup_allowed_binaries_csv(self) -> None:
        """Create a backup of the allowed binaries CSV file."""
        file = self.config.allowed_binary_files_csv
        backup = (
            file.parent
            / f"{file.stem}_backup_{self.date_time_str}{file.suffix}"
        )
        if backup.is_file():
            backup.unlink()
        shutil.copyfile(self.config.allowed_binary_files_csv, backup)

    def print_match_information(self) -> None:
        """
        Print initial match information.

        Both files with sensitive strings in them, and problematic
        binary files.
        """
        if self.matches:
            logger.error(
                "Found %d files containing sensitive strings:\n%s",
                len(self.matches),
                "\n".join(
                    f"    File {file}:\n"
                    + "\n".join(
                        f"        {match.message}" for match in file_matches
                    )
                    for file, file_matches in self.matches.items()
                ),
            )
        if self.unfound_allowed_binary_files:
            logger.error(
                "Expected %d binary files that can't be found:\n%s",
                len(self.unfound_allowed_binary_files),
                "\n".join(
                    f"    {file_fingerprint.relative_path}"
                    for file_fingerprint in self.unfound_allowed_binary_files
                ),
            )
        if self.unknown_binary_files:
            logger.warning(
                "Found %d unexpected binary files:",
                len(self.unknown_binary_files),
            )

    def process_unknown_binary_files(self) -> None:
        """
        Process unknown binary files.

        Deal with binary files discovered that the tool doesn't know
        about via ``--allowed-binaries``.
        """
        approved_files: list[FileFingerprint] = []
        for file in self.unknown_binary_files:
            logger.debug("Searching binary file %s", file.relative_path)
            if possible_matches := self.search_binary_file(file):
                logger.error(
                    "    Found %d possible sensitive issues in file %s:\n%s",
                    len(possible_matches),
                    self.full_path(file.relative_path),
                    "\n".join(
                        f"    {match.message}" for match in possible_matches
                    ),
                )
            else:
                approved_files.append(file)
        for file in approved_files:
            self.unknown_binary_files.remove(file)
            self.allowed_binary_files.append(file)
        if approved_files:
            self.update_allowed_binaries_csv()

    def search_binary_file(self, binary_file: FileFingerprint) -> list[Match]:
        file = self.full_path(binary_file.relative_path)
        if is_image(file):
            if self.config.interactive:
                if image := self.get_image_from_fingerprint(binary_file):
                    if not self.config.assume_yes:
                        self.show_image(file, image)
                    return self.interactive_file_matches(
                        file, "File denied by user"
                    )
                if self.file_approved_by_user(file):
                    return []
            return [Match(0, 0, 0, "", "", "Unknown image file")]
        if file.suffix.lower() == ".h5":
            return self.search_hdf5_file(binary_file)
        return self.interactive_file_matches(file, "Unknown binary file")

    def get_image_from_fingerprint(
        self, file_fingerprint: FileFingerprint
    ) -> Image | None:
        file = self.full_path(file_fingerprint.relative_path)
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

    def show_image(self, file: Path, image: Image) -> bool:
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
        return self.file_approved_by_user(file)

    def interactive_file_matches(
        self, file: Path, message: str
    ) -> list[Match]:
        return (
            []
            if self.file_approved_by_user(file)
            else [Match(0, 0, 0, "", "", message)]
        )

    def file_approved_by_user(self, file: Path) -> bool:
        if self.config.assume_yes:
            return True
        return Confirm.ask(
            f"File:  {file}.  Is this file safe to add?  Does it contain no "
            "sensitive information?"
        )

    def search_hdf5_file(self, hdf5_file: FileFingerprint) -> list[Match]:
        with TemporaryDirectory() as temp_dir:
            # Extract the contents from the HDF5 file
            logger.info("Extracting HDF5 file to %s", temp_dir)
            h5_dir = extract_hdf5_to_directory(
                self.full_path(hdf5_file.relative_path), Path(temp_dir)
            )

            # Create a temporary allowed binary strings file
            tmp_allowed_binary_csv = self.tmp_dir_base / f"{uuid4()}.csv"
            tmp_allowed_binary_csv.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(
                self.config.allowed_binary_files_csv, tmp_allowed_binary_csv
            )

            # Create a searcher for the unzipped directory
            hdf5_searcher = SensitiveStringsSearcher()
            hdf5_searcher.config = self.config
            hdf5_searcher.config.allowed_binary_files_csv = (
                tmp_allowed_binary_csv
            )
            hdf5_searcher.config.cache_file_csv = None
            hdf5_searcher.config.remove_unfound_binaries = False
            hdf5_searcher.config.root_search_dir = h5_dir
            hdf5_searcher.date_time_str = self.date_time_str
            hdf5_searcher.is_hdf5_searcher = True
            hdf5_searcher.tmp_dir_base = self.tmp_dir_base

            # Validate all of the unzipped files
            error = hdf5_searcher.run(git_files_only=False)
            if error != 0:
                return self.handle_hdf5_error(
                    hdf5_searcher.matches, hdf5_file.relative_path
                )
            if hdf5_searcher.matches:
                message = (
                    "No errors were returned for file "
                    f"{hdf5_file.relative_path} but there were "
                    f"{len(hdf5_searcher.matches)} matches found."
                )
                logger.error(message)
                raise RuntimeError(message)
        return []

    def handle_hdf5_error(
        self,
        hdf5_matches: dict[Path, list[Match]],
        relative_path: Path,
    ) -> list[Match]:
        # There was an error, but the user may want to sign off on
        # the file anyway.
        if not hdf5_matches:
            message = (
                f"Errors were returned for file {relative_path} but there "
                "were 0 matches found."
            )
            logger.error(message)
            raise RuntimeError(message)

        # Describe the issues with the HDF5 file
        logger.warning(
            "Found %d possible issues with the HDF5 file %s:\n%s",
            len(hdf5_matches),
            relative_path,
            "\n".join(
                f"    {file}:\n"
                + "\n".join(
                    f"        {match.message}" for match in file_matches
                )
                for file, file_matches in hdf5_matches.items()
            ),
        )

        # Ask the user about signing off
        if self.config.interactive:
            if self.file_approved_by_user(self.full_path(relative_path)):
                return []
            return [Match(0, 0, 0, "", "", "HDF5 file denied by user")]
        matches: list[Match] = []
        for file, file_matches in hdf5_matches.items():
            for match in file_matches:
                dataset_name = file.with_suffix("")
                match.message = f"{dataset_name}::{match.message}"
                matches.append(match)
        return matches

    def ensure_no_binaries_in_cache(self) -> None:
        binary_paths = {
            _.relative_path
            for _ in (
                self.allowed_binary_files + self.unfound_allowed_binary_files
            )
        }
        cached_paths = {_.relative_path for _ in self.new_cached_cleared_files}
        cached_binaries = binary_paths & cached_paths
        if cached_binaries:
            files = "\n".join([f"    {_}" for _ in cached_binaries])
            message = (
                "No binary files should be in the cache, but the following "
                f"binary files were found:\n{files}"
            )
            logger.error(message)
            raise RuntimeError(message)

    def save_cache(self) -> None:
        for file_fingerprint in self.unknown_binary_files:
            for file in self.new_cached_cleared_files:
                if file.relative_path == file_fingerprint.relative_path:
                    self.new_cached_cleared_files.remove(file)
                    break
        if self.config.cache_file_csv and self.new_cached_cleared_files:
            self.config.cache_file_csv.parent.mkdir(
                parents=True, exist_ok=True
            )
            write_to_csv(
                self.config.cache_file_csv,
                self.new_cached_cleared_files,
            )

    def number_of_findings(self) -> int:
        findings = (
            len(self.matches)
            + len(self.unfound_allowed_binary_files)
            + len(self.unknown_binary_files)
        )
        expected_binary_message = (
            f"Did not find {len(self.unfound_allowed_binary_files)} expected "
            "binary files."
            if self.unfound_allowed_binary_files
            else f"Found {len(self.allowed_binary_files)} expected binary "
            "files."
        )
        log = logger.warning if findings else logger.info
        log(
            "Summary:\n%s\nFound %d sensitive string matches.\n%s\nFound %s "
            "unexpected binary files.",
            "<<<FAIL>>>" if findings else "<<<PASS>>>",
            len(self.matches),
            expected_binary_message,
            len(self.unknown_binary_files),
        )
        self.add_binary_matches(self.unfound_allowed_binary_files, "Unfound")
        self.add_binary_matches(self.unknown_binary_files, "Unknown")
        return findings

    def add_binary_matches(
        self, binaries: list[FileFingerprint], label: str
    ) -> None:
        # TODO:  Why does this method exist if self.matches isn't used after?
        for file in binaries:
            message = f"{label} binary file {file.relative_path}"
            self.matches.setdefault(file.relative_path, []).append(
                Match(0, 0, 0, "", "", message)
            )


if __name__ == "__main__":
    searcher = SensitiveStringsSearcher()
    searcher.config.parse_args(sys.argv[1:])
    num_errors = searcher.run(git_files_only=True)
    if num_errors > 0:
        sys.exit(1)
    else:
        sys.exit(0)
