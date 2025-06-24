"""
Handles file caching with CSV serialization.
"""

import dataclasses
from datetime import datetime, timezone
from pathlib import Path

from typing_extensions import Self

from opencsp_sensitive_strings.csv_interface import CsvInterface


@dataclasses.dataclass()
class FileCache(CsvInterface):
    """
    Represents a cached file with its last modified timestamp.
    """

    last_modified: str
    """The system time at which the file was last modified."""

    @classmethod
    def from_csv_line(cls, data: list[str]) -> tuple[Self, list[str]]:
        """
        Construct an instance of this class from CSV line data.

        Args:
            data:  The elements of a line of CSV data.

        Returns:
            * An instance of this class constructed from the CSV data.
            * Any leftover portion of the CSV line that wasn't used.
        """
        relative_path, last_modified = data[0], data[1]
        return cls(Path(relative_path), last_modified), data[2:]

    @classmethod
    def for_file(cls, root_directory: Path, relative_path: Path) -> Self:
        """
        Create an instance of :class:`FileCache` for a given file.

        Retrieve the last modified time of the file, and construct a
        :class:`FileCache` instance with the file's relative path and
        last modified timestamp.

        Args:
            root_directory:  The root directory in which the file lives.
            relative_path:  The path from the ``root_directory`` to the
                file.

        Note:
            Rather than just accepting a file's complete path, we
            distinguish between the root directory and the relative
            path, because it's possible that two distinct files on the
            filesystem could be considered the same (e.g., the same file
            in two separate clones of the same repository).

        Returns:
            The :class:`FileCache` corresponding to the given file.
        """
        full_path = root_directory / relative_path
        modified_time = datetime.fromtimestamp(
            full_path.stat().st_mtime,
            tz=timezone.utc,
        )
        last_modified = modified_time.strftime("%Y-%m-%d %H:%M:%S")
        return cls(relative_path, last_modified)

    def __hash__(self) -> int:
        """
        Get a hash of the :class:`FileCache` instance.

        The hash is computed based on the parent directory of the
        file's path.

        Returns:
            The hash value.
        """
        return hash(self.relative_path.parent)
