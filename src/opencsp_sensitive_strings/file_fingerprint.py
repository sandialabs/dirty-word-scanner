"""
Handles file fingerprints with CSV serialization.
"""

import dataclasses
import hashlib
from pathlib import Path

from typing_extensions import Self

from opencsp_sensitive_strings.csv_interface import CsvInterface


@dataclasses.dataclass()
class FileFingerprint(CsvInterface):
    """
    Represents the fingerprint of a file.
    """

    size: int
    """Size of the file, in bytes."""

    hash_hex: str
    """The latest ``hashlib.sha256([contents]).hexdigest()`` of the file."""

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
        relative_path, size, hash_hex = data[0], data[1], data[2]
        return cls(Path(relative_path), int(size), hash_hex), data[3:]

    @classmethod
    def for_file(cls, root_directory: Path, relative_path: Path) -> Self:
        """
        Create an instance of :class:`FileFingerprint` for a given file.

        Retrieve the size and SHA-256 hash of the file located at the
        specified path, and construct a :class:`FileFingerprint`
        instance with the file's relative path, size, and hash.

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
            The :class:`FileFingerprint` corresponding to the given
            file.
        """
        full_path = root_directory / relative_path
        file_size = full_path.stat().st_size
        with full_path.open("rb") as input_file:
            file_hash = hashlib.sha256(input_file.read()).hexdigest()
        return cls(relative_path, file_size, file_hash)

    def __lt__(self, other: Self) -> bool:
        """
        Compare two :class:`FileFingerprint` instances.

        Args:
            other:  The other :class:`FileFingerprint` to compare
                against.

        Returns:
            ``True`` if the relative path of the current instance is
            less than that of the other instance.
        """
        if not isinstance(other, FileFingerprint):
            return NotImplemented
        return self.relative_path < other.relative_path
