"""
Handles CSV file operations with a base interface and utility function.
"""

import csv
import dataclasses
from abc import ABC, abstractmethod
from collections.abc import Sequence
from pathlib import Path

from typing_extensions import Self


@dataclasses.dataclass()
class CsvInterface(ABC):
    """
    Abstract base class for CSV serializable objects.
    """

    relative_path: Path
    """Path to the file, from the root search directory."""

    def csv_header(self) -> str:
        """
        Generate the CSV header from the object's keys.

        Returns:
            The object's keys serialized as a line in a CSV file.
        """
        return ",".join(dataclasses.asdict(self).keys())

    def to_csv_line(self) -> str:
        """
        Convert the object's values to a CSV line.

        Returns:
            The object's values serialized as a line in a CSV file.
        """
        values = list(dataclasses.asdict(self).values())
        return ",".join([str(_) for _ in values])

    @classmethod
    @abstractmethod
    def from_csv_line(cls, data: list[str]) -> tuple[Self, list[str]]:
        """
        Construct an instance of a subclass from CSV line data.

        Args:
            data:  The elements of a line of CSV data.

        Returns:
            * An instance of the subclass constructed from the CSV data.
            * Any leftover portion of the CSV line that wasn't used.
        """

    @classmethod
    @abstractmethod
    def for_file(cls, root_directory: Path, relative_path: Path) -> Self:
        """
        Create an instance of this class for a given file.

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
            The object corresponding to the given file.
        """

    @classmethod
    def from_csv(cls, file_path: Path) -> list[tuple[Self, list[str]]]:
        """
        Read instances of the class from a CSV file.

        Args:
            file_path:  The path to the CSV file to read from.

        Returns:
            A list of tuples containing instances of the class
            constructed from the CSV data, and any leftover portions of
            the CSV lines.
        """
        with file_path.open() as csv_file:
            data_rows = list(csv.reader(csv_file))
        return [cls.from_csv_line(row) for row in data_rows[1:]]


def write_to_csv(
    file_path: Path,
    objects: Sequence["CsvInterface"],
) -> None:
    """
    Write a list of :class:`CsvInterface` objects to a CSV file.

    Args:
        file_path:  The CSV file to which to write.
        objects:  A list of :class:`CsvInterface` objects (or, more
            likely, objects of child classes) to serialize as rows in
            the CSV file.

    Raises:
        TypeError:  If the objects aren't all of the same (sub)type.
    """
    if not objects:
        return
    first = objects[0]
    if not all(type(_) is type(first) for _ in objects):
        message = "Objects must all be of the same type."
        raise TypeError(message)
    rows = [_.to_csv_line() for _ in objects]
    file_path.parent.mkdir(exist_ok=True, parents=True)
    with file_path.with_suffix(".csv").open("w") as output_stream:
        output_stream.write(first.csv_header() + "\n")
        for data_line in rows:
            output_stream.write(data_line + "\n")
