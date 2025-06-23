import csv
import dataclasses
from abc import ABC, abstractmethod
from pathlib import Path


@dataclasses.dataclass()
class CsvInterface(ABC):
    relative_path: Path
    """Path to the file, from the root search directory."""

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, CsvInterface):
            return NotImplemented
        return self.relative_path == other.relative_path

    def __hash__(self) -> int:
        return hash(self.relative_path)

    def csv_header(self) -> str:
        """The object's keys serialized as a line in a CSV file."""
        return ",".join(dataclasses.asdict(self).keys())

    def to_csv_line(self) -> str:
        """The object's values serialized as a line in a CSV file."""
        values = list(dataclasses.asdict(self).values())
        return ",".join([str(_) for _ in values])

    @classmethod
    @abstractmethod
    def from_csv_line(
        cls, data: list[str]
    ) -> tuple["CsvInterface", list[str]]:
        """
        Construct an instance of a subclass from CSV line data.

        Also return any leftover portion of the CSV line that wasn't
        used.
        """

    @classmethod
    def from_csv(
        cls, file_path: Path
    ) -> list[tuple["CsvInterface", list[str]]]:
        """
        Return N instances of this class from a CSV file.

        One per line in the CSV file, excluding the header.

        Note:
            Subclasses are encouraged to extend this method.
        """
        with file_path.open() as csv_file:
            data_rows = list(csv.reader(csv_file, delimiter=","))
        return [cls.from_csv_line(row) for row in data_rows[1:]]


def write_to_csv(
    file_path: Path,
    objects: list["CsvInterface"],
) -> None:
    """
    Create a CSV file with a header and one or more lines.

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
