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
        return ",".join(dataclasses.asdict(self).keys())

    def to_csv_line(self) -> str:
        """
        Return a string representation of this instance

        To be written to a CSV file.  Does not include a trailing
        newline.
        """
        values = list(dataclasses.asdict(self).values())
        return ",".join([str(_) for _ in values])

    def to_csv(
        self,
        file_path: Path,
        rows: list["CsvInterface"],
    ) -> None:
        """
        Create a CSV file with a header and one or more lines.
        """
        row_strs = [_.to_csv_line() for _ in rows]
        file_path.parent.mkdir(exist_ok=True, parents=True)
        with file_path.with_suffix(".csv").open("w") as output_stream:
            output_stream.write(self.csv_header() + "\n")
            for data_line in row_strs:
                output_stream.write(data_line + "\n")

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
