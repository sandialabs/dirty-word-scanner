import csv
import dataclasses
from datetime import datetime, timezone
from pathlib import Path

from opencsp_sensitive_strings.csv_interface import CsvInterface


@dataclasses.dataclass()
class FileCache(CsvInterface):
    last_modified: str
    """ The system time that the file was last modified at. """

    @classmethod
    def from_csv_line(cls, data: list[str]) -> tuple["FileCache", list[str]]:
        """
        Construct an instance of this class from CSV line data.

        Also return any leftover portion of the csv line that wasn't
        used.
        """
        root, name_ext, last_modified = data[0], data[1], data[2]
        return cls(Path(root), Path(name_ext), last_modified), data[3:]

    @classmethod
    def for_file(
        cls, root_path: Path, relative_path: Path, file_name_ext: Path
    ) -> "FileCache":
        norm_path = root_path / relative_path / file_name_ext
        modified_time = datetime.fromtimestamp(
            norm_path.stat().st_mtime,
            tz=timezone.utc,
        )
        last_modified = modified_time.strftime("%Y-%m-%d %H:%M:%S")
        return cls(relative_path, file_name_ext, last_modified)

    @classmethod
    def from_csv(cls, file_path: Path) -> list[tuple["FileCache", list[str]]]:
        """
        Return N instances of this class from a CSV file.

        One per line in the CSV file, excluding the header.

        Note:
            Subclasses are encouraged to extend this method.
        """
        with file_path.open() as csv_file:
            data_rows = list(csv.reader(csv_file, delimiter=","))
        return [cls.from_csv_line(row) for row in data_rows[1:]]

    def __hash__(self) -> int:
        return hash(self.relative_path)
