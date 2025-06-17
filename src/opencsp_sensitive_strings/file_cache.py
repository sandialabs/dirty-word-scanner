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

        Also return any leftover portion of the CSV line that wasn't
        used.
        """
        relative_path, last_modified = data[0], data[1]
        return cls(Path(relative_path), last_modified), data[2:]

    @classmethod
    def for_file(cls, root_path: Path, relative_path: Path) -> "FileCache":
        norm_path = root_path / relative_path
        modified_time = datetime.fromtimestamp(
            norm_path.stat().st_mtime,
            tz=timezone.utc,
        )
        last_modified = modified_time.strftime("%Y-%m-%d %H:%M:%S")
        return cls(relative_path, last_modified)

    def __hash__(self) -> int:
        return hash(self.relative_path.parent)
