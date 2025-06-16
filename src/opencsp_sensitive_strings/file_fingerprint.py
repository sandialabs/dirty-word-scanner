import csv
import dataclasses
import hashlib
import logging
from pathlib import Path

from opencsp_sensitive_strings.csv_interface import CsvInterface

logger = logging.getLogger(__name__)


@dataclasses.dataclass()
class FileFingerprint(CsvInterface):
    size: int
    """ Size of the file, in bytes """
    hash_hex: str
    """ The latest hashlib.sha256([file_contents]).hexdigest() of the file. """

    @classmethod
    def from_csv_line(
        cls, data: list[str]
    ) -> tuple["FileFingerprint", list[str]]:
        """
        Construct an instance of this class from CSV line data.

        Also return any leftover portion of the CSV line that wasn't
        used.
        """
        root, name_ext, size, hash_hex = data[0], data[1], data[2], data[3]
        return cls(Path(root), Path(name_ext), int(size), hash_hex), data[4:]

    @classmethod
    def for_file(
        cls, root_path: Path, relative_path: Path, file_name_ext: Path
    ) -> "FileFingerprint":
        norm_path = root_path / relative_path / file_name_ext
        file_size = norm_path.stat().st_size
        with norm_path.open("rb") as fin:
            file_hash = hashlib.sha256(fin.read()).hexdigest()
        return cls(relative_path, file_name_ext, file_size, file_hash)

    @classmethod
    def from_csv(
        cls, file_path: Path, file_name_ext: Path
    ) -> list[tuple["FileFingerprint", list[str]]]:
        """
        Return N instances of this class from a CSV file.

        One per line in the CSV file, excluding the header.

        Note:
            Subclasses are encouraged to extend this method.
        """
        input_path_file = file_path / file_name_ext
        with input_path_file.open() as csv_file:
            data_rows = list(csv.reader(csv_file, delimiter=","))
        return [cls.from_csv_line(row) for row in data_rows[1:]]

    def __lt__(self, other: "FileFingerprint") -> bool:
        if not isinstance(other, FileFingerprint):
            message = (
                "'other' is not of type FileFingerprint but instead of type "
                + type(other)
            )
            logger.error(message)
            raise TypeError(message)
        if self.relative_path == other.relative_path:
            return self.name_ext < other.name_ext
        return self.relative_path < other.relative_path
