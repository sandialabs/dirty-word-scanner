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
        relative_path, size, hash_hex = data[0], data[1], data[2]
        return cls(Path(relative_path), int(size), hash_hex), data[3:]

    @classmethod
    def for_file(
        cls, root_path: Path, relative_path: Path
    ) -> "FileFingerprint":
        norm_path = root_path / relative_path
        file_size = norm_path.stat().st_size
        with norm_path.open("rb") as fin:
            file_hash = hashlib.sha256(fin.read()).hexdigest()
        return cls(relative_path, file_size, file_hash)

    def __lt__(self, other: "FileFingerprint") -> bool:
        if not isinstance(other, FileFingerprint):
            message = (
                "'other' is not of type FileFingerprint but instead of type "
                + type(other)
            )
            logger.error(message)
            raise TypeError(message)
        return self.relative_path < other.relative_path
