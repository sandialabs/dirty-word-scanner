import csv
import dataclasses
import os
from datetime import datetime, timezone

import opencsp_sensitive_strings.abstract_file_fingerprint as aff


@dataclasses.dataclass()
class FileCache(aff.AbstractFileFingerprint):
    last_modified: str
    """ The system time that the file was last modified at. """

    @staticmethod
    def csv_header(delimiter: str = ",") -> str:
        """Returns the string that represents the CSV header."""
        keys = list(dataclasses.asdict(FileCache("", "", "")).keys())
        return delimiter.join(keys)

    def to_csv_line(self, delimiter: str = ",") -> str:
        """
        Return a string representation of this instance

        To be written to a CSV file.  Does not include a trailing
        newline.
        """
        values = list(dataclasses.asdict(self).values())
        return delimiter.join([str(value) for value in values])

    @classmethod
    def from_csv_line(cls, data: list[str]) -> tuple["FileCache", list[str]]:
        """
        Construct an instance of this class from CSV line data.

        Also return any leftover portion of the csv line that wasn't
        used.
        """
        root, name_ext, last_modified = data[0], data[1], data[2]
        return cls(root, name_ext, last_modified), data[3:]

    @classmethod
    def for_file(
        cls, root_path: str, relative_path: str, file_name_ext: str
    ) -> "FileCache":
        norm_path = os.path.normpath(
            os.path.join(root_path, relative_path, file_name_ext)
        )
        modified_time = datetime.fromtimestamp(
            os.stat(norm_path).st_mtime,
            tz=timezone.utc,
        )
        last_modified = modified_time.strftime("%Y-%m-%d %H:%M:%S")
        return cls(relative_path, file_name_ext, last_modified)

    @classmethod
    def from_csv(
        cls, file_path: str, file_name_ext: str
    ) -> list[tuple["FileCache", list[str]]]:
        """
        Return N instances of this class from a CSV file.

        One per line in the CSV file, excluding the header.

        Note:
            Subclasses are encouraged to extend this method.
        """
        input_path_file = os.path.join(file_path, file_name_ext)
        with open(input_path_file) as csv_file:
            data_rows = list(csv.reader(csv_file, delimiter=","))
        return [cls.from_csv_line(row) for row in data_rows[1:]]

    def __hash__(self) -> int:
        return hash(self.relative_path)
