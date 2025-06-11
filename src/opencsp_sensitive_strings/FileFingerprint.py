import csv
import dataclasses
import hashlib
import logging
import os

import src.opencsp_sensitive_strings.AbstractFileFingerprint as aff


logger = logging.getLogger(__name__)


@dataclasses.dataclass()
class FileFingerprint(aff.AbstractFileFingerprint):
    # relative_path: str
    # name_ext: str
    size: int
    """ Size of the file, in bytes """
    hash_hex: str
    """ The latest hashlib.sha256([file_contents]).hexdigest() of the file. """

    @staticmethod
    def csv_header(delimiter=",") -> str:
        """Static method. Takes at least one parameter 'delimiter' and returns the string that represents the csv header."""
        keys = list(dataclasses.asdict(FileFingerprint("", "", "", "")).keys())
        return delimiter.join(keys)

    def to_csv_line(self, delimiter=",") -> str:
        """Return a string representation of this instance, to be written to a csv file. Does not include a trailing newline."""
        values = list(dataclasses.asdict(self).values())
        return delimiter.join([str(value) for value in values])

    @classmethod
    def from_csv_line(cls, data: list[str]) -> tuple["FileFingerprint", list[str]]:
        """Construct an instance of this class from the pre-split csv line 'data'. Also return any leftover portion of the csv line that wasn't used."""
        root, name_ext, size, hash_hex = data[0], data[1], data[2], data[3]
        size = int(size)
        return cls(root, name_ext, size, hash_hex), data[4:]

    @classmethod
    def for_file(cls, root_path: str, relative_path: str, file_name_ext: str):
        norm_path = os.path.normpath(os.path.join(root_path, relative_path, file_name_ext))
        file_size = os.path.getsize(norm_path)
        with open(norm_path, "rb") as fin:
            file_hash = hashlib.sha256(fin.read()).hexdigest()
        return cls(relative_path, file_name_ext, file_size, file_hash)

    @classmethod
    def from_csv(cls, file_path: str, file_name_ext: str):
        """Return N instances of this class from a csv file with a header and N lines.

        Basic implementation of from_csv. Subclasses are encouraged to extend this method.
        """
        input_path_file = os.path.join(file_path, file_name_ext)
        data_rows: list[list[str]] = []
        with open(input_path_file) as csv_file:
            reader = csv.reader(csv_file, delimiter=",")
            for row in reader:
                data_rows.append(row)
        return [cls.from_csv_line(row) for row in data_rows[1:]]

    def __lt__(self, other: "FileFingerprint"):
        if not isinstance(other, FileFingerprint):
            message = f"'other' is not of type FileFingerprint but instead of type {type(other)}"
            logger.error(message)
            raise TypeError(message)
        if self.relative_path == other.relative_path:
            return self.name_ext < other.name_ext
        return self.relative_path < other.relative_path
