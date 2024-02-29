import dataclasses

import opencsp.common.lib.file.CsvInterface as ci


@dataclasses.dataclass
class FileFingerprint(ci.CsvInterface):
    relative_path: str
    """ Path to the file, from the root search directory. Usually something like "opencsp/common/lib/tool". """
    name_ext: str
    """ "name.ext" of the file. """
    size: int
    """ Size of the file, in bytes """
    hash_hex: str
    """ The latest hashlib.sha256([file_contents]).hexdigest() of the file. """

    @staticmethod
    def csv_header(delimeter=",") -> str:
        """ Static method. Takes at least one parameter 'delimeter' and returns the string that represents the csv header. """
        keys = list(dataclasses.asdict(FileFingerprint()).keys())
        return delimeter.join(keys)

    def to_csv_line(self, delimeter=",") -> str:
        """ Return a string representation of this instance, to be written to a csv file. Does not include a trailing newline. """
        values = list(dataclasses.asdict(FileFingerprint()).values())
        return delimeter.join([str(value) for value in values])

    @classmethod
    def from_csv_line(cls, data: list[str]) -> tuple['FileFingerprint', list[str]]:
        """ Construct an instance of this class from the pre-split csv line 'data'. Also return any leftover portion of the csv line that wasn't used. """
        root, name_ext, size, hash_hex = data[0], data[1], data[2], data[3]
        size = int(size)
        return cls(root, name_ext, size, hash_hex)
