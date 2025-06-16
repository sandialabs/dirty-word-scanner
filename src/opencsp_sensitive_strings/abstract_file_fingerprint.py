import dataclasses
import os
from abc import ABC


@dataclasses.dataclass()
class AbstractFileFingerprint(ABC):
    relative_path: str
    """Path to the file, from the root search directory."""
    name_ext: str
    """'name.ext' of the file."""

    @property
    def relpath_name_ext(self) -> str:
        return os.path.join(self.relative_path, self.name_ext)

    def eq_aff(self, other: "AbstractFileFingerprint") -> bool:
        if not isinstance(other, AbstractFileFingerprint):
            return False
        return (
            self.relative_path == other.relative_path
            and self.name_ext == other.name_ext
        )

    def csv_header(self) -> str:
        return ",".join(dataclasses.asdict(self).keys())

    def to_csv(
        self,
        file_path: str,
        file_name: str,
        rows: list["AbstractFileFingerprint"],
    ) -> None:
        """
        Create a CSV file with a header and one or more lines.
        """
        row_strs = [_.to_csv_line() for _ in rows]
        output_body_ext = file_name + ".csv"
        output_dir_body_ext = os.path.normpath(
            os.path.join(file_path, output_body_ext)
        )
        os.makedirs(file_path, exist_ok=True)
        output_stream = open(output_dir_body_ext, "w")
        output_stream.write(self.csv_header() + "\n")
        for data_line in row_strs:
            output_stream.write(data_line + "\n")
        output_stream.close()
