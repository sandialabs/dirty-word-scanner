import random
import unittest
from pathlib import Path

from opencsp_sensitive_strings.file_fingerprint import FileFingerprint


class TestFileFingerprint(unittest.TestCase):
    def setUp(self) -> None:
        path = Path(__file__).parent
        self.data_dir = path / "data" / "input" / "FileFingerprint"
        self.out_dir = path / "data" / "output" / "FileFingerprint"
        self.out_dir.mkdir(exist_ok=True)

    def _write_text_file(
        self,
        output_subdirectory: Path,
        output_file: Path,
        output_string: str,
    ) -> None:
        output_directory = self.out_dir / output_subdirectory
        output_directory.mkdir(exist_ok=True)
        output_dir_body_ext = output_directory / output_file
        with output_dir_body_ext.open("w") as output_stream:
            output_stream.write(output_string + "\n")

    def test_equal(self) -> None:
        d1 = Path("equal1")
        d2 = Path("equal2")
        f1 = Path("equal_file.txt")
        f2 = Path("equal_file.txt")
        contents = f"{random.Random().random():0.10f}"  # noqa: S311

        self._write_text_file(d1, f1, contents)
        self._write_text_file(d2, f2, contents)
        ff1 = FileFingerprint.for_file(self.out_dir / d1, f1)
        ff2 = FileFingerprint.for_file(self.out_dir / d2, f2)

        assert ff1 == ff2

    def test_not_equal_relpath(self) -> None:
        d1 = Path("not_equal_relpath1")
        d2 = Path("not_equal_relpath2")
        f1 = Path("equal_file")
        f2 = Path("equal_file")
        contents = f"{random.Random().random():0.10f}"  # noqa: S311

        self._write_text_file(d1, f1, contents)
        self._write_text_file(d2, f2, contents)
        ff1 = FileFingerprint.for_file(self.out_dir, d1 / f1)
        ff2 = FileFingerprint.for_file(self.out_dir, d2 / f2)

        assert ff1 != ff2

    def test_not_equal_filename(self) -> None:
        d1 = Path("not_equal_filename1")
        d2 = Path("not_equal_filename2")
        f1 = Path("equal_file1.txt")
        f2 = Path("equal_file2.txt")
        contents = f"{random.Random().random():0.10f}"  # noqa: S311

        self._write_text_file(d1, f1, contents)
        self._write_text_file(d2, f2, contents)
        ff1 = FileFingerprint.for_file(self.out_dir / d1, f1)
        ff2 = FileFingerprint.for_file(self.out_dir / d2, f2)

        assert ff1 != ff2

    def test_not_equal_hash(self) -> None:
        d1 = Path("not_equal_hash1")
        d2 = Path("not_equal_hash2")
        f1 = Path("not_equal1.txt")
        f2 = Path("not_equal2.txt")
        contents = f"{random.Random().random():0.10f}"  # noqa: S311
        contents1 = contents + " "
        contents2 = " " + contents

        self._write_text_file(d1, f1, contents1)
        self._write_text_file(d2, f2, contents2)
        ff1 = FileFingerprint.for_file(self.out_dir / d1, f1)
        ff2 = FileFingerprint.for_file(self.out_dir / d2, f2)

        assert ff1 != ff2


if __name__ == "__main__":
    unittest.main()
