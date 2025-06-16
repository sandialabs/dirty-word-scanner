import random
import unittest
from pathlib import Path

import opencsp_sensitive_strings.file_fingerprint as ff


class TestFileFingerprint(unittest.TestCase):
    def setUp(self) -> None:
        path = Path(__file__).parent
        self.data_dir = path / "data" / "input" / "FileFingerprint"
        self.out_dir = path / "data" / "output" / "FileFingerprint"
        self.out_dir.mkdir(exist_ok=True)

    def _write_text_file(
        self,
        output_subdirectory: str,
        output_file_basename: str,
        output_string: str,
    ) -> None:
        output_directory = self.out_dir / output_subdirectory
        output_directory.mkdir(exist_ok=True)
        output_dir_body_ext = output_directory / (
            output_file_basename + ".txt"
        )
        with output_dir_body_ext.open("w") as output_stream:
            output_stream.write(output_string + "\n")

    def test_equal(self) -> None:
        d1 = "equal1"
        d2 = "equal2"
        f1 = "equal_file"
        f2 = "equal_file"
        contents = f"{random.Random().random():0.10f}"  # noqa: S311

        self._write_text_file(d1, f1, contents)
        self._write_text_file(d2, f2, contents)
        ff1 = ff.FileFingerprint.for_file(self.out_dir / d1, "", f1 + ".txt")
        ff2 = ff.FileFingerprint.for_file(self.out_dir / d2, "", f2 + ".txt")

        assert ff1 == ff2

    def test_not_equal_relpath(self) -> None:
        d1 = "not_equal_relpath1"
        d2 = "not_equal_relpath2"
        f1 = "equal_file"
        f2 = "equal_file"
        contents = f"{random.Random().random():0.10f}"  # noqa: S311

        self._write_text_file(d1, f1, contents)
        self._write_text_file(d2, f2, contents)
        ff1 = ff.FileFingerprint.for_file(self.out_dir, d1, f1 + ".txt")
        ff2 = ff.FileFingerprint.for_file(self.out_dir, d2, f2 + ".txt")

        assert ff1 != ff2

    def test_not_equal_filename(self) -> None:
        d1 = "not_equal_filename1"
        d2 = "not_equal_filename2"
        f1 = "equal_file1"
        f2 = "equal_file2"
        contents = f"{random.Random().random():0.10f}"  # noqa: S311

        self._write_text_file(d1, f1, contents)
        self._write_text_file(d2, f2, contents)
        ff1 = ff.FileFingerprint.for_file(self.out_dir / d1, "", f1 + ".txt")
        ff2 = ff.FileFingerprint.for_file(self.out_dir / d2, "", f2 + ".txt")

        assert ff1 != ff2

    def test_not_equal_hash(self) -> None:
        d1 = "not_equal_hash1"
        d2 = "not_equal_hash2"
        f1 = "not_equal1"
        f2 = "not_equal2"
        contents = f"{random.Random().random():0.10f}"  # noqa: S311
        contents1 = contents + " "
        contents2 = " " + contents

        self._write_text_file(d1, f1, contents1)
        self._write_text_file(d2, f2, contents2)
        ff1 = ff.FileFingerprint.for_file(self.out_dir / d1, "", f1 + ".txt")
        ff2 = ff.FileFingerprint.for_file(self.out_dir / d2, "", f2 + ".txt")

        assert ff1 != ff2


if __name__ == "__main__":
    unittest.main()
