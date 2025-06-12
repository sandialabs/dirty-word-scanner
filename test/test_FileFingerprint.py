import os
import random
import unittest

import src.opencsp_sensitive_strings.FileFingerprint as ff


class test_FileFingerprint(unittest.TestCase):
    def setUp(self) -> None:
        path = os.path.dirname(__file__)
        self.data_dir = os.path.join(path, "data", "input", "FileFingerprint")
        self.out_dir = os.path.join(path, "data", "output", "FileFingerprint")
        os.makedirs(self.out_dir, exist_ok=True)

    def _write_text_file(
        self,
        output_subdirectory: str,
        output_file_basename: str,
        output_string: str,
    ) -> None:
        output_directory = os.path.join(self.out_dir, output_subdirectory)
        os.makedirs(output_directory, exist_ok=True)
        output_dir_body_ext = os.path.join(
            output_directory,
            output_file_basename + ".txt",
        )
        with open(output_dir_body_ext, "w") as output_stream:
            output_stream.write(output_string + "\n")

    def test_equal(self) -> None:
        d1 = "equal1"
        d2 = "equal2"
        f1 = "equal_file"
        f2 = "equal_file"
        contents = "%0.10f" % random.Random().random()  # noqa: S311

        self._write_text_file(d1, f1, contents)
        self._write_text_file(d2, f2, contents)
        ff1 = ff.FileFingerprint.for_file(
            f"{self.out_dir}/{d1}", "", f1 + ".txt"
        )
        ff2 = ff.FileFingerprint.for_file(
            f"{self.out_dir}/{d2}", "", f2 + ".txt"
        )

        assert ff1 == ff2

    def test_not_equal_relpath(self) -> None:
        d1 = "not_equal_relpath1"
        d2 = "not_equal_relpath2"
        f1 = "equal_file"
        f2 = "equal_file"
        contents = "%0.10f" % random.Random().random()  # noqa: S311

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
        contents = "%0.10f" % random.Random().random()  # noqa: S311

        self._write_text_file(d1, f1, contents)
        self._write_text_file(d2, f2, contents)
        ff1 = ff.FileFingerprint.for_file(
            f"{self.out_dir}/{d1}", "", f1 + ".txt"
        )
        ff2 = ff.FileFingerprint.for_file(
            f"{self.out_dir}/{d2}", "", f2 + ".txt"
        )

        assert ff1 != ff2

    def test_not_equal_hash(self) -> None:
        d1 = "not_equal_hash1"
        d2 = "not_equal_hash2"
        f1 = "not_equal1"
        f2 = "not_equal2"
        contents = "%0.10f" % random.Random().random()  # noqa: S311
        contents1 = contents + " "
        contents2 = " " + contents

        self._write_text_file(d1, f1, contents1)
        self._write_text_file(d2, f2, contents2)
        ff1 = ff.FileFingerprint.for_file(
            f"{self.out_dir}/{d1}", "", f1 + ".txt"
        )
        ff2 = ff.FileFingerprint.for_file(
            f"{self.out_dir}/{d2}", "", f2 + ".txt"
        )

        assert ff1 != ff2


if __name__ == "__main__":
    unittest.main()
