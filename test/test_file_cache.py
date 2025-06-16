import time
import unittest
from datetime import datetime, timezone
from pathlib import Path

import opencsp_sensitive_strings.file_cache as fc


class TestFileCache(unittest.TestCase):
    def setUp(self) -> None:
        path = Path(__file__).parent
        self.data_dir = path / "data" / "input" / "FileCache"
        self.out_dir = path / "data" / "output" / "FileCache"
        self.out_dir.mkdir(exist_ok=True)

    def _write_text_file(self, output_file_basename: str) -> None:
        output_dir_body_ext = self.out_dir / (output_file_basename + ".txt")
        with output_dir_body_ext.open("w") as _:
            pass

    def _delay_1_second(self) -> None:
        """
        Sleep up to 1 second.

        So the file modification time looks different.
        """
        ts1 = datetime.now(tz=timezone.utc).strftime("%H%M%S")
        while ts1 == datetime.now(tz=timezone.utc).strftime("%H%M%S"):
            time.sleep(0.05)

    def test_file_changed(self) -> None:
        outfile = "changing_file.txt"

        self._write_text_file(outfile)
        fc1 = fc.FileCache.for_file(Path(), self.out_dir, outfile + ".txt")
        self._delay_1_second()
        self._write_text_file(outfile)
        fc2 = fc.FileCache.for_file(Path(), self.out_dir, outfile + ".txt")

        assert fc1 != fc2

    def test_file_unchanged(self) -> None:
        outfile = "static_file.txt"

        self._write_text_file(outfile)
        fc1 = fc.FileCache.for_file(Path(), self.out_dir, outfile + ".txt")
        self._delay_1_second()
        fc2 = fc.FileCache.for_file(Path(), self.out_dir, outfile + ".txt")

        assert fc1 == fc2


if __name__ == "__main__":
    unittest.main()
