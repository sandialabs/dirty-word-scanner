import time
import unittest
from datetime import datetime, timezone
from pathlib import Path

from opencsp_sensitive_strings.file_cache import FileCache


class TestFileCache(unittest.TestCase):
    def setUp(self) -> None:
        self.out_dir = Path(__file__).parent / "data" / "output" / "FileCache"
        self.out_dir.mkdir(exist_ok=True)

    def _write_text_file(self, file_name: str) -> Path:
        absolute_output_file = self.out_dir / file_name
        with absolute_output_file.open("w") as _:
            pass
        return absolute_output_file

    def _delay_1_second(self) -> None:
        """
        Sleep up to 1 second.

        So the file modification time looks different.
        """
        ts1 = datetime.now(tz=timezone.utc).strftime("%H%M%S")
        while ts1 == datetime.now(tz=timezone.utc).strftime("%H%M%S"):
            time.sleep(0.05)

    def test_file_changed(self) -> None:
        file_name = "changing_file.txt"
        fc1 = FileCache.for_file(Path(), self._write_text_file(file_name))
        self._delay_1_second()
        fc2 = FileCache.for_file(Path(), self._write_text_file(file_name))
        assert fc1 != fc2

    def test_file_unchanged(self) -> None:
        file_name = "static_file.txt"
        output_file = self._write_text_file(file_name)
        fc1 = FileCache.for_file(Path(), output_file)
        self._delay_1_second()
        fc2 = FileCache.for_file(Path(), output_file)
        assert fc1 == fc2


if __name__ == "__main__":
    unittest.main()
