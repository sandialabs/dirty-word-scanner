import time
from datetime import datetime, timezone
from pathlib import Path

import pytest

from opencsp_sensitive_strings.file_cache import FileCache


@pytest.fixture(scope="module")
def output_directory() -> Path:
    result = Path(__file__).parent / "data" / "output" / "FileCache"
    result.mkdir(exist_ok=True, parents=True)
    return result


def _write_text_file(file_path: Path) -> None:
    with file_path.open("w") as _:
        pass


def _delay_1_second() -> None:
    """
    Sleep up to 1 second.

    So the file modification time looks different.
    """
    ts1 = datetime.now(tz=timezone.utc).strftime("%H%M%S")
    while ts1 == datetime.now(tz=timezone.utc).strftime("%H%M%S"):
        time.sleep(0.05)


def test_file_changed(output_directory: Path) -> None:
    file_path = output_directory / "changing_file.txt"
    _write_text_file(file_path)
    fc1 = FileCache.for_file(Path(), file_path)
    _delay_1_second()
    _write_text_file(file_path)
    fc2 = FileCache.for_file(Path(), file_path)
    assert fc1 != fc2


def test_file_unchanged(output_directory: Path) -> None:
    file_path = output_directory / "static_file.txt"
    _write_text_file(file_path)
    fc1 = FileCache.for_file(Path(), file_path)
    _delay_1_second()
    fc2 = FileCache.for_file(Path(), file_path)
    assert fc1 == fc2
