import random
from pathlib import Path

import pytest

from opencsp_sensitive_strings.file_fingerprint import FileFingerprint


@pytest.fixture(scope="module")
def output_directory() -> Path:
    result = Path(__file__).parent / "data" / "output" / "FileFingerprint"
    result.mkdir(exist_ok=True, parents=True)
    return result


def _write_text_file(output_file: Path, output_string: str) -> None:
    output_file.parent.mkdir(exist_ok=True, parents=True)
    with output_file.open("w") as output_stream:
        output_stream.write(output_string + "\n")


def test_equal(output_directory: Path) -> None:
    d1 = Path("equal1")
    d2 = Path("equal2")
    f1 = Path("equal_file.txt")
    f2 = Path("equal_file.txt")
    contents = f"{random.Random().random():0.10f}"  # noqa: S311
    _write_text_file(output_directory / d1 / f1, contents)
    _write_text_file(output_directory / d2 / f2, contents)
    ff1 = FileFingerprint.for_file(output_directory / d1, f1)
    ff2 = FileFingerprint.for_file(output_directory / d2, f2)
    assert ff1 == ff2


def test_not_equal_relpath(output_directory: Path) -> None:
    d1 = Path("not_equal_relpath1")
    d2 = Path("not_equal_relpath2")
    f1 = Path("equal_file")
    f2 = Path("equal_file")
    contents = f"{random.Random().random():0.10f}"  # noqa: S311
    _write_text_file(output_directory / d1 / f1, contents)
    _write_text_file(output_directory / d2 / f2, contents)
    ff1 = FileFingerprint.for_file(output_directory, d1 / f1)
    ff2 = FileFingerprint.for_file(output_directory, d2 / f2)
    assert ff1 != ff2


def test_not_equal_filename(output_directory: Path) -> None:
    d1 = Path("not_equal_filename1")
    d2 = Path("not_equal_filename2")
    f1 = Path("equal_file1.txt")
    f2 = Path("equal_file2.txt")
    contents = f"{random.Random().random():0.10f}"  # noqa: S311
    _write_text_file(output_directory / d1 / f1, contents)
    _write_text_file(output_directory / d2 / f2, contents)
    ff1 = FileFingerprint.for_file(output_directory / d1, f1)
    ff2 = FileFingerprint.for_file(output_directory / d2, f2)
    assert ff1 != ff2


def test_not_equal_hash(output_directory: Path) -> None:
    d1 = Path("not_equal_hash1")
    d2 = Path("not_equal_hash2")
    f1 = Path("not_equal1.txt")
    f2 = Path("not_equal2.txt")
    contents = f"{random.Random().random():0.10f}"  # noqa: S311
    contents1 = contents + " "
    contents2 = " " + contents
    _write_text_file(output_directory / d1 / f1, contents1)
    _write_text_file(output_directory / d2 / f2, contents2)
    ff1 = FileFingerprint.for_file(output_directory / d1, f1)
    ff2 = FileFingerprint.for_file(output_directory / d2, f2)
    assert ff1 != ff2
