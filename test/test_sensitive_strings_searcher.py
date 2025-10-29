from collections.abc import Generator
from pathlib import Path
from unittest.mock import patch

import pytest

from opencsp_sensitive_strings.sensitive_strings_searcher import (
    SensitiveStringsSearcher,
)


@pytest.fixture(autouse=True)
def mock_sensitive_strings_searcher() -> Generator[None, None, None]:
    patcher_update_allowed_binaries_csv = patch.object(
        SensitiveStringsSearcher,
        "update_allowed_binaries_csv",
        lambda _: None,
    )
    patcher_copyfile = patch(
        "src.opencsp_sensitive_strings.sensitive_strings_searcher.shutil."
        "copyfile",
        lambda *_args, **_kwargs: None,
    )
    patcher_update_allowed_binaries_csv.start()
    patcher_copyfile.start()
    yield
    patcher_update_allowed_binaries_csv.stop()
    patcher_copyfile.stop()


@pytest.mark.parametrize(
    ("sensitive_strings_basename", "allowed_binaries_basename", "expected"),
    [
        ("no_matches", "all_binaries", 0),
        ("name", "all_binaries", 1),
        ("content", "all_binaries", 1),
        ("directory", "all_binaries", 1),
        ("h5_match", "all_binaries", 1),
        ("test_all_matches", "no_binaries", 6),
        ("no_matches", "single_binary", 1),
        ("no_matches", "single_expected_not_found_binary", 3),
    ],
)
def test_run(
    sensitive_strings_basename: str,
    allowed_binaries_basename: str,
    expected: int,
) -> None:
    data = Path(__file__).parent / "data" / "input" / "sensitive_strings"
    sensitive_strings = data / "per_test_sensitive_strings"
    allowed_binaries = data / "per_test_allowed_binaries"
    searcher = SensitiveStringsSearcher()
    searcher.parse_args(
        [
            "--root-search-dir",
            str(data / "root_search_dir"),
            "--sensitive-strings",
            str(sensitive_strings / f"{sensitive_strings_basename}.csv"),
            "--allowed-binaries",
            str(allowed_binaries / f"{allowed_binaries_basename}.csv"),
        ]
    )
    searcher.git_files_only = False
    assert searcher.run() == expected
