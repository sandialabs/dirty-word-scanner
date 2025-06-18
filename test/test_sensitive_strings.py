import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

from src.opencsp_sensitive_strings.sensitive_strings import (
    SensitiveStringsSearcher,
    numpy_to_image,
)


class TestSensitiveStrings(unittest.TestCase):
    def setUp(self) -> None:
        path = Path(__file__).parent
        self.data_dir = path / "data" / "input" / "sensitive_strings"
        self.out_dir = path / "data" / "output" / "sensitive_strings"
        self.out_dir.mkdir(exist_ok=True)

        self.root_search_dir = self.data_dir / "root_search_dir"
        self.ss_dir = self.data_dir / "per_test_sensitive_strings"
        self.allowed_binaries_dir = self.data_dir / "per_test_allowed_binaries"
        self.all_binaries = self.allowed_binaries_dir / "all_binaries.csv"
        self.no_binaries = self.allowed_binaries_dir / "no_binaries.csv"

        self.patcher_update = patch.object(
            SensitiveStringsSearcher,
            "update_allowed_binaries_csv",
            lambda _: None,
        )
        self.patcher_copy = patch(
            "src.opencsp_sensitive_strings.sensitive_strings.shutil.copyfile",
            lambda *_args, **_kwargs: None,
        )
        self.patcher_update.start()
        self.patcher_copy.start()

    def tearDown(self) -> None:
        self.patcher_update.stop()
        self.patcher_copy.stop()

    def test_numpy_to_image_truncate(self) -> None:
        arr8i = np.array([[0, 125, 255]]).astype(np.int8)
        arr16i = np.array([[0, 8192, 16384]]).astype(np.int16)
        arr8f = arr8i.astype(np.float16)
        arr16f = arr16i.astype(np.float16)

        im8i = numpy_to_image(arr8i, rescale_or_clip="truncate")
        im16i = numpy_to_image(arr16i, rescale_or_clip="truncate")
        im8f = numpy_to_image(arr8f, rescale_or_clip="truncate")
        im16f = numpy_to_image(arr16f, rescale_or_clip="truncate")

        np.testing.assert_array_equal(
            np.asarray(im8i), np.array([[0, 125, 255]])
        )
        np.testing.assert_array_equal(
            np.asarray(im16i), np.array([[0, 255, 255]])
        )
        np.testing.assert_array_equal(
            np.asarray(im8f), np.array([[0, 125, 255]])
        )
        np.testing.assert_array_equal(
            np.asarray(im16f), np.array([[0, 255, 255]])
        )

    def test_numpy_to_image_rescale(self) -> None:
        arr8i = np.array([[0, 125, 255]]).astype(np.int8)
        arr16i = np.array([[0, 8192, 16384]]).astype(np.int16)
        arr8f = arr8i.astype(np.float16)
        arr16f = arr16i.astype(np.float16)

        im8i = numpy_to_image(arr8i, rescale_or_clip="rescale")
        im16i = numpy_to_image(arr16i, rescale_or_clip="rescale")
        im8f = numpy_to_image(arr8f, rescale_or_clip="rescale")
        im16f = numpy_to_image(arr16f, rescale_or_clip="rescale")

        np.testing.assert_array_equal(
            np.asarray(im8i), np.array([[0, 125, 255]])
        )
        np.testing.assert_array_equal(
            np.asarray(im16i), np.array([[0, 127, 255]])
        )
        np.testing.assert_array_equal(
            np.asarray(im8f), np.array([[0, 125, 255]])
        )
        np.testing.assert_array_equal(
            np.asarray(im16f), np.array([[0, 127, 255]])
        )

    def test_no_matches(self) -> None:
        sensitive_strings_csv = self.ss_dir / "no_matches.csv"
        searcher = SensitiveStringsSearcher(
            self.root_search_dir, sensitive_strings_csv, self.all_binaries
        )
        searcher.git_files_only = False
        assert searcher.search_files() == 0

    def test_single_matcher(self) -> None:
        # based on file name
        sensitive_strings_csv = self.ss_dir / "test_single_matcher.csv"
        searcher = SensitiveStringsSearcher(
            self.root_search_dir, sensitive_strings_csv, self.all_binaries
        )
        searcher.git_files_only = False
        assert searcher.search_files() == 1

        # based on file content
        sensitive_strings_csv = self.ss_dir / "test_single_matcher_content.csv"
        searcher = SensitiveStringsSearcher(
            self.root_search_dir, sensitive_strings_csv, self.all_binaries
        )
        searcher.git_files_only = False
        assert searcher.search_files() == 1

    def test_directory_matcher(self) -> None:
        sensitive_strings_csv = self.ss_dir / "test_directory_matcher.csv"
        searcher = SensitiveStringsSearcher(
            self.root_search_dir, sensitive_strings_csv, self.all_binaries
        )
        searcher.git_files_only = False
        assert searcher.search_files() == 1

    def test_all_matches(self) -> None:
        sensitive_strings_csv = self.ss_dir / "test_all_matches.csv"
        searcher = SensitiveStringsSearcher(
            self.root_search_dir, sensitive_strings_csv, self.no_binaries
        )
        searcher.git_files_only = False
        # There should be 6 matches:
        # * files:  a.txt, b/b.txt, and c/d/e.txt
        # * images:  c/img1.png, and c/img2.jpg
        # * hdf5:  f.h5/f
        assert searcher.search_files() == 6

    def test_single_unknown_binary(self) -> None:
        sensitive_strings_csv = self.ss_dir / "no_matches.csv"
        single_binary_csv = self.allowed_binaries_dir / "single_binary.csv"
        searcher = SensitiveStringsSearcher(
            self.root_search_dir, sensitive_strings_csv, single_binary_csv
        )
        searcher.git_files_only = False
        assert searcher.search_files() == 1

    def test_single_expected_not_found_binary(self) -> None:
        sensitive_strings_csv = self.ss_dir / "no_matches.csv"
        single_binary_csv = (
            self.allowed_binaries_dir / "single_expected_not_found_binary.csv"
        )
        searcher = SensitiveStringsSearcher(
            self.root_search_dir, sensitive_strings_csv, single_binary_csv
        )
        searcher.git_files_only = False
        # 2 unknown binaries, and 1 expected not found
        assert searcher.search_files() == 3

    def test_hdf5_match(self) -> None:
        sensitive_strings_csv = self.ss_dir / "h5_match.csv"
        searcher = SensitiveStringsSearcher(
            self.root_search_dir, sensitive_strings_csv, self.all_binaries
        )
        searcher.git_files_only = False
        # 2 unknown binaries, and 1 expected not found
        assert searcher.search_files() == 1


if __name__ == "__main__":
    unittest.main()
