import os
import unittest
from unittest.mock import patch

import numpy as np

import src.opencsp_sensitive_strings.sensitive_strings as ss


class test_sensitive_strings(unittest.TestCase):
    def setUp(self) -> None:
        path = os.path.dirname(__file__)
        self.data_dir = os.path.join(
            path, "data", "input", "sensitive_strings"
        )
        self.out_dir = os.path.join(
            path, "data", "output", "sensitive_strings"
        )
        os.makedirs(self.out_dir, exist_ok=True)

        self.root_search_dir = os.path.join(self.data_dir, "root_search_dir")
        self.ss_dir = os.path.join(self.data_dir, "per_test_sensitive_strings")
        self.allowed_binaries_dir = os.path.join(
            self.data_dir, "per_test_allowed_binaries"
        )
        self.all_binaries = os.path.join(
            self.allowed_binaries_dir, "all_binaries.csv"
        )
        self.no_binaries = os.path.join(
            self.allowed_binaries_dir, "no_binaries.csv"
        )

        self.patcher_update = patch.object(
            ss.SensitiveStringsSearcher,
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

        im8i = ss.numpy_to_image(arr8i, rescale_or_clip="truncate")
        im16i = ss.numpy_to_image(arr16i, rescale_or_clip="truncate")
        im8f = ss.numpy_to_image(arr8f, rescale_or_clip="truncate")
        im16f = ss.numpy_to_image(arr16f, rescale_or_clip="truncate")

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

        im8i = ss.numpy_to_image(arr8i, rescale_or_clip="rescale")
        im16i = ss.numpy_to_image(arr16i, rescale_or_clip="rescale")
        im8f = ss.numpy_to_image(arr8f, rescale_or_clip="rescale")
        im16f = ss.numpy_to_image(arr16f, rescale_or_clip="rescale")

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
        sensitive_strings_csv = os.path.join(self.ss_dir, "no_matches.csv")
        searcher = ss.SensitiveStringsSearcher(
            self.root_search_dir, sensitive_strings_csv, self.all_binaries
        )
        searcher.git_files_only = False
        assert searcher.search_files() == 0

    def test_single_matcher(self) -> None:
        # based on file name
        sensitive_strings_csv = os.path.join(
            self.ss_dir, "test_single_matcher.csv"
        )
        searcher = ss.SensitiveStringsSearcher(
            self.root_search_dir, sensitive_strings_csv, self.all_binaries
        )
        searcher.git_files_only = False
        assert searcher.search_files() == 1

        # based on file content
        sensitive_strings_csv = os.path.join(
            self.ss_dir, "test_single_matcher_content.csv"
        )
        searcher = ss.SensitiveStringsSearcher(
            self.root_search_dir, sensitive_strings_csv, self.all_binaries
        )
        searcher.git_files_only = False
        assert searcher.search_files() == 1

    def test_directory_matcher(self) -> None:
        sensitive_strings_csv = os.path.join(
            self.ss_dir, "test_directory_matcher.csv"
        )
        searcher = ss.SensitiveStringsSearcher(
            self.root_search_dir, sensitive_strings_csv, self.all_binaries
        )
        searcher.git_files_only = False
        assert searcher.search_files() == 1

    def test_all_matches(self) -> None:
        sensitive_strings_csv = os.path.join(
            self.ss_dir, "test_all_matches.csv"
        )
        searcher = ss.SensitiveStringsSearcher(
            self.root_search_dir, sensitive_strings_csv, self.no_binaries
        )
        searcher.git_files_only = False
        # There should be 6 matches:
        # * files:  a.txt, b/b.txt, and c/d/e.txt
        # * images:  c/img1.png, and c/img2.jpg
        # * hdf5:  f.h5/f
        assert searcher.search_files() == 6

    def test_single_unknown_binary(self) -> None:
        sensitive_strings_csv = os.path.join(self.ss_dir, "no_matches.csv")
        single_binary_csv = os.path.join(
            self.allowed_binaries_dir, "single_binary.csv"
        )
        searcher = ss.SensitiveStringsSearcher(
            self.root_search_dir, sensitive_strings_csv, single_binary_csv
        )
        searcher.git_files_only = False
        assert searcher.search_files() == 1

    def test_single_expected_not_found_binary(self) -> None:
        sensitive_strings_csv = os.path.join(self.ss_dir, "no_matches.csv")
        single_binary_csv = os.path.join(
            self.allowed_binaries_dir, "single_expected_not_found_binary.csv"
        )
        searcher = ss.SensitiveStringsSearcher(
            self.root_search_dir, sensitive_strings_csv, single_binary_csv
        )
        searcher.git_files_only = False
        # 2 unknown binaries, and 1 expected not found
        assert searcher.search_files() == 3

    def test_hdf5_match(self) -> None:
        sensitive_strings_csv = os.path.join(self.ss_dir, "h5_match.csv")
        searcher = ss.SensitiveStringsSearcher(
            self.root_search_dir, sensitive_strings_csv, self.all_binaries
        )
        searcher.git_files_only = False
        # 2 unknown binaries, and 1 expected not found
        assert searcher.search_files() == 1


if __name__ == "__main__":
    unittest.main()
