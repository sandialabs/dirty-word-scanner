import os
import unittest

import src.opencsp_sensitive_strings.SensitiveStringMatcher as ssm


class test_SensitiveStringMatcher(unittest.TestCase):
    def setUp(self) -> None:
        path = os.path.dirname(__file__)
        self.data_dir = os.path.join(path, "data", "input", "FileCache")
        self.out_dir = os.path.join(path, "data", "output", "FileCache")
        os.makedirs(self.out_dir, exist_ok=True)

    def test_match(self) -> None:
        matcher = ssm.SensitiveStringMatcher("Basic Matcher", "bar")
        matches = matcher.check_lines(["foo", "bar", "baz"])
        assert len(matches) == 1
        assert matches[0].lineno == 2

    def test_partial_match(self) -> None:
        matcher = ssm.SensitiveStringMatcher("Basic Matcher", "bar")
        matches = matcher.check_lines(["foobarbaz"])
        assert len(matches) == 1
        assert matches[0].lineno == 1
        assert matches[0].colno == 3

        matcher = ssm.SensitiveStringMatcher("Basic Matcher", "foo")
        matches = matcher.check_lines(["foobarbaz"])
        assert len(matches) == 1
        assert matches[0].lineno == 1
        assert matches[0].colno == 0

        matcher = ssm.SensitiveStringMatcher("Basic Matcher", "baz")
        matches = matcher.check_lines(["foobarbaz"])
        assert len(matches) == 1
        assert matches[0].lineno == 1
        assert matches[0].colno == 6

    def test_matches(self) -> None:
        matcher = ssm.SensitiveStringMatcher(
            "Basic Matcher", "foo", "bar", "baz"
        )
        matches = matcher.check_lines(["foo", "bar", "baz"])
        assert len(matches) == 3
        assert matches[0].lineno == 1
        assert matches[1].lineno == 2
        assert matches[2].lineno == 3

    def test_dont_match(self) -> None:
        matcher = ssm.SensitiveStringMatcher(
            "Basic Matcher", "foo", "**dont_match", "foo"
        )
        matches = matcher.check_lines(["foo", "bar", "baz"])
        assert len(matches) == 0

    def test_case_sensitive(self) -> None:
        matcher = ssm.SensitiveStringMatcher(
            "Basic Matcher", "**case_sensitive", "foo"
        )
        matches = matcher.check_lines(
            ["foO", "fOo", "fOO", "Foo", "FoO", "FOo", "FOO", "foo"]
        )
        assert len(matches) == 1
        assert matches[0].lineno == 8

    def test_single_regex(self) -> None:
        matcher = ssm.SensitiveStringMatcher(
            "Basic Matcher", "**next_is_regex", r"[a-z]a[a-z]"
        )
        matches = matcher.check_lines(["foo", "bar", "baz"])
        assert len(matches) == 2
        assert matches[0].lineno == 2
        assert matches[0].line_part == "bar"
        assert matches[1].lineno == 3
        assert matches[1].line_part == "baz"

    def test_partial_single_regex(self) -> None:
        matcher = ssm.SensitiveStringMatcher(
            "Regex Matcher", "**next_is_regex", r"[a-z]o[a-z]"
        )
        matches = matcher.check_lines(["foobarbaz"])
        assert len(matches) == 1
        assert matches[0].colno == 0
        assert matches[0].line_part == "foo"

        matcher = ssm.SensitiveStringMatcher(
            "Regex Matcher", "**next_is_regex", r"[a-z]{2}r"
        )
        matches = matcher.check_lines(["foobarbaz"])
        assert len(matches) == 1
        assert matches[0].colno == 3
        assert matches[0].line_part == "bar"

        matcher = ssm.SensitiveStringMatcher(
            "Regex Matcher", "**next_is_regex", r"[a-z]{2}z"
        )
        matches = matcher.check_lines(["foobarbaz"])
        assert len(matches) == 1
        assert matches[0].colno == 6
        assert matches[0].line_part == "baz"

    def test_partial_multiple_regex(self) -> None:
        matcher = ssm.SensitiveStringMatcher(
            "Regex Matcher",
            "**all_regex",
            r"[a-z]o[a-z]",
            r"[a-z]{2}r",
            r"[a-z]{2}z",
        )
        matches = matcher.check_lines(["foobarbaz"])
        assert len(matches) == 3
        assert matches[0].colno == 0
        assert matches[0].line_part == "foo"
        assert matches[1].colno == 3
        assert matches[1].line_part == "bar"
        assert matches[2].colno == 6
        assert matches[2].line_part == "baz"

    def test_mixed_plain_regex(self) -> None:
        matcher = ssm.SensitiveStringMatcher(
            "Basic Matcher", "foo", "**next_is_regex", r"[a-z]{2}r", "baz"
        )

        matches = matcher.check_lines(["foobarbaz"])
        assert len(matches) >= 1
        assert matches[0].colno == 0
        assert matches[0].line_part == "foo"

        matches = matcher.check_lines(["goobarbaz"])
        assert len(matches) >= 1
        assert matches[0].colno == 3
        assert matches[0].line_part == "bar"

        matches = matcher.check_lines(["googgrbaz"])
        assert len(matches) >= 1
        assert matches[0].colno == 3
        assert matches[0].line_part == "ggr"

        matches = matcher.check_lines(["goobanbaz"])
        assert len(matches) == 1
        assert matches[0].colno == 6
        assert matches[0].line_part == "baz"

    def test_regex_dont_match(self) -> None:
        matcher = ssm.SensitiveStringMatcher(
            "Basic Matcher",
            "foo",
            "**dont_match",
            "**next_is_regex",
            r"[a-z]o[a-z]",
        )
        matches = matcher.check_lines(["foo", "bar", "baz"])
        assert len(matches) == 0

        matcher = ssm.SensitiveStringMatcher(
            "Basic Matcher",
            "**all_regex",
            "foo.?",
            "**dont_match",
            "**next_is_regex",
            r"[a-z]{4}",
        )
        matches = matcher.check_lines(["foo", "bar", "baz"])
        assert len(matches) == 1
        matches = matcher.check_lines(["foobarbaz"])
        assert len(matches) == 0


if __name__ == "__main__":
    unittest.main()
