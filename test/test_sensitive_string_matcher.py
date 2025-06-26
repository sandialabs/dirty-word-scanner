import pytest

from opencsp_sensitive_strings.sensitive_string_matcher import (
    SensitiveStringMatcher,
)


@pytest.mark.parametrize(
    (
        "patterns",
        "lines_to_check",
        "expected_line_numbers",
        "expected_column_starts",
        "expected_line_parts",
    ),
    [
        (["bar"], ["foo", "bar", "baz"], [2], [0], ["bar"]),
        (["foo"], ["foobarbaz"], [1], [0], ["foo"]),
        (["bar"], ["foobarbaz"], [1], [3], ["bar"]),
        (["baz"], ["foobarbaz"], [1], [6], ["baz"]),
        (
            ["foo", "**dont_match", "foo"],
            ["foo", "bar", "baz"],
            [],
            [],
            [],
        ),
        (
            ["foo", "bar", "baz"],
            ["foo", "bar", "baz"],
            [1, 2, 3],
            [0, 0, 0],
            ["foo", "bar", "baz"],
        ),
        (
            ["**case_sensitive", "foo"],
            ["foO", "fOo", "fOO", "Foo", "FoO", "FOo", "FOO", "foo"],
            [8],
            [0],
            ["foo"],
        ),
        (
            ["**next_is_regex", r"[a-z]a[a-z]"],
            ["foo", "bar", "baz"],
            [2, 3],
            [0, 0],
            ["bar", "baz"],
        ),
        (
            ["**next_is_regex", r"[a-z]o[a-z]"],
            ["foobarbaz"],
            [1],
            [0],
            ["foo"],
        ),
        (["**next_is_regex", r"[a-z]{2}r"], ["foobarbaz"], [1], [3], ["bar"]),
        (["**next_is_regex", r"[a-z]{2}z"], ["foobarbaz"], [1], [6], ["baz"]),
        (
            ["**all_regex", r"[a-z]o[a-z]", r"[a-z]{2}r", r"[a-z]{2}z"],
            ["foobarbaz"],
            [1, 1, 1],
            [0, 3, 6],
            ["foo", "bar", "baz"],
        ),
        (
            ["foo", "**next_is_regex", r"[a-z]{2}r", "baz"],
            ["foobarbaz"],
            [1, 1, 1],
            [0, 3, 6],
            ["foo", "bar", "baz"],
        ),
        (
            ["foo", "**next_is_regex", r"[a-z]{2}r", "baz"],
            ["goobarbaz"],
            [1, 1],
            [3, 6],
            ["bar", "baz"],
        ),
        (
            ["foo", "**next_is_regex", r"[a-z]{2}r", "baz"],
            ["googgrbaz"],
            [1, 1],
            [3, 6],
            ["ggr", "baz"],
        ),
        (
            ["foo", "**next_is_regex", r"[a-z]{2}r", "baz"],
            ["goobanbaz"],
            [1],
            [6],
            ["baz"],
        ),
        (
            ["foo", "**dont_match", "**next_is_regex", r"[a-z]o[a-z]"],
            ["foo", "bar", "baz"],
            [],
            [],
            [],
        ),
        (
            [
                "**all_regex",
                "foo.?",
                "**dont_match",
                "**next_is_regex",
                r"[a-z]{4}",
            ],
            ["foo", "bar", "baz"],
            [1],
            [0],
            ["foo"],
        ),
        (
            [
                "**all_regex",
                "foo.?",
                "**dont_match",
                "**next_is_regex",
                r"[a-z]{4}",
            ],
            ["foobarbaz"],
            [],
            [],
            [],
        ),
    ],
)
def test_check_lines(
    patterns: list[str],
    lines_to_check: list[str],
    expected_line_numbers: list[int],
    expected_column_starts: list[int],
    expected_line_parts: list[str],
) -> None:
    matcher = SensitiveStringMatcher("Basic Matcher", *patterns)
    matches = matcher.check_lines(lines_to_check)
    assert len(matches) == len(expected_line_numbers)
    for (
        match,
        expected_line_number,
        expected_column_start,
        expected_line_part,
    ) in zip(
        matches,
        expected_line_numbers,
        expected_column_starts,
        expected_line_parts,
    ):
        assert match.line_number == expected_line_number
        assert match.column_start == expected_column_start
        assert match.line_part == expected_line_part
