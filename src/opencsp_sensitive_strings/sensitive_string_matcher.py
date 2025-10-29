"""
Handles finding sensitive strings in text.
"""
from __future__ import annotations

import dataclasses
import logging
import re

logger = logging.getLogger(__name__)


@dataclasses.dataclass
class Match:
    """Represents a match found in a line of text."""

    line_number: int
    """The line number where the match was found."""

    column_start: int
    """The starting index of the match in the line."""

    column_end: int
    """The ending index of the match in the line."""

    line: str
    """The full line of text where the match was found."""

    line_part: str
    """The part of the line that matched the pattern."""

    message: str
    """A message describing the match."""


class SensitiveStringMatcher:
    """
    A class for matching sensitive strings against patterns.

    Attributes:
        name (str):  The name of the matcher.
        patterns (list[re.Pattern | str]):  The patterns to match
            against.
        negative_patterns (list[re.Pattern | str]):  Patterns that
            should not match.
        log (callable):  The logging function to use for messages.
        case_sensitive (bool):  Flag indicating whether matching should
            be case sensitive.
    """

    def __init__(self, name: str, *patterns: str) -> None:
        """
        Initialize the object.

        Args:
            name:  The name of the matcher.
            patterns:  A variable number of patterns to process.
        """
        self.name = name
        self.patterns: list[re.Pattern | str] = []
        self.negative_patterns: list[re.Pattern | str] = []
        self.log = logging.debug
        self.case_sensitive = False
        self._process_patterns(patterns)
        if not self.case_sensitive:
            self._lowercase_patterns()

    def _process_patterns(self, patterns: tuple[str, ...]) -> None:
        """
        Process the provided patterns and configure the matcher.

        The patterns may simply be a list of patterns to search for, but
        the list may also include one or more directives (prefaced with
        ``**``) to control either the matcher or the patterns for which
        it will search.

        Args:
            patterns:  The patterns to process.
        """
        next_is_regex = False
        all_regex = False
        dont_match = False
        log_levels = {
            "debug": logging.debug,
            "info": logging.info,
            "warning": logging.warning,
            "error": logging.error,
        }
        for pattern in patterns:
            if pattern.startswith("**"):
                directive = pattern[2:]
                if directive in log_levels:
                    self.log = log_levels[directive]
                elif directive == "next_is_regex":
                    next_is_regex = True
                elif directive == "all_regex":
                    all_regex = True
                elif directive == "case_sensitive":
                    self.case_sensitive = True
                elif directive == "dont_match":
                    dont_match = True
            else:
                pattern_to_save = (
                    re.compile(pattern)
                    if (next_is_regex or all_regex)
                    else pattern
                )
                next_is_regex = False
                if dont_match:
                    self.negative_patterns.append(pattern_to_save)
                else:
                    self.patterns.append(pattern_to_save)

    def _lowercase_patterns(self) -> None:
        """Convert all patterns to lowercase for case-insensitive matching."""
        self.patterns = [
            _.lower() if isinstance(_, str) else re.compile(_.pattern.lower())
            for _ in self.patterns
        ]
        self.negative_patterns = [
            _.lower() if isinstance(_, str) else re.compile(_.pattern.lower())
            for _ in self.negative_patterns
        ]

    def _search_pattern(
        self, line: str, pattern: re.Pattern | str
    ) -> tuple[int, int] | None:
        """
        Search for a single pattern in the given line.

        Args:
            line:  The line of text to search.
            pattern:  The pattern for which to search.

        Returns:
            The start and end indices of the match, or ``None`` if no
            match is found.
        """
        if isinstance(pattern, str):
            if pattern in line:
                column_start = line.index(pattern)
                column_end = column_start + len(pattern)
                return column_start, column_end
        elif match := pattern.search(line):
            return match.span()
        return None

    def _search_patterns(
        self, line: str
    ) -> dict[re.Pattern | str, tuple[int, int]]:
        """
        Search for all configured patterns in the given line.

        Exclude anything that also matches a ``**dont_match`` pattern.

        Args:
            line:  The line of text to search.

        Returns:
            A mapping from patterns to their start and end indices in
            the line of text.
        """
        matches: dict[re.Pattern | str, tuple[int, int]] = {}
        for pattern in self.patterns:
            if columns := self._search_pattern(line, pattern):
                line_part = line[columns[0] : columns[1]]
                if not any(
                    self._search_pattern(line_part, _)
                    for _ in self.negative_patterns
                ):
                    matches[pattern] = columns
        return matches

    def check_lines(self, lines: list[str]) -> list[Match]:
        """
        Check the lines for any matches and log the results.

        Args:
            lines:  The lines of text to check.

        Returns:
            Any matches found.
        """
        matches: list[Match] = []
        for line_number, line in enumerate(lines, start=1):
            line_to_search = line if self.case_sensitive else line.lower()
            for pattern, (column_start, column_end) in self._search_patterns(
                line_to_search
            ).items():
                line_part = line[column_start:column_end]
                start_context = line[max(column_start - 5, 0) : column_start]
                end_context = line[column_end : min(column_end + 5, len(line))]
                line_context = f"{start_context}`{line_part}`{end_context}"
                match = Match(
                    line_number,
                    column_start,
                    column_end,
                    line,
                    line_part,
                    f"'{self.name}' string matched to pattern '{pattern}' on "
                    f"line {line_number} [{column_start}:{column_end}]: "
                    f'"{line_context.strip()}" ("{line.strip()}")',
                )
                matches.append(match)
                self.log(match.message)
        return matches
