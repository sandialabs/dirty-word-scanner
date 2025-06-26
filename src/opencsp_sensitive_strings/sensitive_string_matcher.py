from __future__ import annotations

import dataclasses
import logging
import re
from typing import Optional, Union

logger = logging.getLogger(__name__)


@dataclasses.dataclass
class Match:
    line_number: int
    column_start: int
    column_end: int
    line: str
    line_part: str
    message: str


class SensitiveStringMatcher:
    def __init__(self, name: str, *patterns: str) -> None:
        self.name = name
        self.patterns: list[Union[re.Pattern, str]] = []
        self.negative_patterns: list[Union[re.Pattern, str]] = []
        self.log = logging.debug
        self.case_sensitive = False
        self._process_patterns(patterns)
        if not self.case_sensitive:
            self._lowercase_patterns()

    def _process_patterns(self, patterns: tuple[str, ...]) -> None:
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
        self.patterns = [
            _.lower() if isinstance(_, str) else re.compile(_.pattern.lower())
            for _ in self.patterns
        ]
        self.negative_patterns = [
            _.lower() if isinstance(_, str) else re.compile(_.pattern.lower())
            for _ in self.negative_patterns
        ]

    def _search_pattern(
        self, line: str, pattern: Union[re.Pattern, str]
    ) -> Optional[tuple[int, int]]:
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
    ) -> dict[Union[re.Pattern, str], tuple[int, int]]:
        matches: dict[Union[re.Pattern, str], tuple[int, int]] = {}
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
