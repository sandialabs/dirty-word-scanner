from __future__ import annotations

import dataclasses
import logging
import re
from typing import Optional, Union

logger = logging.getLogger(__name__)


@dataclasses.dataclass
class Match:
    lineno: int
    colno: int
    colend: int
    line: str
    line_part: str
    matcher: SensitiveStringMatcher
    msg: str = ""


class SensitiveStringMatcher:
    def __init__(self, name: str, *patterns: str) -> None:
        self.name = name
        self.patterns: list[Union[re.Pattern, str]] = []
        self.neg_patterns: list[Union[re.Pattern, str]] = []
        self.log = logging.debug
        self.case_sensitive = False

        next_is_regex = False
        all_regex = False
        remaining_negative_match = False
        for pattern in patterns:
            if pattern.startswith("**"):
                directive = pattern[2:]
                if directive == "debug":
                    self.log = logging.debug
                elif directive == "info":
                    self.log = logging.info
                elif directive == "warning":
                    self.log = logging.warning
                elif directive == "error":
                    self.log = logging.error
                elif directive == "next_is_regex":
                    next_is_regex = True
                elif directive == "all_regex":
                    all_regex = True
                elif directive == "case_sensitive":
                    self.case_sensitive = True
                elif directive == "dont_match":
                    remaining_negative_match = True

            else:
                pattern_to_save = (
                    re.compile(pattern)
                    if (next_is_regex or all_regex)
                    else pattern
                )
                if not remaining_negative_match:
                    self.patterns.append(pattern_to_save)
                else:
                    self.neg_patterns.append(pattern_to_save)

                next_is_regex = False

        # case insensitive matching
        if not self.case_sensitive:
            for pattern_list in [self.patterns, self.neg_patterns]:
                for i, pattern in enumerate(pattern_list):
                    if isinstance(pattern, str):
                        pattern_list[i] = pattern.lower()
                    else:
                        p: re.Pattern = pattern
                        pattern_list[i] = re.compile(p.pattern.lower())

    def _search_pattern(
        self, ihaystack: str, pattern: Union[re.Pattern, str]
    ) -> Optional[list[int]]:
        if isinstance(pattern, str):
            # Check for occurances of string literals
            if pattern in ihaystack:
                start = ihaystack.index(pattern)
                end = start + len(pattern)
                return [start, end]

        elif re_match := pattern.search(ihaystack):
            start, end = re_match.span()[0], re_match.span()[1]
            return [start, end]

        return None

    def _search_patterns(
        self, ihaystack: str, patterns: list[Union[re.Pattern, str]]
    ) -> dict[Union[re.Pattern, str], list[int]]:
        ret: dict[Union[re.Pattern, str], list[int]] = {}

        for pattern in patterns:
            if span := self._search_pattern(ihaystack, pattern):
                ret[pattern] = span

        return ret

    def check_lines(self, lines: list[str]) -> list[Match]:
        matches: list[Match] = []

        for lineno, line in enumerate(lines):
            iline = line if self.case_sensitive else line.lower()

            # Check for matching patterns in this line
            possible_matching = self._search_patterns(iline, self.patterns)

            # Filter out negative matches in the matching patterns
            matching: dict[Union[re.Pattern, str], list[int]] = {}
            for pattern in possible_matching:
                span = possible_matching[pattern]
                line_part = iline[span[0] : span[1]]
                if (
                    len(self._search_patterns(line_part, self.neg_patterns))
                    == 0
                ):
                    matching[pattern] = span

            # Register the matches
            for pattern, span in matching.items():
                start, end = span[0], span[1]
                line_part = line[start:end]
                line_context = f"`{line_part}`"
                if start > 0:
                    line_context = (
                        line[max(start - 5, 0) : start] + line_context
                    )
                if end < len(line):
                    line_context = (
                        line_context + line[end : min(end + 5, len(line))]
                    )

                match = Match(lineno + 1, start, end, line, line_part, self)
                self.set_match_msg(match, pattern, line_context)
                matches.append(match)
                self.log(match.msg)

        return matches

    def set_match_msg(
        self, match: Match, pattern: Union[re.Pattern, str], line_context: str
    ) -> None:
        log_msg = (
            f"'{self.name}' string matched to pattern '{pattern}' on line "
            f"{match.lineno} [{match.colno}:{match.colend}]: "
            f'"{line_context.strip()}" ("{match.line.strip()}")'
        )
        match.msg = log_msg
