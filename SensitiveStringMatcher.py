import dataclasses
import re
from typing import Callable

import opencsp.common.lib.tool.log_tools as lt


@dataclasses.dataclass
class Match():
    lineno: int
    colno: int
    colend: int
    line: str
    line_part: str
    matcher: 'SensitiveStringMatcher'
    msg: str = ""


class SensitiveStringMatcher():
    def __init__(self, name: str, *patterns: str):
        self.name = name
        self.patterns: list[re.Pattern | str] = []
        self.log_type = lt.log.DEBUG
        self.log = lt.debug
        self.compare_to: str = None
        self.case_sensitive = False

        next_is_regex = False
        for pattern in patterns:
            if pattern.startswith("**"):
                directive = pattern[2:]
                if directive == "debug":
                    self.log_type = lt.log.DEBUG
                    self.log = lt.debug
                elif directive == "info":
                    self.log_type = lt.log.INFO
                    self.log = lt.info
                elif directive == "warn":
                    self.log_type = lt.log.WARN
                    self.log = lt.warn
                elif directive == "error":
                    self.log_type = lt.log.ERROR
                    self.log = lt.error
                elif directive.startswith("compare_to="):
                    self.compare_to = directive[len("compare_to="):]
                elif directive == "next_is_regex":
                    next_is_regex = True
                elif directive == "case_sensitive":
                    self.case_sensitive = True

            else:
                if next_is_regex:
                    pattern = re.compile(pattern.lower())
                    self.patterns.append(pattern)
                else:
                    self.patterns.append(pattern)
                next_is_regex = False

    def check_lines(self, lines: list[str]):
        matches: list[Match] = []

        for lineno, line in enumerate(lines):
            for pattern in self.patterns:
                start, end = -1, -1

                if isinstance(pattern, str):
                    # Check for occurances of string literals
                    ipattern = pattern
                    iline = line
                    if not self.case_sensitive:
                        ipattern = pattern.lower()
                        iline = line.lower()
                    if ipattern in iline:
                        start = iline.index(ipattern)
                        end = start + len(pattern)

                else:
                    # Check for instances of regex matches
                    re_match = pattern.match(line.lower())
                    if re_match:
                        for group in range(len(re_match.groups())):
                            start, end = re_match.start(group), re_match.end(group)
                        if re_match.pos >= 0 and re_match.endpos >= 0:
                            start, end = re_match.pos, re_match.endpos

                # There was a match, record it
                if start >= 0 and end >= 0:
                    lpstart, lpend = max(start - 5, 0), min(end + 5, len(line))
                    line_part = line[lpstart:lpend]
                    match = Match(lineno, start, end, line, line_part, self)
                    self.set_match_msg(match, pattern)
                    matches.append(match)
                    self.log(match.msg)

        if self.compare_to:
            return []  # TODO
        else:
            return matches

    def set_match_msg(self, match: Match, pattern: re.Pattern | str):
        log_msg = f"'{self.name}' string matched to pattern '{pattern}' on line {match.lineno} " + \
            f"[{match.colno}:{match.colend}]: \"{match.line_part}\" (\"{match.line}\")"
        match.msg = log_msg
