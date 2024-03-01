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
        all_regex = False
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
                elif directive == "all_regex":
                    all_regex = True
                elif directive == "case_sensitive":
                    self.case_sensitive = True

            else:
                if next_is_regex or all_regex:
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
                    re_match = pattern.search(line.lower())
                    if re_match:
                        if len(re_match.groups()) >= 1:
                            line_part = re_match[1]
                            start = line.index(line_part)
                            end = start + len(line_part)
                        else:
                            start, end = re_match.span()[0], re_match.span()[1]

                # There was a match, record it
                if start >= 0 and end >= 0:
                    line_part = f"`{line[start:end]}`"
                    if start > 0:
                        line_part = line[max(start - 5, 0):start] + line_part
                    if end < len(line):
                        line_part = line_part + line[end:min(end + 5, len(line))]

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
            f"[{match.colno}:{match.colend}]: \"{match.line_part}\" (\"{match.line.strip()}\")"
        match.msg = log_msg
