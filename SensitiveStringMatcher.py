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
        self.patterns: list[re.Pattern] = []
        self.log_type = lt.log.DEBUG
        self.log = lt.debug
        self.compare_to: str = None

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
            else:
                pattern = re.compile(pattern)
                self.patterns.append(pattern)

    def check_lines(self, lines: list[str]):
        matches: list[Match] = []

        for lineno, line in enumerate(lines):
            for pattern in self.patterns:
                match = pattern.match(line)
                if match:
                    for group in range(len(match.groups())):
                        start, end = match.start(group), match.end(group)
                        start = max(start - 5, 0)
                        end = min(end + 5, len(line))
                        match = Match(lineno, start, end, line, line[start:end], self)
                        self.set_match_msg(match)
                        self.log(match.msg)

        if self.compare_to:
            return []
        else:
            return matches

    def set_match_msg(self, match: Match):
        log_msg = f"{self.name} match on line {match.lineno} [{match.colno}:{match.colend}]: \"{match.line_part}\""
        match.msg = log_msg
