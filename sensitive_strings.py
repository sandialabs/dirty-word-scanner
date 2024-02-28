import dataclasses
import re
import os

import opencsp.common.lib.file.SimpleCsv as sc
import opencsp.common.lib.process.subprocess_tools as st
import opencsp.common.lib.tool.file_tools as ft
import opencsp.common.lib.tool.log_tools as lt
import opencsp.common.lib.tool.typing_tools as tt


@dataclasses.dataclass
class Match():
    lineno: int
    colno: int
    colend: int
    line: str
    line_part: str
    matcher: 'SensitiveStringMatcher'


class SensitiveStringMatcher():
    def __init__(self, name: str, *patterns: str):
        self.name = name
        self.patterns: list[re.Pattern] = []
        self.log_type = lt.log.ERROR
        self.log = lt.error
        self.compare_to: str = None

        for pattern in patterns:
            if pattern.startswith("**"):
                directive = pattern[2:]
                if directive == "info":
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
                        start = max(start-5, 0)
                        end = min(end+5, len(line))
                        matches.append(
                            Match(lineno, start, end, line, line[start:end], self))

        for match in matches:
            if self.compare_to:
                pass  # TODO
            else:
                self.log(
                    f"{self.name} match on line {match.lineno} [{match.colno}:{match.colend}]: \"{match.line_part}\"")

        if self.compare_to:
            return []
        else:
            return matches


def build_matchers(match_csv_file: str):
    matchers: list[SensitiveStringMatcher] = []

    path, name, ext = ft.path_components(match_csv_file)
    csv = sc.SimpleCsv("Sensitive Strings", path, name+ext)
    for row in csv.rows:
        name = list(row.values())[0]
        patterns = list(row.values())[1:]
        matchers.append(SensitiveStringMatcher(name, *patterns))

    return matchers


def search_file(root_dir, file_path_name_ext: str, matchers: list[SensitiveStringMatcher]):
    lines = ft.read_text_file(os.path.join(root_dir, file_path_name_ext))
    matches: list[Match] = []
    for matcher in matchers:
        matches += matcher.check_lines([file_path_name_ext])
        matches += matcher.check_lines(lines)


def search_files(root_dir: str, matchers: list[SensitiveStringMatcher]):
    lines = st.run(
        "git ls-tree --full-tree --name-only -r HEAD", cwd=root_dir, stdout="collect", stderr="print")
    files = [line.val for line in lines]
    lt.info(f"Searching for sensitive strings in {len(files)} tracked files")
    for file in files:
        search_file(root_dir, file, matchers)


if __name__ == "__main__":
    lt.logger("C:/Users/bbean/documents/tmp/sensitive_strings_log.txt")
    matchers = build_matchers("C:/Users/bbean/documents/sensitive_strings.csv")
    search_files("C:/Users/bbean/documents/opencsp_code/", matchers)
