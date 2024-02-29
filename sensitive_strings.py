import copy
import hashlib
import os
import sys

import opencsp.common.lib.file.SimpleCsv as sc
import opencsp.common.lib.process.subprocess_tools as st
import opencsp.common.lib.tool.file_tools as ft
import opencsp.common.lib.tool.log_tools as lt
import SensitiveStringMatcher as ssm
import FileFingerprint as ff


class SensitiveStringsSearcher():
    def __init__(self, root_search_dir: str, sensitive_strings_csv: str, allowed_binary_files_csv: str):
        self.root_search_dir = root_search_dir
        self.sensitive_strings_csv = sensitive_strings_csv
        self.allowed_binary_files_csv = allowed_binary_files_csv

        self.matchers = self.build_matchers()
        path, name, ext = ft.path_components(self.allowed_binary_files_csv)
        self.allowed_binary_files: list[ff.FileFingerprint] = ff.FileFingerprint.from_csv("Allowed Binary Files", path, name + ext)
        self.accepted_binary_files: list[ff.FileFingerprint] = []
        self.unknown_binary_files: list[ff.FileFingerprint] = []
        self.unfound_allowed_binary_files: list[ff.FileFingerprint] = copy.copy(self.allowed_binary_files)

    def build_matchers(self):
        matchers: list[ssm.SensitiveStringMatcher] = []

        path, name, ext = ft.path_components(self.sensitive_strings_csv)
        csv = sc.SimpleCsv("Sensitive Strings", path, name + ext)
        for row in csv.rows:
            name = list(row.values())[0]
            patterns = list(row.values())[1:]
            matchers.append(ssm.SensitiveStringMatcher(name, *patterns))

        return matchers

    def parse_file(self, file_path: str, file_name_ext: str):
        file_path_norm = ft.norm_path(os.path.join(self.root_search_dir, file_path, file_name_ext))
        lt.debug(file_path_norm)

        try:
            lines = ft.read_text_file(file_path_norm)
        except UnicodeDecodeError:
            errmsg = f"    UnicodeDecodeError in sensitive_strings.search_file: assuming is a binary file \"{file_path_norm}\""
            path, name, ext = ft.path_components(file_path_norm)
            file_size = ft.file_size(file_path_norm)
            with open(file_path_norm, "rb") as fin:
                file_hash = hashlib.sha256(fin.read()).hexdigest()
            file_fp = ff.FileFingerprint(file_path, name + ext, file_size, file_hash)

            if file_fp in self.allowed_binary_files:
                lt.debug(errmsg)
                self.unfound_allowed_binary_files.remove(file_fp)
                self.accepted_binary_files.append(file_fp)
            else:
                lt.warn(errmsg)
                # we'll deal with unknown files as a group
                self.unknown_binary_files.append(file_fp)

            # don't return anything
            lines = []

        return lines

    def search_file(self, file_path: str, file_name_ext: str):
        lines = self.parse_file(file_path, file_name_ext)

        matches: list[ssm.Match] = []
        for matcher in self.matchers:
            matches += matcher.check_lines([file_path + "/" + file_name_ext])
            matches += matcher.check_lines(lines)

        return matches

    def search_files(self):
        git_stdout = st.run(
            "git ls-tree --full-tree --name-only -r HEAD", cwd=self.root_search_dir, stdout="collect", stderr="print")
        files = sorted([line.val for line in git_stdout])
        lt.info(f"Searching for sensitive strings in {len(files)} tracked files")

        matches: dict[str, list[ssm.Match]] = {}
        for file_path_name_ext in files:
            path, name, ext = ft.path_components(file_path_name_ext)
            file_matches = self.search_file(path, name + ext)
            if len(file_matches) > 0:
                matches[file_path_name_ext] = file_matches

        if len(matches) > 0:
            lt.error(f"Found {len(matches)} files containing sensitive strings:")
            for file in matches:
                lt.error(f"    File {file}:")
                for match in matches[file]:
                    lt.error(f"        {match.msg}")

        if len(self.unfound_allowed_binary_files) > 0:
            lt.error(f"Expected {len(self.unfound_allowed_binary_files)} binary files that aren't part of the git repository:")
            # for file_fp in self.unfound_allowed_binary_files:
            #     lt.error(os.path.join(file_fp.relative_path, file_fp.name_ext))

        if len(self.unknown_binary_files) > 0:
            lt.error(f"Found {len(self.unknown_binary_files)} unknown binary files that aren't part of the git repository:")
            # for file_fp in self.unknown_binary_files:
            #     lt.error(os.path.join(file_fp.relative_path, file_fp.name_ext))

        return len(matches) + len(self.unfound_allowed_binary_files) + len(self.unknown_binary_files)


if __name__ == "__main__":
    log_path = "C:/Users/bbean/documents/tmp/sensitive_strings_log.txt"
    sensitive_strings_csv = "C:/Users/bbean/documents/sensitive_strings.csv"
    allowed_binary_files_csv = "C:/Users/bbean/documents/sensitive_strings_allowed_binary_files.csv"
    root_search_dir = "C:/Users/bbean/documents/opencsp_code/"

    lt.logger(log_path)
    searcher = SensitiveStringsSearcher(root_search_dir, sensitive_strings_csv, allowed_binary_files_csv)
    num_errors = searcher.search_files()

    if num_errors > 0:
        sys.exit(1)
    else:
        sys.exit(0)
