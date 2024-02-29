import copy
import cv2
import hashlib
import os
import numpy as np
from PIL import Image
import sys
import time

import opencsp.common.lib.file.SimpleCsv as sc
import opencsp.common.lib.process.subprocess_tools as st
import opencsp.common.lib.tool.file_tools as ft
import opencsp.common.lib.tool.image_tools as it
import opencsp.common.lib.tool.log_tools as lt
import opencsp.common.lib.tool.time_date_tools as tdt
import SensitiveStringMatcher as ssm
import FileFingerprint as ff


class SensitiveStringsSearcher():
    def __init__(self, root_search_dir: str, sensitive_strings_csv: str, allowed_binary_files_csv: str):
        self.root_search_dir = root_search_dir
        self.sensitive_strings_csv = sensitive_strings_csv
        self.allowed_binary_files_csv = allowed_binary_files_csv
        self.interactive = False

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

    def norm_path(self, file_path, file_name_ext: str):
        return ft.norm_path(os.path.join(self.root_search_dir, file_path, file_name_ext))

    def parse_file(self, file_path: str, file_name_ext: str):
        file_path_norm: str = self.norm_path(file_path, file_name_ext)
        lt.debug(file_path_norm)

        try:
            lines = ft.read_text_file(file_path_norm)
        except UnicodeDecodeError:
            errmsg = f"    UnicodeDecodeError in sensitive_strings.search_file: assuming is a binary file \"{file_path_norm}\""
            file_ff = ff.FileFingerprint.from_file(self.root_search_dir, file_path, file_name_ext)

            if file_ff in self.allowed_binary_files:
                lt.debug(errmsg)
                self.unfound_allowed_binary_files.remove(file_ff)
                self.accepted_binary_files.append(file_ff)
            else:
                lt.warn(errmsg)
                # we'll deal with unknown files as a group
                self.unknown_binary_files.append(file_ff)

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

    def interactive_sign_off(self, file_ff: ff.FileFingerprint):
        file_norm_path = self.norm_path(file_ff.relative_path, file_ff.name_ext)
        _, name, ext = ft.path_components(file_norm_path)

        if ext.lower().lstrip(".") in it.pil_image_formats_rw:
            img = Image.open(file_norm_path).convert('RGB')
            if img.size[0] > 1920:
                scale = 1920 / img.size[0]
                img = img.resize((int(scale * img.size[0]), int(scale * img.size[1])))
            if img.size[0] > 1080:
                scale = 1080 / img.size[1]
                img = img.resize((int(scale * img.size[0]), int(scale * img.size[1])))
            cv_img = np.array(img)
            img.close()

            lt.info("Is this image safe to add, and doesn't contain any sensitive information (y/n)?")
            cv2.imshow(f"{file_ff.relative_path}/{file_ff.name_ext}", cv_img)
            key = cv2.waitKey(0)
            cv2.destroyAllWindows()

            if key == ord('y') or key == ord('Y'):
                val = 'y'
            elif key == ord('n') or key == ord('N'):
                val = 'n'
            else:
                val = '?'

        else:
            val = 'n'
            # val = input("Is this file safe to add, and doesn't contain any sensitive information (y/n)?")

        if val.lower() not in ["y", "n"]:
            lt.error("Did not respond with either 'y' or 'n'. Assuming 'n'.")
            val = 'n'
        time.sleep(0.1)  # small delay to prevent accidental double-bounces

        if val == 'y':
            return True
        else:
            return False

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
            # for file_ff in self.unfound_allowed_binary_files:
            #     lt.error(os.path.join(file_ff.relative_path, file_ff.name_ext))

        if len(self.unknown_binary_files) > 0:
            lt.error(f"Found {len(self.unknown_binary_files)} unknown binary files that aren't part of the git repository:")
            unknowns_copy = copy.copy(self.unknown_binary_files)
            for file_ff in unknowns_copy:
                lt.error(os.path.join(file_ff.relative_path, file_ff.name_ext))
                num_interactively_signed_off_files = 0

                # Maybe this file is ok? Let's ask the user to sign off on the file.
                if self.interactive and self.interactive_sign_off(file_ff):
                    self.unknown_binary_files.remove(file_ff)
                    self.allowed_binary_files.append(file_ff)
                    num_interactively_signed_off_files += 1

            # Add the interactively signed off files to the allowed binary files csv
            if num_interactively_signed_off_files > 0:
                # First, make a backup copy
                path, name, ext = ft.path_components(self.allowed_binary_files_csv)
                ft.copy_file(self.allowed_binary_files_csv, path, f"{name}_{tdt.current_date_time_string_forfile()}{ext}")

                # Overwrite the file with the updated allowed files
                self.allowed_binary_files = sorted(self.allowed_binary_files)
                file_ff.to_csv("Allowed Binary Files", path, name, rows=self.allowed_binary_files)

        return len(matches) + len(self.unfound_allowed_binary_files) + len(self.unknown_binary_files)


if __name__ == "__main__":
    log_path = "C:/Users/bbean/documents/tmp/sensitive_strings_log.txt"
    sensitive_strings_csv = "C:/Users/bbean/documents/sensitive_strings.csv"
    allowed_binary_files_csv = "C:/Users/bbean/documents/sensitive_strings_allowed_binary_files.csv"
    root_search_dir = "C:/Users/bbean/documents/opencsp_code/"

    lt.logger(log_path)
    searcher = SensitiveStringsSearcher(root_search_dir, sensitive_strings_csv, allowed_binary_files_csv)
    searcher.interactive = True
    num_errors = searcher.search_files()

    if num_errors > 0:
        sys.exit(1)
    else:
        sys.exit(0)
