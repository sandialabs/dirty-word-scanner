import shutil
import subprocess
from logging import Logger
from pathlib import Path


class FileFinder:
    def __init__(
        self, root_search_dir: Path, logger: Logger, *, git_files_only: bool
    ) -> None:
        self.git_files_only = git_files_only
        self.logger = logger
        self.root_search_dir = root_search_dir

    def get_files_to_process(self) -> list[Path]:
        return sorted(
            set(
                self._get_tracked_files()
                if self.git_files_only
                else self._get_files_in_directory()
            )
        )

    def _get_tracked_files(self) -> list[Path]:
        files: list[Path] = []
        git = self._get_git_command()
        for command in [
            [git, "ls-tree", "--full-tree", "--name-only", "-r", "HEAD"],
            [git, "diff", "--name-only", "--cached", "--diff-filter=A"],
        ]:
            completed_process = subprocess.run(  # noqa: S603
                command,
                check=True,
                cwd=self.root_search_dir,
                stdout=subprocess.PIPE,
                text=True,
            )
            files.extend(
                Path(file)
                for file in completed_process.stdout.splitlines()
                if (self.root_search_dir / file).is_file()
            )
        self.logger.info(
            "Searching for sensitive strings in %d tracked files", len(files)
        )
        return files

    @staticmethod
    def _get_git_command() -> str:
        """
        Determine the git command to use for getting tracked files.

        Note:
            If this script is evaluated from MobaXTerm, then the
            built-in 16-bit version of git will fail.

        Returns:
            The git executable.
        """
        if (git := shutil.which("git")) is None:
            message = "'git' executable not found."
            raise RuntimeError(message)
        if "mobaxterm" in git:
            git = "git"
        return git

    def _get_files_in_directory(self) -> list[Path]:
        files = [
            file.relative_to(self.root_search_dir)
            for file in self.root_search_dir.rglob("*")
            if file.is_file()
        ]
        self.logger.info(
            "Searching for sensitive strings in %d files", len(files)
        )
        return files
