from pathlib import Path

from rich.prompt import Confirm

from opencsp_sensitive_strings.sensitive_string_matcher import Match


class UserInteraction:
    def __init__(self) -> None:
        self.assume_yes = False
        """Whether to assume 'yes' for any prompts to the user."""

    def file_matches(self, file: Path, message: str) -> list[Match]:
        return [] if self.approved(file) else [Match(0, 0, 0, "", "", message)]

    def approved(self, file: Path) -> bool:
        if self.assume_yes:
            return True
        return Confirm.ask(
            f"File:  {file}.  Is this file safe to add?  Does it contain no "
            "sensitive information?"
        )


user = UserInteraction()
"""A single :class:`UserInteraction` instance to be used across all files."""
