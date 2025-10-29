import logging
from argparse import ArgumentParser, BooleanOptionalAction
from pathlib import Path
from tempfile import gettempdir


class Config:
    """The configuration for the :class:`SensitiveStringsSearcher`."""

    def __init__(self) -> None:
        self.allowed_binary_files_csv = Path("allowed_binaries.csv")
        """
        A CSV file containing a list of allowed binary files.

        These are automatically considered to be contain no sensitive
        information.
        """
        self.assume_yes = False
        """Whether to assume 'yes' for any prompts to the user."""
        self.cache_file_csv: Path | None = None
        """
        An optional CSV file containing cached cleared files.

        These files have already been determined to contain no
        sensitive information, and having such a cached list speeds up
        the scanning of a repository.
        """
        self.interactive = False
        """
        Whether to operate in interactive mode.

        Whether the :class:`SensitiveStringsSearcher` will interactively
        prompt the user to verify binary files.
        """
        self.remove_unfound_binaries = False
        """
        Whether to remove unfound binary files from the allowed list.

        If any binary files present in the
        :attr:`allowed_binary_files_csv` are no longer found in the
        :attr:`root_search_dir`, remove them from the list of allowed
        binary files and overwrite the :attr:`allowed_binary_files_csv`
        file.
        """
        self.root_search_dir = Path.cwd()
        """The root directory in which to search for sensitive strings."""
        self.sensitive_strings_csv = Path("sensitive_strings.csv")
        """
        A CSV file containing the sensitive strings for which to search.

        Each line starts with a name for the rule, followed by one or
        more patterns for which to search.  See
        :class:`SensitiveStringMatcher` for details on supported search
        patterns.
        """

    def _parser(self) -> ArgumentParser:
        """
        Create the argument parser for the script.

        Returns:
            The argument parser.
        """
        argument_parser = ArgumentParser(
            prog=__file__.rstrip(".py"),
            description="Sensitive strings searcher",
        )
        argument_parser.add_argument(
            "--root-search-dir",
            help="The directory in which to search for sensitive strings.",
            required=True,
            type=Path,
        )
        argument_parser.add_argument(
            "--sensitive-strings",
            help="The CSV file defining the sensitive string patterns to "
            "search for.",
            required=True,
            type=Path,
        )
        argument_parser.add_argument(
            "--allowed-binaries",
            help="The CSV file defining the allowed binary files.",
            required=True,
            type=Path,
        )
        argument_parser.add_argument(
            "--interactive",
            action=BooleanOptionalAction,
            help="Whether to interactively ask the user about unknown binary "
            "files.",
        )
        argument_parser.add_argument(
            "--assume-yes",
            action="store_true",
            help="Don't interactively ask the user about unknown binary "
            "files.  Simply accept all as verified on the user's behalf.  "
            "This can be useful when you're confident that the only changes "
            "have been that the binary files have moved but not changed.",
        )
        argument_parser.add_argument(
            "--accept-unfound",
            action="store_true",
            help="Don't fail because of unfound expected binary files.  "
            "Instead remove the expected files from the list of allowed "
            "binaries.  This can be useful when you're confident that the "
            "only changes have been that the binary files have moved but not "
            "changed.",
        )
        argument_parser.add_argument(
            "--log-dir",
            default=Path(gettempdir()) / "sensitive_strings",
            help="The directory in which to store all logs.",
            type=Path,
        )
        argument_parser.add_argument(
            "--cache-file",
            default=None,
            help="The directory in which to store all logs.",
            type=Path,
        )
        argument_parser.add_argument(
            "--verbose",
            action="store_true",
            help="Print more information while running",
        )
        return argument_parser

    def parse_args(self, argv: list[str]) -> None:
        """
        Parse the command line arguments to the script.

        To finish setting the configuration.

        Args:
            argv:  The command line arguments to parse.
        """
        args = self._parser().parse_args(argv)
        self.allowed_binary_files_csv = Path(args.allowed_binaries)
        self.assume_yes = bool(args.assume_yes)
        self.cache_file_csv = (
            Path(args.cache_file) if args.cache_file else None
        )
        self.interactive = bool(args.interactive or args.assume_yes)
        self.remove_unfound_binaries = bool(args.accept_unfound)
        self.root_search_dir = Path(args.root_search_dir)
        self.sensitive_strings_csv = Path(args.sensitive_strings)
        log_path: Path = args.log_dir / "sensitive_strings.log"
        logging.basicConfig(
            filename=log_path,
            level=(logging.DEBUG if args.verbose else logging.INFO),
        )
