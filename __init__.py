_ss_settings_key = "sensitive_strings"
_ss_settings_default = {
    "dir": None,
    "cache_file": None,
}
"""
dir: Where to find the sensitive strings files, for use with opencsp/contrib/scripts/sensitive_strings.
cache_file: Greatly improves the speed of searching for sensitive strings by remembering which files were checked previously.
"""

_settings_list = [
    [_ss_settings_key, _ss_settings_default]
]
