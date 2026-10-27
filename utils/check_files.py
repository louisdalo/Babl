"""A Set of functions to check files' metadata."""

##### IMPORTS ##### ------------------------------------------------------------
import re
import datetime as dt
from pathlib import Path


##### FUNCTIONS ##### ----------------------------------------------------------
def file_date(path: Path):
    """
        Extracts a date from a file's name.
        If date cannot be found in filename, uses the file's modification time.

        Parameters
        ----------
        path: Path
            The filepath to use.

        Returns
        ------
        A tuple (date, source)
    """
    pattern = r"(\d{4}-\d{2}-\d{2})"
    m = re.search(pattern, path.name)
    if m:
        return dt.date.fromisoformat(m.group(1)), "filename"
    return dt.date.fromtimestamp(path.stat().st_mtime), "file modification time"

def find_dump(data_dir: Path, pattern: re.Pattern):
    """
        A function that finds a file inside a repository based on given pattern.

        Parameters
        ----------
        data_dir: Path
            The path to the folder that contains the file to find.
        pattern: re.Pattern
            The pattern of the files that the function should look for.

        Returns
        ------
        The latest file corresponding to the pattern.
    """
    files = [p for p in data_dir.glob("*.txt.gz") if pattern.match(p.name)]
    if not files:
        return None
    return max(files, key=lambda p: file_date(p)[0])

def check_age(path: Path, max_days: int):
    """
        If file at given path is outdated relative to max_days, displays a 
        warning to the user (not breaking).

        Parameters
        ----------
        path: Path
            The dump file to check.
        max_days: int
            The age (in days) above which the warning is shown.

        Returns
        ------
        An int: the age of the file, in days.
    """
    date, source = file_date(path)
    age = (dt.date.today() - date).days
    if age > max_days:
        print(f"[WARNING]: file {path.name} is {age} days old (source: {source})\n"
              f"[WARNING]: limit before warning is {max_days} days.\n")
    return age