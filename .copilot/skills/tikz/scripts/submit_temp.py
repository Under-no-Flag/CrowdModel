#!/usr/bin/env python3
"""Submit a file or directory to the temp preview server.

Usage:
    python3 submit_temp.py <path>              # copy file
    python3 submit_temp.py --link <path>       # symlink instead of copy
    python3 submit_temp.py --name foo.md <path> # custom name in temp dir

Output (two lines):
    Line 1: preview URL
    Line 2: path in temp dir
"""

import argparse
import shutil
import sys
from pathlib import Path

from config import get_host, get_port
from ensure_server import ensure

CACHE_DIR = Path.home() / ".cache" / "tikz-skill"
TEMP_DIR = CACHE_DIR / "temp"


def main():
    parser = argparse.ArgumentParser(description="Submit file/dir to temp preview")
    parser.add_argument("path", help="File or directory to submit")
    parser.add_argument("--link", action="store_true", help="Symlink instead of copy")
    parser.add_argument("--name", help="Custom name in temp dir (default: original name)")
    args = parser.parse_args()

    src = Path(args.path).resolve()
    if not src.exists():
        print(f"ERROR: {args.path} does not exist", file=sys.stderr)
        sys.exit(1)

    TEMP_DIR.mkdir(parents=True, exist_ok=True)

    name = args.name or src.name
    dest = TEMP_DIR / name

    # Remove existing dest if present
    if dest.exists() or dest.is_symlink():
        if dest.is_dir() and not dest.is_symlink():
            shutil.rmtree(dest)
        else:
            dest.unlink()

    if args.link:
        dest.symlink_to(src)
    else:
        if src.is_dir():
            shutil.copytree(src, dest)
        else:
            shutil.copy2(src, dest)

    # Ensure server is running
    ensure()

    host = get_host()
    port = get_port()

    if dest.is_dir():
        url = f"http://{host}:{port}/temp/{name}/"
    elif dest.suffix == ".md":
        url = f"http://{host}:{port}/temp/{name}"
    else:
        url = f"http://{host}:{port}/temp/{name}"

    print(url)
    print(dest)


if __name__ == "__main__":
    main()
