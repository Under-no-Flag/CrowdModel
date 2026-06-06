#!/usr/bin/env python3
"""Manage TikZ render server lifecycle: ensure running or stop."""

import argparse
import json
import os
import signal
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

from config import get_host, get_port

CACHE_DIR = Path.home() / ".cache" / "tikz-skill"
STATE_FILE = CACHE_DIR / "server.json"


def get_server_info():
    """Read server state file."""
    if not STATE_FILE.exists():
        return None
    try:
        return json.loads(STATE_FILE.read_text())
    except (json.JSONDecodeError, OSError):
        return None


def is_pid_alive(pid):
    try:
        os.kill(pid, 0)
        return True
    except (OSError, ProcessLookupError):
        return False


def healthcheck(port, timeout=1.0):
    """Check if server responds on localhost."""
    try:
        url = f"http://127.0.0.1:{port}/healthz"
        resp = urllib.request.urlopen(url, timeout=timeout)
        data = json.loads(resp.read())
        return data.get("status") == "ok"
    except Exception:
        return False


def start_server():
    """Start the TikZ server as a background daemon."""
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    log_file = CACHE_DIR / "server.log"
    server_script = Path(__file__).parent / "tikz_server.py"
    port = get_port()
    host = get_host()

    with open(log_file, "a") as log:
        subprocess.Popen(
            [sys.executable, str(server_script)],
            start_new_session=True,
            stdout=log,
            stderr=log,
            cwd=str(CACHE_DIR),
        )

    # Wait for server to be ready (up to 5 seconds)
    for _ in range(25):
        time.sleep(0.2)
        if healthcheck(port):
            print(f"Server ready at http://{host}:{port}")
            return True

    print("ERROR: Server failed to start within 5 seconds", file=sys.stderr)
    return False


def find_pid_by_port(port):
    """Find PID listening on the given port via lsof."""
    try:
        out = subprocess.check_output(
            ["lsof", "-ti", f":{port}"], text=True, stderr=subprocess.DEVNULL,
        )
        for line in out.strip().splitlines():
            return int(line)
    except Exception:
        return None


def kill_pid(pid):
    """SIGTERM then wait, SIGKILL if needed."""
    os.kill(pid, signal.SIGTERM)
    for _ in range(15):
        time.sleep(0.2)
        if not is_pid_alive(pid):
            return
    try:
        os.kill(pid, signal.SIGKILL)
    except OSError:
        pass


def stop_server():
    """Stop the running TikZ server."""
    port = get_port()
    pid = None

    # Try state file first
    info = get_server_info()
    if info:
        pid = info.get("pid")

    # Fallback: find by port
    if not pid or not is_pid_alive(pid):
        pid = find_pid_by_port(port)

    if pid and is_pid_alive(pid):
        kill_pid(pid)
        print(f"Server (pid={pid}) stopped")
    else:
        print("No server running")

    STATE_FILE.unlink(missing_ok=True)


def ensure():
    """Ensure server is running. Start if needed."""
    port = get_port()
    host = get_host()

    # Probe port — try a few times since server might be slow to respond
    for _ in range(3):
        if healthcheck(port, timeout=2.0):
            print(f"Server already running at http://{host}:{port}")
            return True
        time.sleep(0.5)

    # Port occupied but not responding to healthcheck — kill the stale process
    stale_pid = find_pid_by_port(port)
    if stale_pid:
        kill_pid(stale_pid)
        time.sleep(0.5)

    # Also clean up state file
    STATE_FILE.unlink(missing_ok=True)

    return start_server()


def main():
    parser = argparse.ArgumentParser(description="Manage TikZ render server")
    parser.add_argument("--stop", action="store_true", help="Stop the running server")
    args = parser.parse_args()

    if args.stop:
        stop_server()
    else:
        success = ensure()
        sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
