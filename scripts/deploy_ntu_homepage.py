#!/usr/bin/env python3
"""Replace the NTU public_html contents through explicit FTPS."""

from __future__ import annotations

import argparse
import base64
import ftplib
from pathlib import Path
import xml.etree.ElementTree as ET


ROOT = Path(__file__).resolve().parents[1]
LOCAL_SITE = ROOT / "review_ui/public_site"
FILEZILLA_CONFIG = Path.home() / "AppData/Roaming/FileZilla/sitemanager.xml"
HOST = "homepage.ntu.edu.tw"
USER = "b13901110"


def credentials() -> tuple[str, str]:
    env_path = ROOT / ".env"
    if env_path.exists():
        for line in env_path.read_text(encoding="utf-8-sig").splitlines():
            if "=" not in line or line.lstrip().startswith("#"):
                continue
            name, value = line.split("=", 1)
            if name.strip().strip("\"'") == "NTU_FTPS_PASSWORD":
                password = value.strip().strip("\"'")
                if password:
                    return USER, password
    tree = ET.parse(FILEZILLA_CONFIG)
    for server in tree.findall(".//Server"):
        if server.findtext("Host") == HOST and server.findtext("User") == USER:
            password = server.find("Pass")
            if password is None or not password.text:
                break
            value = password.text
            if password.get("encoding") == "base64":
                value = base64.b64decode(value).decode("utf-8")
            return USER, value
    raise RuntimeError(
        "No FTPS password is available. Add NTU_FTPS_PASSWORD to the local .env "
        "or save the password in the matching FileZilla site."
    )


def remove_tree(ftp: ftplib.FTP_TLS, name: str) -> None:
    original = ftp.pwd()
    try:
        ftp.cwd(name)
    except ftplib.error_perm:
        ftp.delete(name)
        return
    for child, facts in list(ftp.mlsd()):
        if child in {".", ".."}:
            continue
        if facts.get("type") == "dir":
            remove_tree(ftp, child)
        else:
            ftp.delete(child)
    ftp.cwd(original)
    ftp.rmd(name)


def upload_tree(ftp: ftplib.FTP_TLS, local: Path) -> None:
    for path in sorted(local.iterdir()):
        if path.is_dir():
            try:
                ftp.mkd(path.name)
            except ftplib.error_perm as error:
                if not str(error).startswith("550"):
                    raise
            ftp.cwd(path.name)
            upload_tree(ftp, path)
            ftp.cwd("..")
        else:
            with path.open("rb") as handle:
                ftp.storbinary(f"STOR {path.name}", handle)
            print(f"uploaded {path.relative_to(LOCAL_SITE).as_posix()}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="Deploy static review UI to NTU homepage FTPS")
    parser.add_argument("--check", action="store_true", help="verify login and list remote names without changing files")
    parser.add_argument("--replace", action="store_true", help="delete existing public_html children before upload")
    args = parser.parse_args()
    if args.check == args.replace:
        raise SystemExit("Choose exactly one of --check or --replace")
    required = {"index.html", "app.js", "styles.css", "cases.json"}
    missing = [name for name in required if not (LOCAL_SITE / name).is_file()]
    if missing:
        raise RuntimeError(f"Static export incomplete: {missing}")

    username, password = credentials()
    with ftplib.FTP_TLS(HOST, timeout=60) as ftp:
        ftp.login(username, password)
        ftp.prot_p()
        ftp.cwd("public_html")
        remote_root = ftp.pwd().rstrip("/")
        if not remote_root.endswith("/public_html"):
            raise RuntimeError(f"Unsafe remote deployment root: {remote_root}")
        if args.check:
            entries = [name for name, _ in ftp.mlsd() if name not in {".", ".."}]
            print(f"connected {username}@{HOST} root={remote_root} entries={entries}")
            return
        for name, _ in list(ftp.mlsd()):
            if name not in {".", ".."}:
                remove_tree(ftp, name)
        upload_tree(ftp, LOCAL_SITE)
        print(f"deployed {len(list(LOCAL_SITE.rglob('*')))} local entries to {remote_root}")


if __name__ == "__main__":
    main()
