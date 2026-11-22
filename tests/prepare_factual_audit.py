"""Recover diagnostic-only tokenizer tools; never install into the app environment."""

import hashlib
import json
import urllib.request
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CACHE = ROOT / ".cache/factual-audit-tools"


def fetch(url):
    request = urllib.request.Request(
        url, headers={"User-Agent": "summarizer-factual-audit"}
    )
    with urllib.request.urlopen(request, timeout=60) as response:
        return response.read()


def main():
    CACHE.mkdir(parents=True, exist_ok=True)
    release = "https://api.github.com/repos/adoptium/temurin17-binaries/releases/tags/jdk-17.0.21.1%2B2"
    metadata = json.loads(fetch(release))
    asset = next(
        a
        for a in metadata["assets"]
        if "jre_x64_windows_hotspot" in a["name"] and a["name"].endswith(".zip")
    )
    url = asset["browser_download_url"]
    expected = fetch(url + ".sha256.txt").decode().split()[0]
    archive = CACHE / asset["name"]
    if not archive.exists():
        archive.write_bytes(fetch(url))
    assert hashlib.sha256(archive.read_bytes()).hexdigest() == expected
    destination = CACHE / "jre"
    if not list(destination.glob("*/bin/java.exe")):
        with zipfile.ZipFile(archive) as package:
            # Extract only into the explicitly resolved diagnostic cache.
            for member in package.infolist():
                assert (
                    (destination / member.filename)
                    .resolve()
                    .is_relative_to(destination.resolve())
                )
            package.extractall(destination)
    jar_url = "https://repo.maven.apache.org/maven2/edu/stanford/nlp/stanford-corenlp/3.7.0/stanford-corenlp-3.7.0.jar"
    jar = CACHE / "stanford-corenlp-3.7.0.jar"
    expected_sha1 = fetch(jar_url + ".sha1").decode().strip()
    if not jar.exists():
        jar.write_bytes(fetch(jar_url))
    assert hashlib.sha1(jar.read_bytes()).hexdigest() == expected_sha1
    report = {
        "scope": "Portable diagnostic tools only; no app dependencies, PATH or system Java installation changed",
        "java": str(next(destination.glob("*/bin/java.exe")).relative_to(ROOT)),
        "java_release": metadata["tag_name"],
        "java_source": url,
        "java_sha256": expected,
        "jar": str(jar.relative_to(ROOT)),
        "jar_source": jar_url,
        "jar_publisher_sha1": expected_sha1,
        "jar_sha256": hashlib.sha256(jar.read_bytes()).hexdigest(),
        "preprocessing_source": "https://github.com/abisee/cnn-dailymail/blob/master/make_datafiles.py",
        "version_source": "https://github.com/abisee/cnn-dailymail/blob/master/README.md",
    }
    (CACHE / "provenance.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
