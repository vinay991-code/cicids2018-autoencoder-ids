"""Build the reference implementation: the real CICFlowMeter flow code (February 2018 version, commit eaa853d) driven by a small pcap reader.

Why: to check this project's flow builder (app/capture.py) against the tool that made the training data, on the same packets.
What is downloaded (not redistributed here): the CICFlowMeter source from GitHub and three small jars from Maven Central (commons-math3, slf4j).
Nothing in the downloaded flow code is changed: it is the February 2018 source as published. (An earlier build of this tool carried two patches;
running the source with and without them gave byte-identical output on every capture tried, so they were removed.) Why this version: see reports/FLOW_VALIDATION.md.
Needs: git, a JDK (javac), internet access. Run:  python tools/cicflowmeter_ref/build_reference.py
"""
from __future__ import annotations

import os
import shutil
import stat
import subprocess
import sys
import urllib.request
from pathlib import Path

HERE = Path(__file__).resolve().parent
BUILD = HERE / "_build"
COMMIT = "eaa853d"
REPO = "https://github.com/ahlashkari/CICFlowMeter.git"
JARS = {"commons-math3-3.5.jar": "https://repo1.maven.org/maven2/org/apache/commons/commons-math3/3.5/commons-math3-3.5.jar",
        "slf4j-api-1.7.25.jar": "https://repo1.maven.org/maven2/org/slf4j/slf4j-api/1.7.25/slf4j-api-1.7.25.jar",
        "slf4j-nop-1.7.25.jar": "https://repo1.maven.org/maven2/org/slf4j/slf4j-nop/1.7.25/slf4j-nop-1.7.25.jar"}
PKG = Path("src/main/java/cic/cs/unb/ca/jnetpcap")
USED = ["BasicFlow.java", "BasicPacketInfo.java", "FlowGenerator.java", "IdGenerator.java", "DateFormatter.java"]


def _force_remove(func, path, _exc):
    os.chmod(path, stat.S_IWRITE)          # git marks its objects read-only; Windows refuses to delete them otherwise
    func(path)


def classpath() -> str:
    parts = [str(BUILD / "classes")] + [str(BUILD / "libs" / j) for j in JARS]
    return os.pathsep.join(parts)


def built() -> bool:
    return (BUILD / "classes" / "RefDriver.class").exists()


def build(force: bool = False):
    if built() and not force:
        return
    if BUILD.exists():
        shutil.rmtree(BUILD, onexc=_force_remove)
    (BUILD / "libs").mkdir(parents=True)
    subprocess.run(["git", "clone", "--quiet", REPO, str(BUILD / "repo")], check=True)
    subprocess.run(["git", "-C", str(BUILD / "repo"), "checkout", "--quiet", COMMIT], check=True)
    src = BUILD / "src"
    (src / "cic/cs/unb/ca/jnetpcap/worker").mkdir(parents=True)
    (src / "org/jnetpcap/packet/format").mkdir(parents=True)
    for f in USED:
        shutil.copy(BUILD / "repo" / PKG / f, src / "cic/cs/unb/ca/jnetpcap" / f)
    shutil.copy(BUILD / "repo" / PKG / "worker/FlowGenListener.java", src / "cic/cs/unb/ca/jnetpcap/worker/FlowGenListener.java")
    shutil.copy(HERE / "shims/FormatUtils.java", src / "org/jnetpcap/packet/format/FormatUtils.java")
    shutil.copy(HERE / "shims/Utils.java", src / "cic/cs/unb/ca/jnetpcap/Utils.java")
    shutil.copy(HERE / "RefDriver.java", src / "RefDriver.java")
    for name, url in JARS.items():
        urllib.request.urlretrieve(url, BUILD / "libs" / name)
    files = [str(p) for p in src.rglob("*.java")]
    (BUILD / "classes").mkdir()
    subprocess.run(["javac", "-nowarn", "-encoding", "UTF-8", "-cp", classpath(), "-d", str(BUILD / "classes")] + files, check=True)
    print("reference built in", BUILD)


def run_reference(pcap: Path, out: Path):
    build()
    r = subprocess.run(["java", "-cp", classpath(), "RefDriver", str(pcap), str(out)], capture_output=True, text=True)
    if r.returncode:
        raise RuntimeError(r.stderr)
    return r.stderr.strip()


if __name__ == "__main__":
    build(force="--force" in sys.argv)
