"""Exercise the real pre-push gate against isolated, competing checkouts."""

import os
from pathlib import Path
import shutil
import subprocess
import sys

import pytest

GATE = Path(__file__).resolve().parents[1] / "scripts" / "pytest_gate.py"


@pytest.fixture
def competing_checkouts(tmp_path):
    pushed = tmp_path / "pushed"
    neighbor = tmp_path / "neighbor"
    for root, sentinel in ((pushed, "PUSHED"), (neighbor, "NEIGHBOR")):
        package = root / "src" / "arxiv_mcp_server"
        package.mkdir(parents=True)
        (package / "__init__.py").write_text(
            f'SENTINEL = "{sentinel}"\nHEALTHY = True\n', encoding="utf-8"
        )
    empty_template = tmp_path / "empty-template"
    empty_template.mkdir()
    # Hooks export repository-local GIT_* variables; never let them redirect
    # this synthetic init into the repository being pushed.
    git_env = {
        key: value for key, value in os.environ.items() if not key.startswith("GIT_")
    }
    subprocess.run(
        ["git", "init", "-q", f"--template={empty_template}", str(pushed)],
        env=git_env,
        check=True,
        capture_output=True,
    )
    (pushed / "pyproject.toml").write_text(
        '[project]\nname = "arxiv-mcp-pro"\nversion = "0.0.0"\n',
        encoding="utf-8",
    )
    tests = pushed / "tests"
    tests.mkdir()
    (tests / "test_sentinel.py").write_text(
        "from pathlib import Path\n"
        "import arxiv_mcp_server as subject\n"
        'Path("collected").touch()\n'
        "def test_sentinel():\n"
        '    Path("executed").touch()\n'
        '    print("SENTINEL=" + subject.SENTINEL)\n'
        '    print("ORIGIN=" + subject.__file__)\n'
        "    assert subject.HEALTHY\n",
        encoding="utf-8",
    )
    return pushed, neighbor


def _run_gate(pushed, neighbor):
    # No installs or network: the inherited path supplies a healthy neighbor.
    env = {
        key: os.environ[key]
        for key in ("PATH", "SYSTEMROOT", "TMPDIR", "TEMP", "TMP", "USERPROFILE")
        if key in os.environ
    }
    env.update(
        PYTHONPATH=str(neighbor / "src"),
        PYTEST_DISABLE_PLUGIN_AUTOLOAD="1",
        PYTHONDONTWRITEBYTECODE="1",
    )
    return subprocess.run(
        [sys.executable, str(GATE), "-s", "-p", "no:cacheprovider"],
        cwd=pushed,
        env=env,
        capture_output=True,
        text=True,
        timeout=30,
    )


@pytest.mark.parametrize("missing", ["initializer", "package"])
def test_gate_refuses_missing_local_package_before_collection(
    competing_checkouts, missing
):
    """Removing the guard must expose the neighbor's false-green result."""
    pushed, neighbor = competing_checkouts
    package = pushed / "src" / "arxiv_mcp_server"
    if missing == "initializer":
        (package / "__init__.py").unlink()
    else:
        shutil.rmtree(package)

    result = _run_gate(pushed, neighbor)

    assert result.returncode != 0, result.stdout + result.stderr
    assert not (pushed / "collected").exists(), result.stdout + result.stderr
    assert "pytest gate:" in result.stderr


@pytest.mark.parametrize("healthy, expected_exit", [(True, 0), (False, 1)])
def test_gate_tests_local_source_and_propagates_pytest_exit(
    competing_checkouts, healthy, expected_exit
):
    """An intact local package must win, even when its tests fail."""
    pushed, neighbor = competing_checkouts
    initializer = pushed / "src" / "arxiv_mcp_server" / "__init__.py"
    initializer.write_text(
        f'SENTINEL = "PUSHED"\nHEALTHY = {healthy}\n', encoding="utf-8"
    )

    result = _run_gate(pushed, neighbor)

    assert result.returncode == expected_exit, result.stdout + result.stderr
    assert "SENTINEL=PUSHED" in result.stdout
    origin = next(
        line.removeprefix("ORIGIN=")
        for line in result.stdout.splitlines()
        if line.startswith("ORIGIN=")
    )
    assert Path(origin).resolve() == initializer.resolve()


def test_gate_refuses_a_shadowing_package_before_collection(competing_checkouts):
    """Checking local files alone must not overlook the child's import origin."""
    pushed, neighbor = competing_checkouts
    shadow = pushed / "arxiv_mcp_server"
    shadow.mkdir()
    (shadow / "__init__.py").write_text(
        'SENTINEL = "SHADOW"\nHEALTHY = True\n', encoding="utf-8"
    )

    result = _run_gate(pushed, neighbor)

    assert result.returncode != 0, result.stdout + result.stderr
    assert not (pushed / "collected").exists(), result.stdout + result.stderr
    assert "pytest gate:" in result.stderr


def test_gate_refuses_neighbor_imported_after_conftest_changes_path(
    competing_checkouts,
):
    """A preflight-only guard misses path changes during pytest startup."""
    pushed, neighbor = competing_checkouts
    (pushed / "conftest.py").write_text(
        f"import sys\nsys.path.insert(0, {str(neighbor / 'src')!r})\n",
        encoding="utf-8",
    )

    result = _run_gate(pushed, neighbor)

    assert result.returncode != 0, result.stdout + result.stderr
    assert "pytest gate:" in result.stdout + result.stderr
    assert (pushed / "collected").exists(), result.stdout + result.stderr
    assert not (pushed / "executed").exists(), result.stdout + result.stderr
