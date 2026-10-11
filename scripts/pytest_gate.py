"""Run the suite against the worktree being pushed, not whatever is installed.

`python -m pytest` inherits whatever `import arxiv_mcp_server` resolves to, and
in a dev environment that is the editable install — which points at the checkout
that ran `pip install -e .`, not at the branch under the push. The gate then
collects one branch's tests and runs them against another branch's source, so it
passes a broken push whenever that other checkout happens to be fine.

Putting the pushed worktree's `src` first establishes precedence; the child also
checks package origin before collection so missing local files cannot fall back
to another checkout, then rechecks the imported package after collection in case
pytest startup changed the path. Neither fixes *distribution* metadata, which
is why this also checks the version — see below.
"""

from __future__ import annotations

import os
import subprocess
import sys
import tomllib
from pathlib import Path

# Resolve without importing the package (and triggering server startup imports).
# Keep the check and pytest in one interpreter with the same cwd and sys.path.
_PYTEST_RUNNER = """
import importlib.util
import os
from pathlib import Path
import sys

sys.path[0] = os.getcwd()
expected = Path(sys.argv[1]).resolve() / "src/arxiv_mcp_server/__init__.py"
if not expected.is_file():
    sys.exit(f"pytest gate: missing local arxiv_mcp_server initializer: {expected}")

spec = importlib.util.find_spec("arxiv_mcp_server")
origin = Path(spec.origin).resolve() if spec and spec.origin else None
if origin != expected:
    sys.exit(
        f"pytest gate: arxiv_mcp_server resolves to {origin}; expected {expected}"
    )

import pytest

class _OriginGuard:
    def pytest_collection_finish(self):
        module = sys.modules.get("arxiv_mcp_server")
        if module is None:
            return
        filename = getattr(module, "__file__", None)
        origin = Path(filename).resolve() if filename else None
        if origin is None or not origin.is_relative_to(expected.parent):
            pytest.exit(
                f"pytest gate: collected arxiv_mcp_server from {origin}; "
                f"expected a file under {expected.parent}",
                returncode=1,
            )

sys.exit(pytest.main(["-q", *sys.argv[2:]], plugins=[_OriginGuard()]))
"""


def _worktree_root() -> Path:
    out = subprocess.run(
        ["git", "rev-parse", "--show-toplevel"],
        capture_output=True,
        text=True,
        check=True,
    )
    return Path(out.stdout.strip())


def _warn_if_metadata_is_from_another_checkout(root: Path) -> None:
    """`PYTHONPATH` redirects imports; it does not redirect `.dist-info`.

    `config.py` reads `importlib.metadata.version("arxiv-mcp-pro")`, which still
    comes from whichever checkout was installed. On a branch that changes the
    version, the suite therefore runs this source under the *other* checkout's
    `APP_VERSION` — the value advertised as the MCP server version and sent in
    outbound user agents. Overlaying the metadata is not worth the fragility, so
    the mismatch is made loud instead of left silent.
    """
    try:
        from importlib.metadata import PackageNotFoundError, version

        installed = version("arxiv-mcp-pro")
    except (ImportError, PackageNotFoundError):
        return

    try:
        with (root / "pyproject.toml").open("rb") as handle:
            declared = tomllib.load(handle)["project"]["version"]
    except (OSError, KeyError, tomllib.TOMLDecodeError):
        return

    if installed != declared:
        print(
            f"warning: this worktree declares version {declared}, but the "
            f"installed distribution metadata says {installed}. Tests read the "
            f"worktree's source and the installed version. Reinstall with "
            f"`pip install -e .` from here if the version matters to what you "
            f"are testing.",
            file=sys.stderr,
        )


def main() -> int:
    root = _worktree_root()

    env = dict(os.environ)
    existing = env.get("PYTHONPATH")
    # os.pathsep, not ":" — on Windows the separator is ";", and joining with a
    # colon would fuse `src` onto the first existing entry, leaving neither
    # importable and the editable checkout still winning.
    env["PYTHONPATH"] = os.pathsep.join(
        [str(root / "src"), *([existing] if existing else [])]
    )

    _warn_if_metadata_is_from_another_checkout(root)

    return subprocess.run(
        [sys.executable, "-c", _PYTEST_RUNNER, str(root), *sys.argv[1:]],
        cwd=root,
        env=env,
    ).returncode


if __name__ == "__main__":
    raise SystemExit(main())
