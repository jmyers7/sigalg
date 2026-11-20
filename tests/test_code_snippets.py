import subprocess
import sys
from pathlib import Path

import pytest

SCRIPTS_DIR = Path(__file__).resolve().parent.parent / "docs" / "docs" / "scripts"
SCRIPTS = sorted(SCRIPTS_DIR.rglob("*.py"))


@pytest.mark.parametrize(
    "script", SCRIPTS, ids=lambda p: p.relative_to(SCRIPTS_DIR).as_posix()
)
def test_docs_script_runs(script, tmp_path):
    result = subprocess.run(
        [sys.executable, str(script)],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        timeout=300,
        env={**__import__("os").environ, "MPLBACKEND": "Agg"},
    )
    assert result.returncode == 0, result.stderr
