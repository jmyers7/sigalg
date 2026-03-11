import os
import runpy
from pathlib import Path

import pytest

SNIPPETS_DIR = (
    Path(__file__).parent.parent / "docs" / "docs" / "code_snippets" / "scripts"
)

snippet_files = sorted(SNIPPETS_DIR.glob("*.py"))


@pytest.mark.parametrize("script", snippet_files, ids=[f.name for f in snippet_files])
def test_snippet(script):
    original_dir = Path.cwd()
    try:
        os.chdir(script.parent)
        runpy.run_path(str(script))
    finally:
        os.chdir(original_dir)
