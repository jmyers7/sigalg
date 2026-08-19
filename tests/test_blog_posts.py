# import os
# import re
# from pathlib import Path

# import pytest

# POSTS_DIR = Path(__file__).parent.parent.parent / "site" / "posts"


# def uses_sigalg(path: Path) -> bool:
#     text = path.read_text()
#     return "import sigalg" in text or "from sigalg" in text


# def extract_title(path: Path) -> str:
#     text = path.read_text()
#     match = re.search(r'^title:\s*["\']?(.+?)["\']?\s*$', text, re.MULTILINE)
#     return match.group(1) if match else path.parent.name


# def extract_python_cells(path: Path) -> list[str]:
#     cells = []
#     in_cell = False
#     current_lines = []

#     for line in path.read_text().splitlines():
#         if not in_cell and re.fullmatch(r"```\{python\}", line.strip()):
#             in_cell = True
#             current_lines = []
#         elif in_cell and line.strip() == "```":
#             in_cell = False
#             cells.append("\n".join(current_lines))
#         elif in_cell:
#             current_lines.append(line)

#     return cells


# def strip_options(cell: str) -> str:
#     lines = cell.splitlines()
#     code_lines = [line for line in lines if not re.match(r"#\|", line)]
#     return "\n".join(code_lines)


# qmd_files = sorted(f for f in POSTS_DIR.glob("**/*.qmd") if uses_sigalg(f))
# qmd_titles = [extract_title(f) for f in qmd_files]


# @pytest.mark.parametrize("post", qmd_files, ids=qmd_titles)
# def test_qmd_executes(post: Path):
#     cells = extract_python_cells(post)
#     namespace = {}
#     original_dir = Path.cwd()
#     try:
#         os.chdir(post.parent)
#         for cell in cells:
#             exec(strip_options(cell), namespace)
#     finally:
#         os.chdir(original_dir)
