import re
from pathlib import Path


BANNED_REGEXES = [
    re.compile("\"" + "highvol" + "ranging" + "\""),
    re.compile("\"" + "side" + "way" + "\""),
    re.compile("\"" + "rang" + "ing" + "\""),
    re.compile("\"" + "moment" + "um" + "\""),
    re.compile("\"" + "trend" + "\""),
    re.compile("regime" + r"\s*==\s*" + "\"" + "range" + "\""),
]


def test_no_regime_aliases_present():
    project_root = Path(__file__).resolve().parents[1]

    offending = {}
    for path in project_root.rglob("*.py"):
        if any(part.startswith(".venv") for part in path.parts):
            continue
        if path.name == Path(__file__).name:
            continue

        content = path.read_text(encoding="utf-8")
        for pattern in BANNED_REGEXES:
            if pattern.search(content):
                offending.setdefault(pattern.pattern, []).append(
                    str(path.relative_to(project_root))
                )

    assert not offending, f"Found banned regime aliases: {offending}"
