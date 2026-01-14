#!/usr/bin/env python3
from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path
import sys


def parse_corpus(text: str) -> dict:
    lines = text.splitlines()
    corpus_title = None
    toc_lines: list[str] = []

    for line in lines:
        stripped = line.strip()
        if stripped.startswith("CORPUS_TITLE:"):
            corpus_title = stripped.split(":", 1)[1].strip()

    toc_start = None
    for index, line in enumerate(lines):
        if line.strip() == "TABLE_OF_CONTENTS:":
            toc_start = index + 1
            break

    if toc_start is not None:
        for line in lines[toc_start:]:
            stripped = line.strip()
            if stripped in {"<<<END_CORPUS_HEADER>>>", "<<<BEGIN_TEXT>>>", "<<<BEGIN_CORPUS>>>"}:
                break
            toc_lines.append(line.rstrip())

    if not corpus_title:
        raise ValueError("Missing CORPUS_TITLE in header.")

    while toc_lines and not toc_lines[-1].strip():
        toc_lines.pop()

    text_count = text.count("<<<BEGIN_TEXT>>>")

    return {
        "corpus_title": corpus_title,
        "toc_lines": toc_lines,
        "text_count": text_count,
    }


def build_master_header(entries: list[dict]) -> str:
    total_texts = sum(entry["text_count"] for entry in entries)
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    lines: list[str] = [
        "<<<BEGIN_CORPUS>>>",
        "CORPUS_TITLE: ++MASTER_CORPUS",
        f"CREATED_AT: {now}",
        f"TEXT_COUNT: {total_texts}",
        f"SOURCE_COUNT: {len(entries)}",
        "NOTES:",
        "- Merged from binder corpora in filename-sorted order.",
        "- Each binder corpus retains its original header and internal TOC.",
        "- Master TOC lists binder corpora and their internal TOC entries.",
        "",
        "TABLE_OF_CONTENTS:",
    ]

    for index, entry in enumerate(entries, start=1):
        lines.append(
            f"{index:02d}. {entry['corpus_title']} "
            f"(texts: {entry['text_count']}) | file: {entry['file_name']}"
        )
        for toc_line in entry["toc_lines"]:
            if toc_line.strip():
                lines.append(f"    {toc_line}")
            else:
                lines.append("")

    lines.append("<<<END_CORPUS_HEADER>>>")
    return "\n".join(lines)


def merge_corpora(folder: Path, output: Path) -> Path:
    if not folder.exists() or not folder.is_dir():
        raise FileNotFoundError(f"Folder not found: {folder}")

    corpus_files = sorted(p for p in folder.glob("*.txt") if p.resolve() != output.resolve())
    if not corpus_files:
        raise FileNotFoundError("No .txt files found to merge.")

    entries = []
    contents = []
    for path in corpus_files:
        text = path.read_text(encoding="utf-8")
        info = parse_corpus(text)
        info["file_name"] = path.name
        entries.append(info)
        contents.append(text.strip())

    header = build_master_header(entries)
    merged = "\n\n".join([header, *contents]) + "\n"
    output.write_text(merged, encoding="utf-8")
    return output


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Merge binder-level corpora into a single master corpus."
    )
    parser.add_argument("folder", help="Folder containing .txt corpora files.")
    parser.add_argument(
        "--output",
        default="MASTER_CORPUS__MERGED.txt",
        help="Output file name (default: MASTER_CORPUS__MERGED.txt).",
    )
    args = parser.parse_args()

    folder = Path(args.folder).expanduser().resolve()
    output = (folder / args.output).resolve()

    try:
        result = merge_corpora(folder, output)
    except Exception as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1

    print(f"Merged {folder} -> {result}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
