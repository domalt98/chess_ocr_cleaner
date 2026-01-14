#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""chess_ocr_cleaner_final.py

Unified chess OCR-to-text cleaner.

Primary goals
- Reduce token-heavy OCR exports while keeping prose.
- Replace obvious diagram/board rubble with a single [diagram] marker.
- Apply conservative, high-confidence OCR->notation conversions (move-context gated).

Optimization levels
  1 = regular
      - Unicode normalize (BOM/NBSP/soft hyphen, quotes/dashes, ligatures)
      - Collapse whitespace
      - Diagram squashing ([diagram])
      - High-confidence OCR->notation conversions (Level-1 rules)
  2 = aggressive
      - Level 1 +
      - Frequent header/footer removal (by repetition count)
      - Optional front matter drop (unless --keep-front)
      - Optional Index/Bibliography drop (unless --keep-index/--keep-bib)
      - Extra OCR->notation conversions (Level-2 rules)
  3 = final (default)
      - Level 2 +
      - More junk suppression (very low alpha, single-letter rubble)
      - Slightly stronger diagram/garbage detection
      - Extra OCR->notation conversions (Level-3 rules; still conservative)

Move pruning
  --drop-moves-to N
    N is the number of SAN-ish tokens to keep on move-heavy lines.
    - N = 0  -> drop move-heavy lines entirely
    - N > 0  -> keep ~N SAN-ish tokens then append ellipsis

Folder mode
  If INPUT is a directory, every .txt under it is cleaned.
    - add --recurse to include subfolders
  If --out is specified, outputs are written under that folder (mirroring subfolders
  when --recurse is used, to avoid name collisions).

Output naming
  <stem>_CLEAN_LVL_<level>[_DROP_MOVES_TO_<n>].txt

Examples
  py chess_ocr_cleaner_final.py "C:\\chess\\++OPENINGS.txt"
  py chess_ocr_cleaner_final.py "C:\\chess\\++OPENINGS.txt" --drop-moves-to 0
  py chess_ocr_cleaner_final.py "C:\\chess" --recurse --out "C:\\chess\\out" --drop-moves-to 12

"""

from __future__ import annotations

import argparse
import collections
import re
import sys
import unicodedata
from dataclasses import dataclass
from pathlib import Path
from typing import Counter, Iterator, List, Optional, Sequence, Set


# ------------------------------------------------------------
# Corpus markers / metadata (never remove)
# ------------------------------------------------------------
CORPUS_MARKERS = {
    "<<<BEGIN_CORPUS>>>",
    "<<<END_CORPUS>>>",
    "<<<BEGIN_TEXT>>>",
    "<<<END_TEXT>>>",
    "<<<END_CORPUS_HEADER>>>",
}
META_PREFIXES = ("TITLE:", "FILENAME:", "ORDER:", "SHA256:", "BYTES:")


def is_marker_or_meta(line: str) -> bool:
    s = line.strip()
    if not s:
        return False
    if s in CORPUS_MARKERS:
        return True
    if s.startswith(META_PREFIXES):
        return True
    # delimiter bars
    if len(s) >= 20 and set(s) <= set("=-"):
        return True
    return False


# ------------------------------------------------------------
# Unicode + whitespace
# ------------------------------------------------------------

def normalize_unicode(s: str) -> str:
    # Remove BOM
    s = s.replace("\ufeff", "")
    # Remove soft hyphen, NBSP
    s = s.replace("\u00ad", "")
    s = s.replace("\u00a0", " ")
    # Normalize (fold compatibility glyphs)
    s = unicodedata.normalize("NFKC", s)

    # Quotes
    s = (
        s.replace("“", '"')
        .replace("”", '"')
        .replace("„", '"')
        .replace("‘", "'")
        .replace("’", "'")
        .replace("‚", "'")
        .replace("«", '"')
        .replace("»", '"')
    )

    # Dashes/minus
    s = s.replace("—", "-").replace("–", "-").replace("−", "-").replace("\u2212", "-")

    # Ligatures
    s = s.replace("ﬁ", "fi").replace("ﬂ", "fl")

    return s


def collapse_spaces_keep_indent(line: str, max_indent: int = 8) -> str:
    """Collapse internal whitespace; keep up to max_indent leading spaces."""
    if line.strip() == "":
        return ""
    lead = len(line) - len(line.lstrip(" \t"))
    body = re.sub(r"[ \t]+", " ", line.strip())
    return (" " * min(lead, max_indent)) + body


# ------------------------------------------------------------
# Notation conversion (move-context gated)
# ------------------------------------------------------------

MOVE_BODY_RE = re.compile(r"(?:[a-h]|[1-8])?x?[a-h][1-8](?:=[QRBN])?[+#]?")

SAN_TOKEN_RE = re.compile(
    r"^(?:O-O-O|O-O|0-0-0|0-0|"
    r"[KQRBN]?[a-h]?[1-8]?x?[a-h][1-8](?:=[QRBN])?[+#]?|"
    r"[a-h]x?[a-h][1-8](?:=[QRBN])?[+#]?|"
    r"[a-h][1-8][+#]?)$"
)
MOVE_NUM_TOKEN_RE = re.compile(r"^\d+\.(?:\.\.)?$|^\d+\.\.\.$")


def normalize_castling(s: str) -> str:
    # long first
    s = re.sub(r"\b(?:0|O)[-\s]?(?:0|O)[-\s]?(?:0|O)\b", "O-O-O", s)
    s = re.sub(r"\b(?:0|O)[-\s]?(?:0|O)\b", "O-O", s)
    return s


def convert_capture_x(s: str) -> str:
    s = s.replace("×", "x")
    s = re.sub(r"\b([KQRBN])([a-h1-8]?)X(?=[a-h][1-8])", r"\1\2x", s)
    s = re.sub(r"\b([a-h])X(?=[a-h][1-8])", r"\1x", s)
    return s


@dataclass(frozen=True)
class ConversionRule:
    pattern: re.Pattern
    repl: str
    name: str


def _compile_rules() -> tuple[list[ConversionRule], list[ConversionRule], list[ConversionRule]]:
    """Return (lvl1, lvl2, lvl3) conversion rule lists."""

    lvl1: list[ConversionRule] = [
        ConversionRule(re.compile(r"®"), "Q", "® -> Q (queen)"),
        ConversionRule(re.compile(r"©"), "Q", "© -> Q (queen)"),
        ConversionRule(re.compile(r"§"), "R", "§ -> R (rook)"),

        # Knights
        ConversionRule(
            re.compile(r"(?<!\w)&(?=(?:[a-h]|[1-8])?x?[a-h][1-8])"),
            "N",
            "& -> N (knight figurine)",
        ),
        ConversionRule(
            re.compile(r"(?<!\w)£(?:[\|\?])?(?=(?:[a-h]|[1-8])?x?[a-h][1-8])"),
            "N",
            "£/£|/£? -> N (knight figurine)",
        ),
        ConversionRule(
            re.compile(r"(?<!\w)(?:€\)|2\))(?=(?:[a-h]|[1-8])?x?[a-h][1-8])"),
            "N",
            "€) / 2) -> N (knight figurine)",
        ),

        # Bishops
        ConversionRule(
            re.compile(r"(?<!\w)JL(?=(?:[a-h]|[1-8])?x?[a-h][1-8])"),
            "B",
            "JL -> B (bishop figurine)",
        ),
        ConversionRule(
            re.compile(r"(?<!\w)i(?=(?:[a-h]|[1-8])?x?[a-h][1-8])"),
            "B",
            "i -> B (bishop figurine)",
        ),
        ConversionRule(
            re.compile(r"(?<!\w)5\)(?=(?:[a-h]|[1-8])?x?[a-h][1-8])"),
            "B",
            "5) -> B (bishop figurine)",
        ),

        # Rooks
        ConversionRule(
            re.compile(r"(?<!\w)S\)(?=(?:[a-h]|[1-8])?x?[a-h][1-8])"),
            "R",
            "S) -> R (rook figurine)",
        ),
        ConversionRule(
            re.compile(r"(?<!\w)W(?=(?:[a-h]|[1-8])?x?[a-h][1-8])"),
            "R",
            "W -> R (rook OCR swap)",
        ),
    ]

    lvl2: list[ConversionRule] = [
        ConversionRule(
            re.compile(r"(?<!\w)I(?=(?:[a-h]|[1-8])?x?[a-h][1-8])"),
            "B",
            "I -> B (bishop OCR swap)",
        ),
        ConversionRule(
            re.compile(r"(?<!\w)E(?=(?:[a-h]|[1-8])?x?[a-h][1-8])"),
            "R",
            "E -> R (rook OCR swap)",
        ),
        ConversionRule(
            re.compile(r"(?<!\w)H(?=(?:[a-h]|[1-8])?x?[a-h][1-8])"),
            "N",
            "H -> N (knight OCR swap)",
        ),
    ]

    # Level 3 stays conservative.
    lvl3: list[ConversionRule] = [
        ConversionRule(
            re.compile(r"(?<!\w)£[)\]>lI](?=(?:[a-h]|[1-8])?x?[a-h][1-8])"),
            "N",
            "£) / £> / £l / £I -> N (knight variants)",
        ),
    ]

    return lvl1, lvl2, lvl3


LVL1_RULES, LVL2_RULES, LVL3_RULES = _compile_rules()


def apply_conversions(line: str, level: int) -> str:
    if not line or line.strip() == "":
        return line

    s = normalize_castling(line)
    s = convert_capture_x(s)

    for rule in LVL1_RULES:
        s = rule.pattern.sub(rule.repl, s)
    if level >= 2:
        for rule in LVL2_RULES:
            s = rule.pattern.sub(rule.repl, s)
    if level >= 3:
        for rule in LVL3_RULES:
            s = rule.pattern.sub(rule.repl, s)

    return s


# ------------------------------------------------------------
# Diagram / garbage detection
# ------------------------------------------------------------

PAGE_NUM_RE = re.compile(r"^\s*(?:page\s*)?\d{1,5}\s*$", re.IGNORECASE)
COORD_LINE_RE = re.compile(r"^\s*(?:a\s*b\s*c\s*d\s*e\s*f\s*g\s*h|abcdefgh)\s*$", re.IGNORECASE)
RANK_LINE_RE = re.compile(r"^\s*[1-8]\s+(?:[a-hA-H0-8\|\[\]\\/\-\+\*\._]{6,})\s*$")
BOARD_GLYPH_RE = re.compile(r"[\u25a0\u25a1\u25cb\u25cf\u2591\u2592\u2593]")
BOXDRAW_RE = re.compile(r"[\u2500-\u257f]")
DIAGRAM_CAPTION_RE = re.compile(r"\b(?:diagram|position|white\s+to\s+move|black\s+to\s+move|fen)\b", re.IGNORECASE)

SINGLE_LETTER_NOISE_RE = re.compile(
    r"^(?:[A-HI]|E|H|"
    r"A{1,12}|"
    r"(?:A\s+){1,12}A|"
    r"(?:AA|AAA|AAAA|AAAAA|AAAAAA|AAAAAAA|AAAAAAAA|AAAAAAAAA|AAAAAAAAAA)|"
    r"(?:A A|A A A|A A A A|AA AA|AAA AAA|A A A A A)|"
    r"(?:I I|H H|E E))$"
)


def alpha_ratio(s: str) -> float:
    if not s:
        return 1.0
    a = sum(ch.isalpha() for ch in s)
    return a / max(1, len(s))


def looks_like_diagram_line(s: str, level: int) -> bool:
    if not s:
        return False
    t = s.strip()
    if not t:
        return False

    if DIAGRAM_CAPTION_RE.search(t):
        return True
    if COORD_LINE_RE.match(t) or RANK_LINE_RE.match(t):
        return True
    if BOARD_GLYPH_RE.search(t) or BOXDRAW_RE.search(t):
        return True
    if re.search(r"[\|\+]{4,}", t) and re.search(r"[-_]{4,}", t):
        return True

    if level >= 3:
        non_ascii = sum(1 for ch in t if ord(ch) > 127)
        if non_ascii >= 6:
            return True

    return False


def is_move_dense_line(s: str) -> bool:
    s2 = re.sub(r"(\d+)\.(\.\.)?\s*([a-hKQRBNO0])", r"\1. \3", s)
    tokens = [t.strip("()[]{}.,;:") for t in s2.split()]
    if len(tokens) < 8:
        return False
    sanish = 0
    for t in tokens:
        if not t:
            continue
        if MOVE_NUM_TOKEN_RE.match(t):
            sanish += 1
        elif SAN_TOKEN_RE.match(t):
            sanish += 1
        elif re.match(r"^\d+\.[a-h][1-8]$", t):
            sanish += 1
    return (sanish / len(tokens)) >= 0.55


MOVE_LINE_START_RE = re.compile(r"^\d+\.(?:\.\.)?\s*\S")


def is_any_move_line(s: str) -> bool:
    if MOVE_LINE_START_RE.match(s):
        return True
    if is_move_dense_line(s):
        return True
    toks = [t.strip("()[]{}.,;:") for t in s.split()]
    sanish = sum(1 for t in toks if SAN_TOKEN_RE.match(t) or MOVE_NUM_TOKEN_RE.match(t))
    if len(toks) >= 6 and sanish >= 5:
        return True
    return False


def prune_move_line(s: str, keep_san_tokens: int) -> str:
    """Keep only the first keep_san_tokens SAN-ish tokens in a move-heavy line."""
    s = re.sub(r"\[[^\]]*\]", "", s)
    for _ in range(8):
        new = re.sub(r"\([^()]{0,500}\)", "", s)
        if new == s:
            break
        s = new
    s = collapse_spaces_keep_indent(s)

    s2 = re.sub(r"(\d+)\.(\.\.)?\s*([a-hKQRBNO0])", r"\1. \3", s)
    tokens = s2.split()

    kept: list[str] = []
    san_seen = 0
    for t in tokens:
        t2 = t.strip("()[]{}.,;:")
        if MOVE_NUM_TOKEN_RE.match(t2) or SAN_TOKEN_RE.match(t2) or re.match(r"^\d+\.[a-h][1-8]$", t2):
            san_seen += 1
        kept.append(t)
        if san_seen >= keep_san_tokens:
            break

    out = collapse_spaces_keep_indent(" ".join(kept))
    if out and out != s:
        out += " …"
    return out or s


# ------------------------------------------------------------
# Front/back matter pruning (per book block when markers exist)
# ------------------------------------------------------------

CONTENT_START_RE = re.compile(
    r"^(?:Introduction|Foreword|Preface|Prologue|"
    r"Chapter\s+\d+|Part\s+\d+|"
    r"\d+\.\s+)\b",
    re.IGNORECASE,
)
INDEX_START_RE = re.compile(r"^(Index|Index of|General Index)\b", re.IGNORECASE)
BIB_START_RE = re.compile(r"^(Bibliography|References|Further Reading)\b", re.IGNORECASE)


# ------------------------------------------------------------
# Header/footer frequency removal (global pass)
# ------------------------------------------------------------

def find_frequent_lines(lines: Sequence[str], min_count: int, len_min: int = 10, len_max: int = 90) -> Set[str]:
    ctr: Counter[str] = collections.Counter()
    for raw in lines:
        s = collapse_spaces_keep_indent(normalize_unicode(raw)).strip()
        if not s:
            continue
        if is_marker_or_meta(s):
            continue
        if PAGE_NUM_RE.match(s):
            continue
        if len(s) < len_min or len(s) > len_max:
            continue
        if DIAGRAM_CAPTION_RE.search(s):
            continue
        ctr[s] += 1
    return {s for s, c in ctr.items() if c >= min_count}


# ------------------------------------------------------------
# Main cleaning pipeline
# ------------------------------------------------------------

@dataclass
class Options:
    optimize: int = 3
    keep_front: bool = False
    keep_index: bool = False
    keep_bib: bool = False
    header_min_count: int = 20
    drop_moves_to: Optional[int] = None
    recurse: bool = False
    outdir: Optional[Path] = None


def clean_lines(lines: Sequence[str], opt: Options) -> List[str]:
    norm0 = [normalize_unicode(l.rstrip("\n\r")) for l in lines]

    frequent: Set[str] = set()
    if opt.optimize >= 2 and opt.header_min_count > 0:
        frequent = find_frequent_lines(norm0, min_count=opt.header_min_count)

    out: List[str] = []

    in_block = False
    dropping_front = False
    saw_content_start = False
    dropping_index = False
    dropping_bib = False

    def reset_book_state() -> None:
        nonlocal dropping_front, saw_content_start, dropping_index, dropping_bib
        dropping_front = (opt.optimize >= 2 and not opt.keep_front)
        saw_content_start = False
        dropping_index = False
        dropping_bib = False

    has_markers = any(l.strip() in CORPUS_MARKERS for l in norm0)
    if not has_markers:
        reset_book_state()
        in_block = True

    for raw in norm0:
        s = collapse_spaces_keep_indent(raw)

        if is_marker_or_meta(s):
            out.append(s.strip() if s.strip() else "")
            if s.strip() == "<<<BEGIN_TEXT>>>":
                in_block = True
                reset_book_state()
            elif s.strip() == "<<<END_TEXT>>>":
                in_block = False
            continue

        if opt.optimize >= 2 and PAGE_NUM_RE.match(s):
            continue

        if opt.optimize >= 2 and s.strip() in frequent:
            continue

        if in_block:
            if opt.optimize >= 2 and not opt.keep_index and not dropping_index and INDEX_START_RE.match(s.strip()):
                dropping_index = True
            if opt.optimize >= 2 and not opt.keep_bib and not dropping_bib and BIB_START_RE.match(s.strip()):
                dropping_bib = True
            if dropping_index or dropping_bib:
                continue

            if dropping_front and not saw_content_start:
                if s.strip() == "":
                    continue
                if CONTENT_START_RE.match(s.strip()):
                    saw_content_start = True
                else:
                    letters = [ch for ch in s if ch.isalpha()]
                    if letters:
                        upper_ratio = sum(ch.isupper() for ch in letters) / len(letters)
                        if upper_ratio > 0.9 and len(s) <= 70:
                            pass
                        else:
                            continue

        s = apply_conversions(s, level=opt.optimize)

        if looks_like_diagram_line(s, level=opt.optimize):
            if not out or out[-1].strip() != "[diagram]":
                out.append("[diagram]")
            continue

        if opt.optimize >= 3:
            t = s.strip()
            if t and SINGLE_LETTER_NOISE_RE.match(t):
                continue
            ar = alpha_ratio(t)
            if len(t) > 20 and ar < 0.16 and not is_any_move_line(t):
                continue

        if opt.drop_moves_to is not None:
            t = s.strip()
            if opt.drop_moves_to == 0:
                if is_any_move_line(t):
                    continue
            else:
                if is_any_move_line(t) or is_move_dense_line(t):
                    s = prune_move_line(s, keep_san_tokens=opt.drop_moves_to)

        out.append(s.rstrip())

    # De-dupe consecutive [diagram]
    dedup: List[str] = []
    for l in out:
        if l.strip() == "[diagram]" and dedup and dedup[-1].strip() == "[diagram]":
            continue
        dedup.append(l)

    # Collapse multiple blank lines to one
    final: List[str] = []
    blank_run = 0
    for l in dedup:
        if l.strip() == "":
            blank_run += 1
            if blank_run > 1:
                continue
            final.append("")
        else:
            blank_run = 0
            final.append(l.rstrip())

    while final and final[-1].strip() == "":
        final.pop()

    return final


# ------------------------------------------------------------
# IO helpers
# ------------------------------------------------------------

def compute_output_path(infile: Path, root_in: Optional[Path], opt: Options) -> Path:
    stem = infile.stem
    suffix = f"_CLEAN_LVL_{opt.optimize}"
    if opt.drop_moves_to is not None:
        suffix += f"_DROP_MOVES_TO_{opt.drop_moves_to}"
    out_name = stem + suffix + ".txt"

    if opt.outdir:
        if root_in and root_in.is_dir():
            rel = infile.relative_to(root_in)
            out_subdir = opt.outdir / rel.parent
            out_subdir.mkdir(parents=True, exist_ok=True)
            return out_subdir / out_name
        opt.outdir.mkdir(parents=True, exist_ok=True)
        return opt.outdir / out_name

    return infile.with_name(out_name)


def iter_txt_files(root: Path, recurse: bool) -> Iterator[Path]:
    if recurse:
        yield from root.rglob("*.txt")
    else:
        yield from root.glob("*.txt")


# ------------------------------------------------------------
# CLI
# ------------------------------------------------------------

def parse_args(argv: Sequence[str]) -> tuple[Options, Path]:
    ap = argparse.ArgumentParser(
        prog="chess_ocr_cleaner_final.py",
        description="Clean chess OCR .txt files (file or folder input).",
    )

    ap.add_argument("input", help="Input .txt file OR a folder containing .txt files")

    ap.add_argument(
        "--opt",
        "--optimize",
        type=int,
        choices=(1, 2, 3),
        default=3,
        help="Optimization level: 1=regular, 2=aggressive, 3=final (default: 3)",
    )

    ap.add_argument(
        "--out",
        default=None,
        help="Output directory. If provided, cleaned files are written under this folder.",
    )

    ap.add_argument(
        "--recurse",
        action="store_true",
        help="When INPUT is a folder, include subfolders (recursive).",
    )

    ap.add_argument(
        "--drop-moves-to",
        type=int,
        default=None,
        help="Move pruning: 0 drops move-heavy lines; N>0 keeps ~N SAN-ish tokens on move-heavy lines.",
    )

    ap.add_argument(
        "--header-min-count",
        type=int,
        default=20,
        help="Header/footer removal: drop lines repeated >= this count (level 2+). Use 0 to disable.",
    )

    ap.add_argument("--keep-front", action="store_true", help="Keep front matter (level 2+ normally drops it).")
    ap.add_argument("--keep-index", action="store_true", help="Keep Index sections (level 2+ normally drops them).")
    ap.add_argument("--keep-bib", action="store_true", help="Keep Bibliography/References sections (level 2+ normally drops them).")

    ns = ap.parse_args(argv)

    opt = Options(
        optimize=ns.opt,
        keep_front=ns.keep_front,
        keep_index=ns.keep_index,
        keep_bib=ns.keep_bib,
        header_min_count=ns.header_min_count,
        drop_moves_to=ns.drop_moves_to,
        recurse=ns.recurse,
        outdir=Path(ns.out).expanduser().resolve() if ns.out else None,
    )

    in_path = Path(ns.input).expanduser().resolve()
    return opt, in_path


def run(opt: Options, input_path: Path) -> int:
    if not input_path.exists():
        print(f"ERROR: input not found: {input_path}", file=sys.stderr)
        return 2

    files: List[Path]
    root_in: Optional[Path] = None

    if input_path.is_dir():
        root_in = input_path
        files = [p for p in iter_txt_files(input_path, recurse=opt.recurse) if p.is_file()]
        if not files:
            print(f"No .txt files found in: {input_path}", file=sys.stderr)
            return 1
    else:
        files = [input_path]

    for infile in files:
        try:
            with open(infile, "r", encoding="utf-8", errors="replace") as f:
                lines = f.readlines()
        except Exception as e:
            print(f"ERROR reading {infile}: {e}", file=sys.stderr)
            continue

        cleaned = clean_lines(lines, opt)
        out_path = compute_output_path(infile, root_in=root_in, opt=opt)

        try:
            with open(out_path, "w", encoding="utf-8", newline="\n") as f:
                f.write("\n".join(cleaned) + "\n")
        except Exception as e:
            print(f"ERROR writing {out_path}: {e}", file=sys.stderr)
            continue

        print(f"Cleaned: {infile}")
        print(f"   ->   {out_path}")
        print(f"   lines: {len(lines)} -> {len(cleaned)}")

    return 0


def main(argv: Optional[Sequence[str]] = None) -> int:
    opt, in_path = parse_args(sys.argv[1:] if argv is None else argv)
    return run(opt, in_path)


if __name__ == "__main__":
    raise SystemExit(main())
