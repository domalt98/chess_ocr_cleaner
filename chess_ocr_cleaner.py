#!/usr/bin/env python3
"""
Chess OCR Cleaner (Unified)

Goal: aggressively reduce OCR noise while preserving prose.
- Default optimize level: 3
- Input can be a single .txt file OR a folder of .txt files.
- Folder mode supports --recurse and --out (dump all outputs into OUTDIR, mirroring subfolders to avoid collisions).
- Move-dump pruning can be controlled with --drop-moves-to N:
    * 0 => drop move-heavy lines entirely
    * N>0 => keep roughly N SAN-ish tokens from move-heavy lines, then add an ellipsis

Optimization levels:
  1 = regular (safe)
  2 = aggressive (headers/footers + front/index/bib pruning + moderate move pruning)
  3 = final (stronger junk/diagram suppression + stronger default move pruning)

Note: piece/notation conversions are applied primarily in "move-like" tokens to avoid corrupting prose.
"""
from __future__ import annotations

import argparse
import os
import re
import sys
from collections import Counter
from pathlib import Path

# -----------------------------
# Unicode normalization helpers
# -----------------------------
SOFT_HYPHEN = "\u00ad"
NBSP = "\u00a0"

TRANSLATION_MAP = str.maketrans({
    NBSP: " ",
    "\u2018": "'", "\u2019": "'",
    "\u201c": '"', "\u201d": '"',
    "\u2013": "-", "\u2014": "-", "\u2212": "-",  # en/em/minus
    "\ufb01": "fi", "\ufb02": "fl",
})

def normalize_text(s: str) -> str:
    s = s.replace(SOFT_HYPHEN, "")
    s = s.translate(TRANSLATION_MAP)
    # normalize weird whitespace
    s = s.replace("\r\n", "\n").replace("\r", "\n")
    s = re.sub(r"[ \t]+", " ", s)
    return s

# -----------------------------
# Move / SAN detection
# -----------------------------
SAN_TOKEN_RE = re.compile(
    r"""^(?:
        O-O-O|O-O|
        [KQRBN][a-h]?[1-8]?x?[a-h][1-8](?:=[QRBN])?[+#]?|
        [a-h]x[a-h][1-8](?:=[QRBN])?[+#]?|
        [a-h][1-8](?:=[QRBN])?[+#]?|
        [a-h][1-8]e\.p\.|
        [a-h][1-8]\+|
        \d+\.(?:\.\.)?
    )$""",
    re.VERBOSE
)

SQUARE_RE = re.compile(r"[a-h][1-8]")
MOVE_NUM_RE = re.compile(r"\b\d{1,3}\.(?:\.\.)?\b")

def token_is_move_like(tok: str) -> bool:
    t = tok.strip(" ,;:()[]{}<>\"'")
    if not t:
        return False
    if t in ("O-O", "O-O-O"):
        return True
    if SAN_TOKEN_RE.match(t):
        return True
    # OCR-mangled but still looks like a move destination
    if SQUARE_RE.search(t):
        return True
    return False

def count_san_like(tokens: list[str]) -> int:
    c = 0
    for tok in tokens:
        t = tok.strip(" ,;:()[]{}<>\"'")
        if not t:
            continue
        if t in ("O-O", "O-O-O"):
            c += 1
            continue
        if SAN_TOKEN_RE.match(t):
            c += 1
            continue
        # looser: coordinate + optional capture/promo/check
        if re.fullmatch(r"[KQRBN]?[a-h]?x?[a-h][1-8](?:=[QRBN])?[+#]?", t):
            c += 1
            continue
    return c

def is_move_dump_line(line: str) -> bool:
    s = line.strip()
    if len(s) < 140:
        return False

    tokens = s.split()
    san_count = count_san_like(tokens)
    move_nums = len(MOVE_NUM_RE.findall(s))
    squares = len(SQUARE_RE.findall(s))

    # Move dumps usually have lots of SAN-ish tokens, move numbers, and/or many squares.
    if san_count >= 14:
        return True
    if move_nums >= 6 and squares >= 16:
        return True
    if squares >= 28 and len(tokens) >= 20:
        return True
    return False

def prune_move_dump(line: str, keep_tokens: int) -> str | None:
    """keep_tokens=0 => drop line entirely; otherwise keep first N move-like tokens (plus any leading move-number token)."""
    if keep_tokens <= 0:
        return None
    s = line.strip()
    tokens = s.split()

    kept: list[str] = []
    count = 0

    # Preserve a leading move number token like "12." or "12..."
    if tokens:
        lead = tokens[0].strip()
        if re.fullmatch(r"\d{1,3}\.(?:\.\.)?", lead):
            kept.append(tokens[0])
            tokens = tokens[1:]

    for tok in tokens:
        if token_is_move_like(tok):
            kept.append(tok)
            count += 1
            if count >= keep_tokens:
                break
        else:
            # keep some separators like "..." if they appear early
            if tok in ("...", "…") and count < keep_tokens:
                kept.append(tok)

    if not kept:
        return None
    return " ".join(kept) + " …"

# -----------------------------
# Diagram / garbage detection
# -----------------------------
BOARD_COORD_RE = re.compile(r"\b(a\s*b\s*c\s*d\s*e\s*f\s*g\s*h|abcdefgh)\b", re.IGNORECASE)
RANKS_RE = re.compile(r"^\s*[8-1]\s*(?:[^\w\s]\s*){4,}")
BOX_CHARS_RE = re.compile(r"[■□▪▫▢▣◆◇♔♕♖♗♘♙♚♛♜♝♞♟]")

def is_diagramish_line(line: str) -> bool:
    s = line.strip()
    if not s:
        return False

    if BOARD_COORD_RE.search(s):
        return True
    if BOX_CHARS_RE.search(s):
        return True
    if RANKS_RE.match(s):
        return True

    # common OCR "tm tm tm" style rubble
    if len(s) > 40 and re.search(r"(?:\btm\b\s*){6,}", s, re.IGNORECASE):
        return True

    # Symbol/garbage heuristics
    total = len(s)
    alpha = sum(ch.isalpha() for ch in s)
    digit = sum(ch.isdigit() for ch in s)
    space = sum(ch.isspace() for ch in s)
    non_ascii = sum(ord(ch) > 127 for ch in s)
    symbols = total - alpha - digit - space

    alpha_ratio = alpha / total
    sym_ratio = symbols / total
    non_ascii_ratio = non_ascii / total

    # If it's not move-heavy, and it's mostly symbols/non-ascii, treat as diagram/junk.
    if not is_move_dump_line(s):
        if alpha_ratio < 0.22 and (sym_ratio > 0.35 or non_ascii_ratio > 0.12):
            return True

    # Very long lines with tiny alphabetic content are usually scanned boards
    if total > 120 and alpha_ratio < 0.18 and sym_ratio > 0.25:
        return True

    return False

# -----------------------------
# Notation conversions per level
# -----------------------------
# Castling normalization on whole line (safe)
CASTLING_FIXES = [
    (re.compile(r"\b0-0-0\b"), "O-O-O"),
    (re.compile(r"\bO-0-0\b", re.IGNORECASE), "O-O-O"),
    (re.compile(r"\b0-0\b"), "O-O"),
    (re.compile(r"\bO-0\b", re.IGNORECASE), "O-O"),
    (re.compile(r"\b00\b"), "O-O"),
    (re.compile(r"\b000\b"), "O-O-O"),
]

# Move-token-only replacements (gated)
def apply_token_conversions(tok: str, level: int) -> str:
    raw = tok
    t = tok.strip(" \t")
    # only operate on tokens that look like they contain a move square or castling
    if not token_is_move_like(t):
        return raw

    # Normalize multiplication sign to capture
    t = t.replace("×", "x")

    # £-family: ambiguous. Use guarded rules:
    # - "£6" is almost certainly "f6" (pawn destination without file letter)
    # - "£f3"/"£xe5" etc likely a piece letter OCR -> treat as N in that context
    # - keep it conservative by only converting when pattern matches.
    t = re.sub(r"(^|\b)(£)([1-8])\b", r"\1f\3", t)
    t = re.sub(r"(^|\b)(£[\|\?]?)(?=[a-h]x?[a-h][1-8]|x[a-h][1-8]|[a-h][1-8])", r"\1", t)  # no-op for grouping
    t = re.sub(r"(^|\b)£[\|\?]?(?=[a-h]x?[a-h][1-8]|x[a-h][1-8]|[a-h][1-8])", "N", t)

    # Level 1 core figurine-ish OCR
    t = t.replace("®", "Q").replace("©", "Q")
    # Rook
    t = t.replace("§", "R")
    # Some OCR outputs "S)" as a rook-like glyph
    t = t.replace("S)", "R")
    # Bishop
    t = t.replace("JL", "B")
    # Only change lowercase i when followed by square/capture-ish pattern
    t = re.sub(r"(^|\b)i(?=[a-h]x?[a-h][1-8]|x[a-h][1-8]|[a-h][1-8])", "B", t)
    t = t.replace("5)", "B")

    # Knight
    # '&' used in figurines sometimes
    t = re.sub(r"(^|\b)&(?=[a-h]x?[a-h][1-8]|x[a-h][1-8]|[a-h][1-8])", "N", t)
    t = t.replace("€)", "N").replace("2)", "N")

    # Misc capture X -> x in move context
    t = re.sub(r"(?<=\w)X(?=[a-h][1-8])", "x", t)
    t = re.sub(r"(?<=\w)X(?=x?[a-h][1-8])", "x", t)

    if level >= 2:
        # Rook: E is a common OCR confusion, but only in move context
        t = re.sub(r"(^|\b)E(?=[a-h]x?[a-h][1-8]|x[a-h][1-8]|[a-h][1-8])", "R", t)
        # Knight: H sometimes appears for N
        t = re.sub(r"(^|\b)H(?=[a-h]x?[a-h][1-8]|x[a-h][1-8]|[a-h][1-8])", "N", t)
        # Bishop: I sometimes appears for B
        t = re.sub(r"(^|\b)I(?=[a-h]x?[a-h][1-8]|x[a-h][1-8]|[a-h][1-8])", "B", t)
        # W -> R (rare but seen)
        t = re.sub(r"(^|\b)W(?=[a-h][1-8])", "R", t)

    if level >= 3:
        # Extra £ variants (still gated via token_is_move_like)
        t = re.sub(r"(^|\b)£[)\]>lI](?=[a-h]x?[a-h][1-8]|x[a-h][1-8]|[a-h][1-8])", "N", t)

    return t

def apply_line_conversions(line: str, level: int) -> str:
    s = line
    for rx, rep in CASTLING_FIXES:
        s = rx.sub(rep, s)

    # Token-wise conversions for move-like tokens
    tokens = s.split(" ")
    for i, tok in enumerate(tokens):
        if tok and any(ch in tok for ch in ("®", "©", "§", "&", "£", "€", "2", "5", "JL", "S)", "E", "H", "I", "W", "×", "X")):
            tokens[i] = apply_token_conversions(tok, level)
    return " ".join(tokens)

# -----------------------------
# Front-matter / index / bib logic
# -----------------------------
CONTENT_START_RE = re.compile(
    r"""^\s*(?:
        introduction|preface|foreword|prologue|epilogue|
        chapter\b|part\b|book\b|section\b|lesson\b|
        \d{1,3}\.?\s+[A-Za-z]|
        [IVXLC]{1,8}\.?\s+[A-Za-z]
    )""",
    re.IGNORECASE | re.VERBOSE
)

def looks_like_standalone_heading(line: str) -> bool:
    s = line.strip()
    return bool(re.fullmatch(r"[A-Za-z][A-Za-z \t]{2,60}", s)) and s.upper() == s

def is_full_line_heading(line: str, word: str) -> bool:
    return bool(re.fullmatch(rf"\s*{re.escape(word)}\s*", line, re.IGNORECASE))

# -----------------------------
# Header/footer removal
# -----------------------------
def normalize_header_key(line: str) -> str:
    s = line.strip()
    s = re.sub(r"\b\d+\b", "", s)  # remove page numbers
    s = re.sub(r"\s+", " ", s)
    return s.lower().strip()

def remove_repeated_headers(lines: list[str], min_count: int) -> list[str]:
    if min_count <= 0:
        return lines

    keys = [normalize_header_key(ln) for ln in lines if ln.strip()]
    freq = Counter(k for k in keys if 3 <= len(k) <= 120)
    kill = {k for k, c in freq.items() if c >= min_count}

    out: list[str] = []
    for ln in lines:
        k = normalize_header_key(ln)
        if k in kill:
            # Don't remove if it is clearly a move line (safety)
            if is_move_dump_line(ln) or count_san_like(ln.split()) >= 6:
                out.append(ln)
                continue
            continue
        out.append(ln)
    return out

# -----------------------------
# Main cleaning pipeline
# -----------------------------
def clean_lines(raw_text: str, level: int,
                drop_moves_to: int | None,
                header_min_count: int,
                keep_front: bool,
                keep_index: bool,
                keep_bib: bool) -> str:
    text = normalize_text(raw_text)
    lines = text.split("\n")

    # line-by-line normalize (preserve blank lines for later collapse)
    lines = [ln.rstrip() for ln in lines]

    # Remove null bytes / weird controls
    lines = [ln.replace("\x00", "") for ln in lines]

    # Remove repeated headers/footers early for better downstream stats (level 2+)
    if level >= 2:
        lines = remove_repeated_headers(lines, header_min_count)

    # Front matter dropping (level 2+) with failsafe to prevent catastrophic deletion
    if level >= 2 and not keep_front:
        out: list[str] = []
        saw_content = False
        skipped_nonempty = 0
        MAX_FRONT_SKIP = 200  # anti-nuke
        for ln in lines:
            s = ln.strip()
            if not saw_content:
                if s:
                    skipped_nonempty += 1
                    if CONTENT_START_RE.match(s):
                        saw_content = True
                    elif skipped_nonempty >= MAX_FRONT_SKIP:
                        saw_content = True

                if not saw_content:
                    # keep only short ALL-CAPS headings, discard the rest
                    letters = [c for c in s if c.isalpha()]
                    upper_ratio = (sum(c.isupper() for c in letters) / len(letters)) if letters else 0.0
                    if upper_ratio > 0.90 and len(s) <= 70 and len(letters) >= 6:
                        out.append(ln)
                    continue

            out.append(ln)
        lines = out

    # Diagram/junk suppression with [diagram] marker collapse
    out2: list[str] = []
    in_diag = False
    for ln in lines:
        if is_diagramish_line(ln):
            if not in_diag:
                out2.append("[diagram]")
                in_diag = True
            continue
        else:
            in_diag = False
            out2.append(ln)
    lines = out2

    # Apply conversions + move pruning
    if drop_moves_to is None:
        # default per level
        if level == 1:
            default_keep = None
        elif level == 2:
            default_keep = 32
        else:
            default_keep = 16
    else:
        default_keep = drop_moves_to

    out3: list[str] = []
    for ln in lines:
        s = ln.strip()
        if not s:
            out3.append("")
            continue

        # prune move dumps (level 2+ or explicit)
        if default_keep is not None and is_move_dump_line(ln):
            pruned = prune_move_dump(ln, default_keep)
            if pruned is None:
                continue
            ln = pruned

        # line conversions
        ln = apply_line_conversions(ln, level)
        out3.append(ln)
    lines = out3

    # Index/Bibliography truncation (level 2+) - safer:
    # only trigger on full-line heading and only if it appears in the late part of the book.
    if level >= 2:
        N = len(lines)
        cut_at = None
        for idx, ln in enumerate(lines):
            s = ln.strip()
            if not s:
                continue
            late = idx > int(0.60 * max(1, N))
            if late:
                if not keep_index and is_full_line_heading(s, "index"):
                    cut_at = idx
                    break
                if not keep_bib and (is_full_line_heading(s, "bibliography") or is_full_line_heading(s, "references")):
                    cut_at = idx
                    break
        if cut_at is not None:
            lines = lines[:cut_at]

    # Level 3: stronger garbage pruning (but avoid deleting prose)
    if level >= 3:
        cleaned: list[str] = []
        for ln in lines:
            s = ln.strip()
            if not s:
                cleaned.append("")
                continue

            total = len(s)
            alpha = sum(ch.isalpha() for ch in s)
            digit = sum(ch.isdigit() for ch in s)
            space = sum(ch.isspace() for ch in s)
            non_ascii = sum(ord(ch) > 127 for ch in s)
            symbols = total - alpha - digit - space

            alpha_ratio = alpha / total
            sym_ratio = symbols / total
            non_ascii_ratio = non_ascii / total

            # Drop lines that are extremely non-prose and not move-like
            if not token_is_move_like(s) and not is_move_dump_line(s):
                if alpha_ratio < 0.15 and (sym_ratio > 0.35 or non_ascii_ratio > 0.15) and total > 40:
                    continue
            cleaned.append(ln)
        lines = cleaned

    # Collapse excessive blank lines (keep at most 1 consecutive blank)
    final_lines: list[str] = []
    blank_run = 0
    for ln in lines:
        if ln.strip() == "":
            blank_run += 1
            if blank_run <= 1:
                final_lines.append("")
        else:
            blank_run = 0
            final_lines.append(ln)

    return "\n".join(final_lines).strip() + "\n"

# -----------------------------
# IO / path handling
# -----------------------------
def read_text_file(path: Path) -> str:
    # try utf-8-sig first, then plain utf-8, then latin-1 as last resort
    for enc in ("utf-8-sig", "utf-8", "cp1252", "latin-1"):
        try:
            return path.read_text(encoding=enc, errors="strict")
        except Exception:
            continue
    # last resort: replace errors
    return path.read_text(encoding="utf-8", errors="replace")

def build_output_path(infile: Path, outdir: Path | None, optimize: int, drop_moves_to: int | None, explicit_drop_flag: bool, root_in: Path | None = None) -> Path:
    suffix = f"_CLEAN_LVL_{optimize}"
    if explicit_drop_flag:
        # reflect explicit drop-moves-to usage only
        dm = 0 if drop_moves_to is None else drop_moves_to
        suffix += f"_DROP_MOVES_TO_{dm}"

    out_name = infile.stem + suffix + infile.suffix

    if outdir is None:
        return infile.with_name(out_name)

    outdir.mkdir(parents=True, exist_ok=True)

    # Mirror subfolders if root_in provided and infile is under it
    if root_in is not None:
        try:
            rel = infile.relative_to(root_in)
            rel_parent = rel.parent
            target_dir = outdir / rel_parent
            target_dir.mkdir(parents=True, exist_ok=True)
            return target_dir / out_name
        except Exception:
            pass

    return outdir / out_name

def iter_txt_files(folder: Path, recurse: bool) -> list[Path]:
    if recurse:
        return sorted([p for p in folder.rglob("*.txt") if p.is_file()])
    return sorted([p for p in folder.glob("*.txt") if p.is_file()])

def main() -> int:
    ap = argparse.ArgumentParser(prog="chess_ocr_cleaner_final.py")
    ap.add_argument("input", help="Input .txt file OR a folder containing .txt files")
    ap.add_argument("--opt", "--optimize", dest="optimize", type=int, choices=(1,2,3), default=3,
                    help="Optimization level (1=regular,2=aggressive,3=final). Default=3.")
    ap.add_argument("--out", dest="out", default=None,
                    help="Output directory. If set, cleaned files are written into this folder (subfolders mirrored when recursing).")
    ap.add_argument("--recurse", action="store_true",
                    help="If input is a folder, process .txt files in all subfolders as well.")
    ap.add_argument("--drop-moves-to", dest="drop_moves_to", type=int, default=None,
                    help="Move pruning: 0 drops move-heavy lines entirely; N>0 keeps ~N move tokens on move-heavy lines.")
    ap.add_argument("--header-min-count", dest="header_min_count", type=int, default=5,
                    help="For opt>=2: lines repeated >= N times are treated as headers/footers and removed. Set 0 to disable.")
    ap.add_argument("--keep-front", action="store_true",
                    help="For opt>=2: keep front matter (otherwise attempts to remove TOC/preface with failsafe).")
    ap.add_argument("--keep-index", action="store_true",
                    help="For opt>=2: keep index (otherwise may truncate from an INDEX heading near end).")
    ap.add_argument("--keep-bib", action="store_true",
                    help="For opt>=2: keep bibliography/references (otherwise may truncate near end).")

    args = ap.parse_args()

    in_path = Path(args.input).expanduser()
    outdir = Path(args.out).expanduser() if args.out else None

    # Determine whether user explicitly set drop-moves-to (for naming convention)
    explicit_drop_flag = ("--drop-moves-to" in sys.argv)

    if in_path.is_dir():
        files = iter_txt_files(in_path, args.recurse)
        if not files:
            print(f"[warn] No .txt files found in folder: {in_path}")
            return 0

        for fp in files:
            try:
                raw = read_text_file(fp)
                cleaned = clean_lines(
                    raw_text=raw,
                    level=args.optimize,
                    drop_moves_to=args.drop_moves_to,
                    header_min_count=args.header_min_count,
                    keep_front=args.keep_front,
                    keep_index=args.keep_index,
                    keep_bib=args.keep_bib,
                )
                out_path = build_output_path(fp, outdir, args.optimize, args.drop_moves_to, explicit_drop_flag, root_in=in_path)
                out_path.write_text(cleaned, encoding="utf-8", errors="strict")
            except Exception as e:
                print(f"[error] Failed: {fp} :: {e}")
        return 0

    if in_path.is_file():
        raw = read_text_file(in_path)
        cleaned = clean_lines(
            raw_text=raw,
            level=args.optimize,
            drop_moves_to=args.drop_moves_to,
            header_min_count=args.header_min_count,
            keep_front=args.keep_front,
            keep_index=args.keep_index,
            keep_bib=args.keep_bib,
        )
        out_path = build_output_path(in_path, outdir, args.optimize, args.drop_moves_to, explicit_drop_flag)
        out_path.write_text(cleaned, encoding="utf-8", errors="strict")
        return 0

    print(f"[error] Input path does not exist: {in_path}")
    return 2

if __name__ == "__main__":
    raise SystemExit(main())
