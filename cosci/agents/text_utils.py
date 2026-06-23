"""Pure-text helpers for turning raw model prose into clean, atomic hypotheses.

Two jobs the generation/evolution agents both need:

* ``clean_title`` — derive a human-readable title from a hypothesis body,
  skipping conversational preambles ("I'll initiate this discourse…"),
  markdown emphasis, and label prefixes ("Proposed Hypothesis:", "Hypothesis 2:").
* ``split_atomic_hypotheses`` — when one generation response bundles several
  numbered proposals ("**Hypothesis 1: …**", "**Hypothesis 2: …**"), split it
  into one chunk per proposal so each becomes a first-class hypothesis that is
  reviewed, ranked, and cited on its own. A single-proposal response is left
  whole. This keeps the tournament comparing atomic mechanisms, not bundles.
"""
from __future__ import annotations

import re

# Conversational openers that are never a real title — skip the line entirely.
_PREAMBLE_RE = re.compile(
    r"^(?:"
    r"i['’]?(?:ll|m|d|ve)\b|i \b(?:will|am|would|propose|present|offer|suggest)\b|"
    r"let(?:'s| me| us)\b|here (?:is|are)\b|below\b|the following\b|allow me\b|"
    r"sure\b|certainly\b|okay\b|ok\b|of course\b|as (?:a |an |the |my )|"
    r"to (?:begin|start|address|propose)\b|first,|we (?:propose|present|begin|will)\b"
    r")",
    re.IGNORECASE,
)

# Label prefixes attached to a title, e.g. "Proposed Hypothesis:", "Hypothesis 2 -".
_LABEL_RE = re.compile(
    r"^(?:proposed|candidate|final|core|novel|primary|the)?\s*"
    r"(?:hypothesis|proposal|idea|mechanism|theory|title|claim)"
    r"\s*#?\s*\d*\s*[:.\)\-–—]\s*",
    re.IGNORECASE,
)

# A line that opens a distinct numbered proposal: optional markdown, a keyword,
# then a number — e.g. "**Hypothesis 1:**", "### Proposal 2", "Mechanism #3".
# Tied to the numeric "Hypothesis N" form models actually emit for debate prompts;
# an unlabeled or lettered list degrades to a single hypothesis (no false splits).
_PROPOSAL_MARKER_RE = re.compile(
    r"(?im)^[ \t]*[#>*•\-]*[ \t]*"
    r"(?:hypothesis|proposal|idea|theory|mechanism|candidate)"
    r"[ \t]*#?[ \t]*\d+\b"
)


def _strip_md(line: str) -> str:
    """Remove leading heading/bullet markers and all emphasis/code markers."""
    s = line.strip()
    s = re.sub(r"^#{1,6}\s*", "", s)          # markdown headings
    s = re.sub(r"^[>•\-]\s*", "", s)      # blockquote / dash-bullets
    s = re.sub(r"[*`]+", "", s)                # bold / italic / inline code
    return s.strip()


def _truncate(s: str, max_len: int) -> str:
    s = re.sub(r"\s+", " ", s).strip().strip(" .:;,–—-")
    if len(s) <= max_len:
        return s
    cut = s[:max_len].rsplit(" ", 1)[0].rstrip(" .:;,–—-")
    return (cut or s[:max_len]) + "…"


def clean_title(text: str, max_len: int = 80) -> str:
    """First line that reads like a title: markdown/label stripped, preambles skipped."""
    for raw in text.splitlines():
        line = _strip_md(raw)
        if not line:
            continue
        line = _LABEL_RE.sub("", line).strip()
        if not line or _PREAMBLE_RE.match(line):
            continue
        if len(line) < 3 or not re.search(r"[A-Za-z]", line):
            continue
        return _truncate(line, max_len)
    # Fallback: first non-empty line, lightly cleaned.
    first = next((_strip_md(l) for l in text.splitlines() if l.strip()), "")
    return _truncate(first, max_len) or "untitled hypothesis"


def split_atomic_hypotheses(text: str, min_len: int = 40) -> list[str]:
    """Split a response into one chunk per numbered proposal; else return it whole.

    Only splits when two or more ``Hypothesis N``-style markers are present, so a
    single hypothesis (or an internal "1./2./3." assumption chain) is never split.
    """
    text = text.strip()
    if not text:
        return [text]
    starts = [m.start() for m in _PROPOSAL_MARKER_RE.finditer(text)]
    if len(starts) < 2:
        return [text]
    segments = []
    for i, s in enumerate(starts):
        e = starts[i + 1] if i + 1 < len(starts) else len(text)
        seg = text[s:e].strip()
        if len(seg) >= min_len:
            segments.append(seg)
    return segments if len(segments) >= 2 else [text]
