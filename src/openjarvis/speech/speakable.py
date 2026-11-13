"""Reduce an assistant's Markdown reply to something a voice can read.

The CLI renders the reply as Markdown for the terminal, but voice mode used to
hand the same raw string to the TTS backend, which read the markup aloud.
Backends disagree about which markers they drop, and Kokoro's G2P is
context-dependent -- an isolated ``**42F**`` is silent, but ``*42F*`` in a
sentence becomes "forty two eff asterisk" -- so this cannot be left to them.

A plain-text transform, not a Markdown parse: the goal is a readable sentence,
not a faithful document. Anything unrecognized is read literally, which is the
behaviour this replaces.
"""

from __future__ import annotations

import re

__all__ = ["to_speakable_text"]

# --- Markdown ------------------------------------------------------------

# Drop whole code blocks, not just the fences: Kokoro turns ``` into a quote
# sound and then spells out the language name.
_FENCED_CODE = re.compile(
    r"^[ \t]*(?:```|~~~).*?(?:^[ \t]*(?:```|~~~)[ \t]*$|\Z)", re.DOTALL | re.MULTILINE
)
# Inline code is usually a real word in the sentence ("use `ls`").
_INLINE_CODE = re.compile(r"`+([^`]*)`+")
_IMAGE = re.compile(r"!\[([^\]]*)\]\([^)]*\)")
_LINK = re.compile(r"\[([^\]]*)\]\([^)]*\)")
# A URL read character by character is unusable.
_BARE_URL = re.compile(r"<?\bhttps?://[^\s<>)\]]+>?")
_HEADING = re.compile(r"^[ \t]*#{1,6}[ \t]*(.*?)[ \t]*#*[ \t]*$", re.MULTILINE)
# Thematic breaks and setext underlines.
_RULE = re.compile(r"^[ \t]*(?:[-*_][ \t]*){3,}$|^[ \t]*=+[ \t]*$", re.MULTILINE)
_TABLE_SEP = re.compile(
    r"^[ \t]*\|?[ \t]*:?-{2,}:?[ \t]*(?:\|[ \t]*:?-{2,}:?[ \t]*)*\|?[ \t]*$",
    re.MULTILINE,
)
# Outer pipes are a border; stripping them first stops a spoken row opening
# and closing on a comma.
_TABLE_EDGE = re.compile(r"^[ \t]*\||\|[ \t]*$", re.MULTILINE)
_TABLE_PIPE = re.compile(r"[ \t]*\|[ \t]*")
# Only unordered markers. "1." is kept -- the number means something when heard.
_BULLET = re.compile(r"^[ \t]*[-*+][ \t]+", re.MULTILINE)
_BLOCKQUOTE = re.compile(r"^[ \t]*>[ \t]?", re.MULTILINE)
_STRIKETHROUGH = re.compile(r"~~(.+?)~~", re.DOTALL)
# Includes the unmatched leftovers models emit. A literal "*" in prose is rare
# enough that removing all of them is the safer trade.
_ASTERISK = re.compile(r"\*+")
# Only when it wraps a word, so snake_case and file_name.txt survive.
_UNDERSCORE = re.compile(r"(?<![\w\\])_{1,3}([^_\n]+?)_{1,3}(?!\w)")
# Kokoro pronounces these ("🌤️" -> "sun"), which is rarely what was meant.
_EMOJI = re.compile("[\U0001f000-\U0001faff←-⇿⌀-➿⬀-⯿︎️‍]+")
_BLANK_LINES = re.compile(r"\n{3,}")
_TRAILING_SPACE = re.compile(r"[ \t]+$", re.MULTILINE)
_SPACES = re.compile(r"[ \t]{2,}")


# --- Units ----------------------------------------------------------------
#
# Stripping markup leaves symbols the backend reads literally: "42°F" becomes
# "forty two degrees eff", "10 mph" becomes "ten em pee aych". Only what
# measures wrong is rewritten -- "72°", "50%" and "&" are already correct.

# The scale letter is consumed, giving "42 degrees Fahrenheit".
_DEGREES = re.compile(r"\s*°\s*([FCK])\b")
_DEGREE_SCALE = {"F": "Fahrenheit", "C": "Celsius", "K": "Kelvin"}

# "in" and "ft" are excluded: they collide with ordinary words.
_UNIT_WORDS = {
    r"mph": "miles per hour",
    r"km/h": "kilometers per hour",
    r"kph": "kilometers per hour",
    r"m/s": "meters per second",
    r"kt": "knots",
    r"hPa": "hectopascals",
    r"mb": "millibars",
}
_UNITS = re.compile(
    r"(?<![\w/])(" + "|".join(re.escape(u) for u in _UNIT_WORDS) + r")(?![\w/])"
)

_COMPASS_PARTS = {"N": "north", "S": "south", "E": "east", "W": "west"}
_COMPASS = re.compile(r"\b([NS]?[NSEW][EW]?)\b")

# The backend drops a spaced hyphen between numbers entirely, so "42°F - 57°F"
# reads as two unrelated figures. Group 1 keeps the unit on the left. A bare
# hyphen needs the spacing, which leaves "2026-11-13" and "555-1234" alone.
_RANGE = re.compile(r"(?<=\d)(\s*°\s*[FCK]\b)?(?:\s+-\s+|\s*[–—]\s*)(?=\d)")


def _expand_compass(match: re.Match[str]) -> str:
    # A lone N/S/E/W is more often an initial than a bearing.
    token = match.group(1)
    if len(token) < 2 or any(c not in _COMPASS_PARTS for c in token):
        return token
    return "".join(_COMPASS_PARTS[c] for c in token)


def _expand_units(text: str) -> str:
    # Ranges first: expanding "°F" to a word would leave the range pattern
    # with nothing to anchor on.
    text = _RANGE.sub(lambda m: f"{m.group(1) or ''} to ", text)
    text = _DEGREES.sub(lambda m: f" degrees {_DEGREE_SCALE[m.group(1)]}", text)
    text = _UNITS.sub(lambda m: _UNIT_WORDS[m.group(1)], text)
    return _COMPASS.sub(_expand_compass, text)


def to_speakable_text(text: str) -> str:
    """Return *text* with Markdown and symbols reduced to spoken words.

    Lossy by design: anything that only means something on screen (fences,
    rules, table borders, emphasis, emoji, URLs) is dropped, and symbols a
    backend spells out letter by letter are expanded. Returns ``""`` when the
    input was nothing but markup, which callers should treat as nothing to say.
    """
    if not text:
        return ""

    # Code first, so its contents are never treated as markup.
    text = _FENCED_CODE.sub(" ", text)
    text = _INLINE_CODE.sub(r"\1", text)
    text = _IMAGE.sub(r"\1", text)
    text = _LINK.sub(r"\1", text)
    text = _BARE_URL.sub(" ", text)

    # Terminate headings so the voice pauses instead of running them into the
    # paragraph below.
    text = _HEADING.sub(
        lambda m: f"{m.group(1).rstrip('.:')}." if m.group(1) else "", text
    )

    text = _TABLE_SEP.sub(" ", text)
    text = _RULE.sub(" ", text)
    text = _BLOCKQUOTE.sub("", text)
    text = _BULLET.sub("", text)
    text = _TABLE_EDGE.sub("", text)
    text = _TABLE_PIPE.sub(", ", text)

    text = _STRIKETHROUGH.sub(r"\1", text)
    text = _UNDERSCORE.sub(r"\1", text)
    text = _ASTERISK.sub("", text)
    text = _EMOJI.sub(" ", text)

    text = _expand_units(text)

    text = _TRAILING_SPACE.sub("", text)
    text = _SPACES.sub(" ", text)
    text = _BLANK_LINES.sub("\n\n", text)
    # Drop lines left holding only punctuation, e.g. an emptied table row.
    lines = [line for line in text.split("\n") if line.strip(" ,.")]
    return "\n".join(lines).strip()
