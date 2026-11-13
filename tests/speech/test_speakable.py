"""Markdown -> speakable prose.

Models reply in Markdown and voice mode used to hand that straight to the TTS
backend, which read the markup aloud. Backends disagree about which markers
they drop, so these pin the normalization rather than any one backend's
behaviour.
"""

from __future__ import annotations

import pytest

from openjarvis.speech.speakable import to_speakable_text


class TestEmphasis:
    """The reported bug: "*42F*" was heard as "forty two eff asterisk"."""

    @pytest.mark.parametrize(
        "markdown,expected",
        [
            ("*42F*", "42F"),
            ("**42F**", "42F"),
            ("**Temperature:** 42F", "Temperature: 42F"),
            ("**42F** and *cold*", "42F and cold"),
            ("~~old~~ new", "old new"),
            ("_emphasis_ here", "emphasis here"),
            ("__strong__ here", "strong here"),
            # Models emit unmatched markers too; they must not survive either.
            ("**unclosed and *mixed", "unclosed and mixed"),
        ],
    )
    def test_markers_removed(self, markdown, expected):
        assert to_speakable_text(markdown) == expected

    def test_underscores_inside_words_survive(self):
        """snake_case and filenames are words, not emphasis."""
        text = "a snake_case name and file_name.txt stay"
        assert to_speakable_text(text) == text


class TestBlockStructure:
    def test_heading_markers_removed_and_sentence_terminated(self):
        # "#### Heading" was spoken as "hash hash hash heading".
        assert to_speakable_text("#### Heading") == "Heading."

    def test_heading_keeps_existing_terminator(self):
        assert to_speakable_text("## Already done.") == "Already done."

    def test_bullets_and_blockquotes_lose_their_markers(self):
        assert to_speakable_text("- item one") == "item one"
        assert to_speakable_text("> quoted line") == "quoted line"

    def test_ordered_list_numbers_are_kept(self):
        """The number carries meaning when heard; the bullet glyph does not."""
        assert to_speakable_text("1. first\n2. second") == "1. first\n2. second"

    def test_thematic_break_says_nothing(self):
        assert to_speakable_text("---") == ""

    def test_table_reads_as_comma_separated_cells(self):
        table = "| City | Temp |\n|------|------|\n| Crossville | 42F |"
        spoken = to_speakable_text(table)
        assert "|" not in spoken
        assert "---" not in spoken
        assert "City, Temp" in spoken
        assert "Crossville, 42F" in spoken
        # No row should open or close on a comma.
        for line in spoken.splitlines():
            assert not line.strip().startswith(",")
            assert not line.strip().endswith(",")


class TestCodeAndLinks:
    def test_fenced_code_is_dropped_entirely(self):
        """Kokoro renders ``` as a quote sound and spells the language."""
        text = "Before\n\n```python\nx = 1\n```\n\nAfter"
        spoken = to_speakable_text(text)
        assert "Before" in spoken and "After" in spoken
        assert "python" not in spoken
        assert "x = 1" not in spoken

    def test_inline_code_keeps_its_words(self):
        assert to_speakable_text("Use `ls -la` to list") == "Use ls -la to list"

    def test_link_says_the_label_not_the_url(self):
        text = "See [the forecast](https://weather.com/x?a=1) now"
        assert to_speakable_text(text) == "See the forecast now"

    def test_bare_url_is_dropped(self):
        spoken = to_speakable_text("Read https://example.com/a/b?c=1 today")
        assert "http" not in spoken
        assert "example.com" not in spoken
        assert "Read" in spoken and "today" in spoken

    def test_image_contributes_only_alt_text(self):
        assert to_speakable_text("![a chart](http://x/y.png)") == "a chart"


class TestEmoji:
    def test_emoji_removed(self):
        # "### 🌤️ Today" was spoken as "hash sun today".
        assert to_speakable_text("### 🌤️ **Today**") == "Today."

    def test_emoji_between_words_leaves_clean_spacing(self):
        assert to_speakable_text("warm 🌤️ today") == "warm today"


class TestWholeReplies:
    def test_reported_weather_reply(self):
        reply = (
            "### 🌤️ **Today (Friday, November 13)**\n"
            "- **Current:** 42F - Mostly cloudy\n"
            "- **Tonight:** Low around 31F"
        )
        spoken = to_speakable_text(reply)
        for marker in ("*", "#", "🌤"):
            assert marker not in spoken
        assert "Today (Friday, November 13)." in spoken
        assert "Current: 42F" in spoken
        assert "Tonight: Low around 31F" in spoken

    def test_markup_only_reply_has_nothing_to_say(self):
        """Callers treat "" as "do not synthesize"; some backends error on it."""
        assert to_speakable_text("```\ncode only\n```") == ""
        assert to_speakable_text("---") == ""
        assert to_speakable_text("**") == ""

    def test_empty_input(self):
        assert to_speakable_text("") == ""

    def test_plain_prose_is_untouched(self):
        text = "It is 42 degrees and cloudy in Crossville, Tennessee."
        assert to_speakable_text(text) == text


class TestSpeakIntegration:
    """`speak()` must apply the normalization, not just offer it."""

    @staticmethod
    def _wire(monkeypatch, synthesized, played):
        import openjarvis.speech  # noqa: F401
        from openjarvis.core.registry import TTSRegistry
        from openjarvis.speech.tts import TTSResult

        class Backend:
            backend_id = "kokoro"

            def __init__(self, **_kwargs):
                pass

            def health(self):
                return True

            def synthesize(self, text, **_kwargs):
                synthesized.append(text)
                return TTSResult(audio=b"wav", sample_rate=24000, format="wav")

        monkeypatch.setattr(TTSRegistry, "contains", lambda key: key == "kokoro")
        monkeypatch.setattr(TTSRegistry, "get", lambda key: Backend)
        monkeypatch.setattr(
            "openjarvis.speech.voice_io.play_wav",
            lambda audio, **kwargs: played.append(audio),
        )

    def test_speak_sends_normalized_text_to_the_backend(self, monkeypatch):
        from openjarvis.cli._voice_chat import speak

        synthesized, played = [], []
        self._wire(monkeypatch, synthesized, played)

        class Console:
            def print(self, message):
                pass

        speak("### 🌤️ **Today**\n- **Current:** 42F", Console(), None)

        assert synthesized, "backend was never called"
        sent = synthesized[0]
        assert "*" not in sent and "#" not in sent
        assert "Current: 42F" in sent
        assert played == [b"wav"]

    def test_speak_skips_a_reply_with_nothing_to_say(self, monkeypatch):
        """A markup-only reply must not reach the backend at all."""
        from openjarvis.cli._voice_chat import speak

        synthesized, played = [], []
        self._wire(monkeypatch, synthesized, played)

        class Console:
            def print(self, message):
                pass

        speak("```\nx = 1\n```", Console(), None)

        assert synthesized == []
        assert played == []


class TestUnits:
    """Symbols a backend spells out letter by letter become words.

    Measured against Kokoro before the fix: "42°F" was "forty two degrees
    eff", "10 mph" was "ten em pee aych", "NW" was "en dubbelyu".
    """

    @pytest.mark.parametrize(
        "markdown,expected",
        [
            ("42°F", "42 degrees Fahrenheit"),
            ("42°C", "42 degrees Celsius"),
            ("300°K", "300 degrees Kelvin"),
            ("42 °F", "42 degrees Fahrenheit"),
            ("**42°F**", "42 degrees Fahrenheit"),
        ],
    )
    def test_degree_scales_are_named(self, markdown, expected):
        assert to_speakable_text(markdown) == expected

    def test_bare_degree_sign_is_left_alone(self):
        """ "72°" already reads as "seventy two degrees"."""
        assert to_speakable_text("72°") == "72°"

    @pytest.mark.parametrize(
        "markdown,expected",
        [
            ("10 mph", "10 miles per hour"),
            ("20 km/h", "20 kilometers per hour"),
            ("20 kph", "20 kilometers per hour"),
            ("5 kt", "5 knots"),
            ("1013 hPa", "1013 hectopascals"),
        ],
    )
    def test_unit_abbreviations_expand(self, markdown, expected):
        assert to_speakable_text(markdown) == expected

    def test_word_like_units_are_not_touched(self):
        """ "in" and "ft" collide with ordinary words, so they are left."""
        assert to_speakable_text("0.5 in of rain") == "0.5 in of rain"

    @pytest.mark.parametrize(
        "markdown,expected",
        [
            ("NW 10 mph", "northwest 10 miles per hour"),
            ("winds from the SSE", "winds from the southsoutheast"),
            ("NE wind", "northeast wind"),
        ],
    )
    def test_compass_bearings_expand(self, markdown, expected):
        assert to_speakable_text(markdown) == expected

    @pytest.mark.parametrize(
        "text", ["the N train", "Sgt. S. Wright went W", "plan B", "vitamin E"]
    )
    def test_single_letters_are_never_expanded(self, text):
        """A lone N/S/E/W is more often an initial than a bearing."""
        assert to_speakable_text(text) == text


class TestRanges:
    """A spaced hyphen between numbers is dropped by the backend entirely."""

    @pytest.mark.parametrize(
        "markdown,expected",
        [
            ("Highs 42 - 57", "Highs 42 to 57"),
            ("42°F - 57°F", "42 degrees Fahrenheit to 57 degrees Fahrenheit"),
            ("10–15 mph", "10 to 15 miles per hour"),
            ("10 — 15", "10 to 15"),
        ],
    )
    def test_number_ranges_say_to(self, markdown, expected):
        assert to_speakable_text(markdown) == expected

    @pytest.mark.parametrize(
        "text",
        [
            "2026-11-13 is a date",
            "phone 555-1234",
            "see page 10-12b",
        ],
    )
    def test_unspaced_hyphens_are_left_alone(self, text):
        """Dates and identifiers must survive; only a spaced dash is a range."""
        assert to_speakable_text(text) == text

    def test_reported_forecast_line(self):
        line = "(High around 42°F - 57°F depending on the source) NW 10 mph"
        spoken = to_speakable_text(line)
        assert "degrees Fahrenheit to 57 degrees Fahrenheit" in spoken
        assert "northwest 10 miles per hour" in spoken
        assert "°" not in spoken
        assert "mph" not in spoken
