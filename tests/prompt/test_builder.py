from __future__ import annotations

from datetime import datetime
from pathlib import Path

import pytest

from openjarvis.core.config import MemoryFilesConfig, SystemPromptConfig


@pytest.fixture
def memory_dir(tmp_path: Path) -> Path:
    soul = tmp_path / "SOUL.md"
    soul.write_text("You are a helpful research assistant.")
    memory = tmp_path / "MEMORY.md"
    memory.write_text("- User prefers concise answers\n- User is a data scientist")
    user = tmp_path / "USER.md"
    user.write_text("- Name: Alice\n- Role: ML Engineer")
    return tmp_path


def test_build_frozen_prefix(memory_dir: Path):
    from openjarvis.prompt.builder import SystemPromptBuilder

    builder = SystemPromptBuilder(
        agent_template="You are Jarvis.",
        memory_files_config=MemoryFilesConfig(
            soul_path=str(memory_dir / "SOUL.md"),
            memory_path=str(memory_dir / "MEMORY.md"),
            user_path=str(memory_dir / "USER.md"),
        ),
        system_prompt_config=SystemPromptConfig(),
    )
    prompt = builder.build()
    assert "Jarvis" in prompt
    assert "helpful research assistant" in prompt
    assert "concise answers" in prompt
    assert "Alice" in prompt


def test_config_prefix_prepended(memory_dir: Path):
    """Regression for #401: a configured system_prompt.prefix leads the
    assembled prompt, ahead of the agent template, and is exposed as a
    'prefix' section."""
    from openjarvis.prompt.builder import SystemPromptBuilder

    builder = SystemPromptBuilder(
        agent_template="You are Jarvis.",
        memory_files_config=MemoryFilesConfig(
            soul_path=str(memory_dir / "SOUL.md"),
            memory_path=str(memory_dir / "MEMORY.md"),
            user_path=str(memory_dir / "USER.md"),
        ),
        system_prompt_config=SystemPromptConfig(prefix="ALWAYS ANSWER AS JARVIS."),
    )
    prompt = builder.build()
    assert prompt.startswith("ALWAYS ANSWER AS JARVIS.")
    assert "You are Jarvis." in prompt
    # Prefix is visible in the inspection API too (#457), as a frozen section.
    section_names = [s.name for s in builder.sections()]
    assert section_names[0] == "prefix"
    assert builder.sections()[0].cache_segment == "frozen_prefix"


def test_empty_prefix_leaves_prompt_unchanged(memory_dir: Path):
    """Backward compatibility: the default empty prefix adds no section and
    leaves build() output identical to having no prefix configured."""
    from openjarvis.prompt.builder import SystemPromptBuilder

    def _make(prefix: str) -> SystemPromptBuilder:
        return SystemPromptBuilder(
            agent_template="You are Jarvis.",
            memory_files_config=MemoryFilesConfig(
                soul_path=str(memory_dir / "SOUL.md"),
                memory_path=str(memory_dir / "MEMORY.md"),
                user_path=str(memory_dir / "USER.md"),
            ),
            system_prompt_config=SystemPromptConfig(prefix=prefix),
        )

    assert _make("").build() == _make("").build()
    # No 'prefix' section is emitted when prefix is empty.
    assert "prefix" not in [s.name for s in _make("").sections()]
    # And the first section is the agent template, as before.
    assert _make("").sections()[0].name == "agent_template"


def test_frozen_prefix_stability(memory_dir: Path):
    from openjarvis.prompt.builder import SystemPromptBuilder

    builder = SystemPromptBuilder(
        agent_template="You are Jarvis.",
        memory_files_config=MemoryFilesConfig(
            soul_path=str(memory_dir / "SOUL.md"),
            memory_path=str(memory_dir / "MEMORY.md"),
            user_path=str(memory_dir / "USER.md"),
        ),
        system_prompt_config=SystemPromptConfig(),
    )
    first = builder.build()
    (memory_dir / "MEMORY.md").write_text("- CHANGED CONTENT")
    second = builder.build()
    assert first == second


def test_char_limit_truncation(memory_dir: Path):
    from openjarvis.prompt.builder import SystemPromptBuilder

    (memory_dir / "SOUL.md").write_text("x" * 10000)
    builder = SystemPromptBuilder(
        agent_template="You are Jarvis.",
        memory_files_config=MemoryFilesConfig(
            soul_path=str(memory_dir / "SOUL.md"),
            memory_path=str(memory_dir / "MEMORY.md"),
            user_path=str(memory_dir / "USER.md"),
        ),
        system_prompt_config=SystemPromptConfig(soul_max_chars=100),
    )
    prompt = builder.build()
    assert prompt.count("x") <= 100
    assert "truncated" in prompt.lower()


def test_skill_index_in_prompt(memory_dir: Path):
    from openjarvis.prompt.builder import SystemPromptBuilder

    skills = [("api_health_check", "Check API health across all endpoints")]
    builder = SystemPromptBuilder(
        agent_template="You are Jarvis.",
        memory_files_config=MemoryFilesConfig(
            soul_path=str(memory_dir / "SOUL.md"),
            memory_path=str(memory_dir / "MEMORY.md"),
            user_path=str(memory_dir / "USER.md"),
        ),
        system_prompt_config=SystemPromptConfig(),
        skill_index=skills,
    )
    prompt = builder.build()
    assert "api_health_check" in prompt
    assert "Check API health" in prompt


def test_dynamic_section_appended(memory_dir: Path):
    from openjarvis.prompt.builder import SystemPromptBuilder

    builder = SystemPromptBuilder(
        agent_template="You are Jarvis.",
        memory_files_config=MemoryFilesConfig(
            soul_path=str(memory_dir / "SOUL.md"),
            memory_path=str(memory_dir / "MEMORY.md"),
            user_path=str(memory_dir / "USER.md"),
        ),
        system_prompt_config=SystemPromptConfig(),
        session_context="Platform: CLI | Session: abc123",
    )
    prompt = builder.build()
    assert "Platform: CLI" in prompt


def test_sections_expose_prompt_metadata(memory_dir: Path):
    from openjarvis.prompt.builder import SystemPromptBuilder

    builder = SystemPromptBuilder(
        agent_template="You are Jarvis.",
        memory_files_config=MemoryFilesConfig(
            soul_path=str(memory_dir / "SOUL.md"),
            memory_path=str(memory_dir / "MEMORY.md"),
            user_path=str(memory_dir / "USER.md"),
        ),
        system_prompt_config=SystemPromptConfig(),
        session_context="Platform: CLI | Session: abc123",
        previous_state="Last task: summarize telemetry.",
    )

    sections = builder.sections()

    assert [section.name for section in sections] == [
        "agent_template",
        "soul",
        "memory",
        "user",
        "current_date",
        "session_context",
        "previous_state",
    ]
    assert sections[1].source == str(memory_dir / "SOUL.md")
    assert sections[1].cache_segment == "frozen_prefix"
    assert sections[-1].cache_segment == "dynamic_suffix"
    assert builder.build() == "\n\n".join(section.content for section in sections)


def test_sections_keep_frozen_file_content_stable(memory_dir: Path):
    from openjarvis.prompt.builder import SystemPromptBuilder

    builder = SystemPromptBuilder(
        agent_template="You are Jarvis.",
        memory_files_config=MemoryFilesConfig(
            soul_path=str(memory_dir / "SOUL.md"),
            memory_path=str(memory_dir / "MEMORY.md"),
            user_path=str(memory_dir / "USER.md"),
        ),
        system_prompt_config=SystemPromptConfig(),
    )

    first = builder.sections()
    (memory_dir / "MEMORY.md").write_text("- CHANGED CONTENT")
    second = builder.sections()

    assert first == second


def test_missing_files_handled(tmp_path: Path):
    from openjarvis.prompt.builder import SystemPromptBuilder

    builder = SystemPromptBuilder(
        agent_template="You are Jarvis.",
        memory_files_config=MemoryFilesConfig(
            soul_path=str(tmp_path / "missing_soul.md"),
            memory_path=str(tmp_path / "missing_memory.md"),
            user_path=str(tmp_path / "missing_user.md"),
        ),
        system_prompt_config=SystemPromptConfig(),
    )
    prompt = builder.build()
    assert "Jarvis" in prompt


# -- current date injection ------------------------------------------------
#
# A model is never told what day it is, so "today" questions get answered from
# training data as a confidently wrong date. These pin the fix, and pin that it
# stays out of the cached frozen prefix.


def _fixed_clock(dt: datetime):
    return lambda: dt


def test_current_date_injected_into_prompt(tmp_path: Path):
    from openjarvis.prompt.builder import SystemPromptBuilder

    builder = SystemPromptBuilder(
        agent_template="You are Jarvis.",
        memory_files_config=MemoryFilesConfig(
            soul_path="", memory_path="", user_path=""
        ),
        system_prompt_config=SystemPromptConfig(),
        now=_fixed_clock(datetime(2026, 11, 13, 14, 30)),
    )

    prompt = builder.build()

    assert "## Current Date" in prompt
    assert "2026-11-13" in prompt
    assert "Friday, 13 November 2026" in prompt
    assert "14:30" in prompt


def test_current_date_is_dynamic_suffix_not_frozen(tmp_path: Path):
    """The date must never land in the cached frozen prefix."""
    from openjarvis.prompt.builder import SystemPromptBuilder

    builder = SystemPromptBuilder(
        agent_template="You are Jarvis.",
        memory_files_config=MemoryFilesConfig(
            soul_path="", memory_path="", user_path=""
        ),
        system_prompt_config=SystemPromptConfig(),
        now=_fixed_clock(datetime(2026, 11, 13, 14, 30)),
    )

    date_sections = [s for s in builder.sections() if s.name == "current_date"]
    assert len(date_sections) == 1
    assert date_sections[0].cache_segment == "dynamic_suffix"

    frozen = [s for s in builder.sections() if s.cache_segment == "frozen_prefix"]
    assert all("2026-11-13" not in s.content for s in frozen)


def test_frozen_prefix_stable_across_date_change(tmp_path: Path):
    """Rolling the clock must not invalidate the frozen prefix."""
    from openjarvis.prompt.builder import SystemPromptBuilder

    def _builder(dt: datetime):
        return SystemPromptBuilder(
            agent_template="You are Jarvis.",
            memory_files_config=MemoryFilesConfig(
                soul_path="", memory_path="", user_path=""
            ),
            system_prompt_config=SystemPromptConfig(),
            now=_fixed_clock(dt),
        )

    day_one = _builder(datetime(2026, 11, 13, 9, 0))
    day_two = _builder(datetime(2027, 3, 1, 23, 45))

    def frozen(b):
        return [s.content for s in b.sections() if s.cache_segment == "frozen_prefix"]

    assert frozen(day_one) == frozen(day_two)
    assert day_one.build() != day_two.build()


def test_current_date_can_be_disabled(tmp_path: Path):
    from openjarvis.prompt.builder import SystemPromptBuilder

    builder = SystemPromptBuilder(
        agent_template="You are Jarvis.",
        memory_files_config=MemoryFilesConfig(
            soul_path="", memory_path="", user_path=""
        ),
        system_prompt_config=SystemPromptConfig(include_current_date=False),
        now=_fixed_clock(datetime(2026, 11, 13, 14, 30)),
    )

    assert "## Current Date" not in builder.build()
    assert all(s.name != "current_date" for s in builder.sections())


def test_configured_timezone_used_for_date(tmp_path: Path):
    from openjarvis.prompt.builder import SystemPromptBuilder

    builder = SystemPromptBuilder(
        agent_template="You are Jarvis.",
        memory_files_config=MemoryFilesConfig(
            soul_path="", memory_path="", user_path=""
        ),
        system_prompt_config=SystemPromptConfig(timezone="America/New_York"),
    )

    now = builder._resolve_now()
    assert now.tzinfo is not None
    assert now.tzname() in {"EST", "EDT"}


def test_invalid_timezone_falls_back_to_local(tmp_path: Path):
    """A typo'd timezone must not break every prompt the agent builds."""
    from openjarvis.prompt.builder import SystemPromptBuilder

    builder = SystemPromptBuilder(
        agent_template="You are Jarvis.",
        memory_files_config=MemoryFilesConfig(
            soul_path="", memory_path="", user_path=""
        ),
        system_prompt_config=SystemPromptConfig(timezone="Not/A_Zone"),
    )

    prompt = builder.build()
    assert "## Current Date" in prompt


def test_invalid_timezone_warns_once_not_per_prompt(caplog):
    """The zone is resolved per builder, not per build()."""
    import logging

    from openjarvis.prompt.builder import SystemPromptBuilder

    with caplog.at_level(logging.WARNING, logger="openjarvis.prompt.builder"):
        builder = SystemPromptBuilder(
            agent_template="You are Jarvis.",
            memory_files_config=MemoryFilesConfig(
                soul_path="", memory_path="", user_path=""
            ),
            system_prompt_config=SystemPromptConfig(timezone="Not/A_Zone"),
        )
        for _ in range(5):
            builder.build()

    matching = [r for r in caplog.records if "system_prompt.timezone" in r.message]
    assert len(matching) == 1
