"""`[engine.ollama].num_ctx` / `num_gpu` persist the runtime panel's answer.

Without them `jarvis chat` asks for context size and GPU offload on every
interactive run, and there is nowhere to record the answer: the values could
only be passed as `--num-ctx` / `--num-gpu` each time.
"""

from __future__ import annotations

from openjarvis.cli._runtime_panel import ChatRuntimeOptions
from openjarvis.core.config import JarvisConfig, OllamaEngineConfig


class TestOllamaRuntimeConfig:
    def test_defaults_are_unset(self):
        """Unset must be distinguishable from 0, which means CPU-only."""
        cfg = OllamaEngineConfig()
        assert cfg.num_ctx is None
        assert cfg.num_gpu is None

    def test_values_round_trip_from_toml(self, tmp_path):
        from openjarvis.core.config import load_config

        path = tmp_path / "config.toml"
        path.write_text(
            '[engine]\ndefault = "ollama"\n\n'
            "[engine.ollama]\nnum_ctx = 8192\nnum_gpu = -1\n"
        )
        cfg = load_config(path)
        assert cfg.engine.ollama.num_ctx == 8192
        assert cfg.engine.ollama.num_gpu == -1

    def test_absent_keys_stay_unset(self, tmp_path):
        from openjarvis.core.config import load_config

        path = tmp_path / "config.toml"
        path.write_text('[engine]\ndefault = "ollama"\n\n[engine.ollama]\nhost = ""\n')
        cfg = load_config(path)
        assert cfg.engine.ollama.num_ctx is None
        assert cfg.engine.ollama.num_gpu is None

    def test_zero_num_gpu_is_preserved_as_cpu_only(self, tmp_path):
        """0 means CPU-only and must not be read as "unset"."""
        from openjarvis.core.config import load_config

        path = tmp_path / "config.toml"
        path.write_text(
            '[engine]\ndefault = "ollama"\n\n[engine.ollama]\nnum_gpu = 0\n'
        )
        cfg = load_config(path)
        assert cfg.engine.ollama.num_gpu == 0
        assert ChatRuntimeOptions(num_gpu=0).to_engine_kwargs(engine_name="ollama") == {
            "num_gpu": 0
        }


class TestRuntimeOptionsTranslation:
    def test_negative_num_gpu_means_all_layers(self):
        from openjarvis.cli._runtime_panel import MAX_NUM_GPU_LAYERS

        kwargs = ChatRuntimeOptions(num_gpu=-1).to_engine_kwargs(engine_name="ollama")
        assert kwargs["num_gpu"] == MAX_NUM_GPU_LAYERS

    def test_num_ctx_is_capped(self):
        from openjarvis.cli._runtime_panel import MAX_NUM_CTX

        kwargs = ChatRuntimeOptions(num_ctx=10**9).to_engine_kwargs(
            engine_name="ollama"
        )
        assert kwargs["num_ctx"] == MAX_NUM_CTX

    def test_non_ollama_engine_takes_no_runtime_options(self):
        opts = ChatRuntimeOptions(num_ctx=8192, num_gpu=-1)
        assert opts.to_engine_kwargs(engine_name="vllm") == {}

    def test_unset_options_produce_no_kwargs(self):
        assert ChatRuntimeOptions().to_engine_kwargs(engine_name="ollama") == {}


def test_config_default_is_unset_on_a_fresh_config():
    assert JarvisConfig().engine.ollama.num_ctx is None
    assert JarvisConfig().engine.ollama.num_gpu is None


class TestResolveRuntimeOptions:
    """The panel must stop asking once the answer is recorded in config."""

    @staticmethod
    def _config(num_ctx=None, num_gpu=None):
        cfg = JarvisConfig()
        cfg.engine.ollama.num_ctx = num_ctx
        cfg.engine.ollama.num_gpu = num_gpu
        return cfg

    @staticmethod
    def _no_panel():
        def _boom(*_args, **_kwargs):
            raise AssertionError("runtime panel was shown")

        return _boom

    def test_configured_values_skip_the_panel(self, monkeypatch):
        from openjarvis.cli import _runtime_panel

        monkeypatch.setattr(_runtime_panel, "tty_wants_runtime_panel", lambda _s: True)
        monkeypatch.setattr(
            _runtime_panel, "interactive_pick_runtime_options", self._no_panel()
        )

        opts = _runtime_panel.resolve_runtime_options(
            None, engine_name="ollama", config=self._config(num_ctx=8192, num_gpu=-1)
        )

        assert opts.num_ctx == 8192
        assert opts.num_gpu == -1

    def test_only_num_gpu_configured_still_skips_the_panel(self, monkeypatch):
        from openjarvis.cli import _runtime_panel

        monkeypatch.setattr(_runtime_panel, "tty_wants_runtime_panel", lambda _s: True)
        monkeypatch.setattr(
            _runtime_panel, "interactive_pick_runtime_options", self._no_panel()
        )

        opts = _runtime_panel.resolve_runtime_options(
            None, engine_name="ollama", config=self._config(num_gpu=0)
        )

        assert opts.num_gpu == 0
        assert opts.num_ctx is None

    def test_cli_flags_override_config(self, monkeypatch):
        from openjarvis.cli import _runtime_panel

        monkeypatch.setattr(_runtime_panel, "tty_wants_runtime_panel", lambda _s: True)
        monkeypatch.setattr(
            _runtime_panel, "interactive_pick_runtime_options", self._no_panel()
        )

        opts = _runtime_panel.resolve_runtime_options(
            None,
            engine_name="ollama",
            config=self._config(num_ctx=8192, num_gpu=-1),
            cli_num_ctx=2048,
        )

        assert opts.num_ctx == 2048
        assert opts.num_gpu == -1  # config still supplies the unflagged one

    def test_panel_still_runs_when_nothing_is_configured(self, monkeypatch):
        from openjarvis.cli import _runtime_panel

        shown = []
        monkeypatch.setattr(_runtime_panel, "tty_wants_runtime_panel", lambda _s: True)
        monkeypatch.setattr(
            _runtime_panel,
            "interactive_pick_runtime_options",
            lambda *a, **k: shown.append(True) or ChatRuntimeOptions(num_ctx=4096),
        )

        opts = _runtime_panel.resolve_runtime_options(
            None, engine_name="ollama", config=self._config()
        )

        assert shown == [True]
        assert opts.num_ctx == 4096

    def test_non_ollama_engine_never_prompts_or_configures(self, monkeypatch):
        from openjarvis.cli import _runtime_panel

        monkeypatch.setattr(_runtime_panel, "tty_wants_runtime_panel", lambda _s: True)
        monkeypatch.setattr(
            _runtime_panel, "interactive_pick_runtime_options", self._no_panel()
        )

        opts = _runtime_panel.resolve_runtime_options(
            None, engine_name="vllm", config=self._config(num_ctx=8192)
        )

        assert opts == ChatRuntimeOptions()

    def test_skip_panel_without_config_falls_back_to_defaults(self, monkeypatch):
        from openjarvis.cli import _runtime_panel

        monkeypatch.setattr(
            _runtime_panel, "interactive_pick_runtime_options", self._no_panel()
        )

        opts = _runtime_panel.resolve_runtime_options(
            None, engine_name="ollama", config=self._config(), skip_panel=True
        )

        assert opts == ChatRuntimeOptions()
