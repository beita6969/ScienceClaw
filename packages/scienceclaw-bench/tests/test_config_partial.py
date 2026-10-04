"""config.load_config: partial nested mappings keep the default instance's other fields."""
from scienceclaw.config import RunConfig, load_config


def test_partial_role_override_keeps_default_instance_fields():
    cfg = load_config(None, {"llm.policy.model": "lab-gpt-4.1-mini"})
    assert cfg.llm.policy.model == "lab-gpt-4.1-mini"
    assert cfg.llm.policy.json_mode is True                  # default_factory value, not the ModelRole default
    assert cfg.llm.executor.max_tokens == RunConfig().llm.executor.max_tokens


def test_empty_config_equals_defaults_and_instances_are_independent():
    a, b = load_config(None, {}), load_config(None, {"bench.disciplines": ["TOY"]})
    assert a.to_dict() == RunConfig().to_dict()
    assert a.bench.disciplines == [] and b.bench.disciplines == ["TOY"]


def test_gate_warning_when_noise_guard_exceeds_n_val():
    cfg = load_config(None, {"bench.n_val": 1})
    assert cfg.evolution.min_improved_episodes == 2 and len(cfg.gate_warnings()) == 1
    assert load_config(None, {"bench.n_val": 1, "evolution.min_improved_episodes": 1}).gate_warnings() == []
    assert load_config(None, {}).gate_warnings() == []
