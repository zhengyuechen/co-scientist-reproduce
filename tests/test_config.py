from cosci.config import load_config

def test_load_defaults(tmp_path):
    cfg_text = (
        'default_model: "m/free"\n'
        'models: { generation: "m/strong" }\n'
        'elo: { k_factor: 32, scale: 400 }\n'
        'budget: { max_ideas: 20, max_matches_per_idea: 8, max_wallclock_s: 1800, max_tokens: null }\n'
        'temperature: { generation: 0.7, reflection: 0.5, ranking: 0.3, evolution: 0.8 }\n'
        'debate: { turns_typical_min: 3, turns_typical_max: 5, turns_max: 10 }\n'
        'evolution: { top_k: 5 }\n'
        'overview: { top_n: 10 }\n'
        'workers: 4\n'
        'scheduler: "continuous"\n'
        'proximity: { method: "embeddings", model: "mini" }\n'
        'grounding: { backend: "arxiv" }\n'
        'plateau: { window: 10, epsilon: 5.0 }\n'
    )
    p = tmp_path / "config.yaml"
    p.write_text(cfg_text)
    cfg = load_config(str(p))
    assert cfg.elo.k_factor == 32
    assert cfg.budget.max_ideas == 20
    assert cfg.model_for("generation") == "m/strong"   # override used
    assert cfg.model_for("ranking") == "m/free"          # falls back to default
    assert cfg.scheduler == "continuous"
