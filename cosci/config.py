"""Typed configuration loaded from config.yaml + .env. All defaults are OURS (spec §9)."""
from __future__ import annotations
import os
import yaml
from pydantic import BaseModel
from dotenv import load_dotenv

class EloCfg(BaseModel):
    k_factor: int = 32
    scale: int = 400

class BudgetCfg(BaseModel):
    max_ideas: int = 20
    max_matches_per_idea: int = 8
    max_wallclock_s: int = 1800
    max_tokens: int | None = None

class DebateCfg(BaseModel):
    turns_typical_min: int = 3
    turns_typical_max: int = 5
    turns_max: int = 10

class EvoCfg(BaseModel):
    top_k: int = 5

class OverviewCfg(BaseModel):
    top_n: int = 10
    min_novelty: float = 5.0   # hypotheses scoring below this are excluded from the overview as restatements

class ProxCfg(BaseModel):
    method: str = "embeddings"
    model: str = "sentence-transformers/all-MiniLM-L6-v2"

class GroundCfg(BaseModel):
    backend: str = "arxiv"

class PlateauCfg(BaseModel):
    window: int = 10
    epsilon: float = 5.0

class Config(BaseModel):
    default_model: str
    models: dict[str, str] = {}
    elo: EloCfg = EloCfg()
    budget: BudgetCfg = BudgetCfg()
    temperature: dict[str, float] = {}
    debate: DebateCfg = DebateCfg()
    evolution: EvoCfg = EvoCfg()
    overview: OverviewCfg = OverviewCfg()
    workers: int = 4
    scheduler: str = "continuous"
    proximity: ProxCfg = ProxCfg()
    grounding: GroundCfg = GroundCfg()
    plateau: PlateauCfg = PlateauCfg()

    def model_for(self, agent: str) -> str:
        return self.models.get(agent, self.default_model)

def load_config(path: str = "config.yaml") -> Config:
    load_dotenv()  # populate os.environ from .env if present
    with open(path) as f:
        data = yaml.safe_load(f)
    return Config(**data)

def require_openrouter_key() -> str:
    key = os.environ.get("OPENROUTER_API_KEY")
    if not key:
        raise RuntimeError("OPENROUTER_API_KEY not set (add it to .env or the environment).")
    return key
