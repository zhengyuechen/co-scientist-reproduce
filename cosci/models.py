"""Domain models. Field set matches spec §8 context-memory schema."""
from __future__ import annotations
from enum import Enum
from pydantic import BaseModel, Field

class AgentName(str, Enum):
    GENERATION = "generation"
    REFLECTION = "reflection"
    RANKING = "ranking"
    EVOLUTION = "evolution"
    PROXIMITY = "proximity"
    META_REVIEW = "meta_review"

class TaskType(str, Enum):
    CREATE_INITIAL_HYPOTHESES = "create_initial_hypotheses"
    REVIEW_HYPOTHESIS = "review_hypothesis"
    ADD_TO_TOURNAMENT = "add_to_tournament"
    RUN_TOURNAMENT_BATCH = "run_tournament_batch"
    EVOLVE_TOP = "evolve_top"
    UPDATE_PROXIMITY = "update_proximity"
    GENERATE_SYSTEM_FEEDBACK = "generate_system_feedback"
    GENERATE_FINAL_OVERVIEW = "generate_final_overview"

class Origin(str, Enum):
    GENERATED = "generated"
    EVOLVED = "evolved"
    USER_SEED = "user_seed"

class Safety(str, Enum):
    UNREVIEWED = "unreviewed"
    SAFE = "safe"
    UNSAFE = "unsafe"

class DebateMode(str, Enum):
    SINGLE_TURN = "single_turn"
    MULTI_TURN = "multi_turn"

class ResearchPlan(BaseModel):
    goal: str
    preferences: list[str] = []
    attributes: list[str] = []
    constraints: list[str] = []

class Hypothesis(BaseModel):
    id: str
    text: str
    title: str
    source_strategy: str
    parent_ids: list[str] = []
    elo_rating: float | None = None
    matches_played: int = 0
    active: bool = True
    created_tick: int = 0
    safety: Safety = Safety.UNREVIEWED
    safety_reason: str | None = None
    origin: Origin = Origin.GENERATED

class Review(BaseModel):
    hypothesis_id: str
    type: str
    scores: dict[str, float] = {}
    text: str = ""
    references: list[str] = []
    tool_grounded: bool = False
    safety: Safety = Safety.UNREVIEWED
    safety_reason: str | None = None

class Task(BaseModel):
    agent: AgentName
    action: TaskType
    target_id: str | None = None
    priority: int = 5  # lower = sooner
    payload: dict = Field(default_factory=dict)

class MatchResult(BaseModel):
    a_id: str
    b_id: str
    mode: DebateMode
    winner_id: str
    loser_id: str
    elo_before: dict[str, float]
    elo_after: dict[str, float]
    transcript: str
    tick: int
