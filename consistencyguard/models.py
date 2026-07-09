from pydantic import BaseModel, Field
from typing import Optional, Any
from datetime import datetime
from enum import Enum


class ViolationSeverity(str, Enum):
    CRITICAL = "critical"   # responses contradict each other
    WARNING = "warning"     # responses differ materially
    INFO = "info"           # minor phrasing variation


class ValidationIssue(str, Enum):
    INVALID_JSON = "invalid_json"       # response could not be parsed as JSON
    SCHEMA_MISMATCH = "schema_mismatch" # parsed JSON missing required keys or wrong types


class LLMCall(BaseModel):
    id: Optional[int] = None
    prompt: str
    response: str
    model: str
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    prompt_embedding: Optional[list[float]] = None
    agent_id: Optional[str] = "default"


class SimilarCall(BaseModel):
    call_id: int
    prompt: str
    response: str
    similarity_score: float
    timestamp: datetime



class ConsistencyViolation(BaseModel):
    call_id_new: int
    call_id_ref: int
    prompt_similarity: float
    response_divergence: float
    severity: ViolationSeverity
    new_prompt: str
    new_response: str
    ref_response: str
    explanation: str
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    agent_id: str = "default"
    validation_issues: list[ValidationIssue] = Field(default_factory=list)


class GuardReport(BaseModel):
    total_calls: int
    total_violations: int
    critical_count: int
    warning_count: int
    info_count: int
    top_violating_prompts: list[str]
    violation_rate: float
    period_hours: int


class TrendBucket(BaseModel):
    bucket: str
    count: int
    critical: int
    warning: int
    info: int


class AgentStats(BaseModel):
    agent_id: str
    total_violations: int
    critical: int
    warning: int
    info: int
    total_calls: int
    violation_rate: float
    last_violation: Optional[datetime] = None
