from enum import Enum
from typing import Any

from pydantic import BaseModel, Field


class AgentRole(str, Enum):
    CONTROLLER = "controller"
    MEMORY_MANAGER = "memory_manager"
    PROFILE_MANAGER = "profile_manager"
    KNOWLEDGE_MANAGER = "knowledge_manager"
    WEB_SEARCHER = "web_searcher"
    RESPONDER = "responder"


class TaskStatus(str, Enum):
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    SKIPPED = "skipped"


class TaskSpec(BaseModel):
    id: str
    owner: AgentRole
    goal: str
    instructions: str = ""
    payload: dict[str, Any] = Field(default_factory=dict)
    depends_on: list[str] = Field(default_factory=list)
    required_payload_fields: list[str] = Field(default_factory=list)
    required_context_fields: list[str] = Field(default_factory=list)
    max_retries: int = 0
    retry_count: int = 0
    run_if: dict[str, Any] = Field(default_factory=dict)
    status: TaskStatus = TaskStatus.PENDING


class TaskResult(BaseModel):
    task_id: str
    owner: AgentRole
    status: TaskStatus
    summary: str
    attempts: int = 1
    error: str = ""
    data: dict[str, Any] = Field(default_factory=dict)


class SemanticDecision(BaseModel):
    response_mode: str = "direct"
    reason: str = ""
    read_memory: bool = False
    write_memory: bool = False
    read_profile: bool = False
    write_profile: bool = False
    use_knowledge: bool = False
    use_web: bool = False
    knowledge_query: str = ""
    web_query: str = ""
