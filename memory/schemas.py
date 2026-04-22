from enum import Enum
from typing import Any

from pydantic import BaseModel, Field


class AgentRole(str, Enum):
    ROUTER = "router"
    MEMORY_MANAGER = "memory_manager"
    PROFILE_MANAGER = "profile_manager"
    KNOWLEDGE_MANAGER = "knowledge_manager"
    WEB_SEARCHER = "web_searcher"
    INTERACTION = "interaction"


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
    status: TaskStatus = TaskStatus.PENDING


class TaskResult(BaseModel):
    task_id: str
    owner: AgentRole
    status: TaskStatus
    summary: str
    data: dict[str, Any] = Field(default_factory=dict)


class ExecutionPlan(BaseModel):
    mode: str = "direct"
    rationale: str = ""
    tasks: list[TaskSpec] = Field(default_factory=list)

