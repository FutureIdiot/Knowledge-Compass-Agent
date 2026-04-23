import os

from llm.glm import GLMAdapter
from llm.siliconflow import SiliconFlowAdapter


ROLE_ENV_PREFIXES = {
    "controller": "CONTROLLER",
    "memory_manager": "MEMORY_MANAGER",
    "profile_manager": "PROFILE_MANAGER",
    "knowledge_manager": "KNOWLEDGE_MANAGER",
    "web_searcher": "WEB_SEARCHER",
    "responder": "RESPONDER",
}


def get_role_llm(role_name: str):
    provider = _get_role_env(role_name, "LLM_PROVIDER") or os.getenv("LLM_PROVIDER", "glm")
    model_name = _get_role_env(role_name, "MODEL_NAME") or os.getenv("MODEL_NAME")

    if provider == "glm":
        return GLMAdapter(model=model_name)
    if provider == "siliconflow":
        return SiliconFlowAdapter(model=model_name)
    raise ValueError(f"不支持的 LLM 提供商: {provider}")


def _get_role_env(role_name: str, key: str) -> str | None:
    prefix = ROLE_ENV_PREFIXES.get(role_name)
    legacy_prefixes = {
        "controller": "ROUTER",
        "responder": "INTERACTION",
    }
    if prefix:
        value = os.getenv(f"{prefix}_{key}")
        if value:
            return value
    legacy_prefix = legacy_prefixes.get(role_name)
    if legacy_prefix:
        return os.getenv(f"{legacy_prefix}_{key}")
    return None
