# core/base_tool.py
from abc import ABC, abstractmethod

class BaseTool(ABC):
    """所有工具的基类，规定工具必须有哪些基本动作"""
    
    name: str = ""
    description: str = ""

    @abstractmethod
    def execute(self, input_data: str) -> str:
        """
        执行工具的具体逻辑
        参数统一为字符串，返回也必须是字符串（方便LLM理解）
        """
        pass

    def get_schema(self) -> dict:
        """
        将工具转换为 OpenAI Function Calling 需要的 JSON 格式
        （因为我们用的智谱GLM兼容这个格式）
        """
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": {
                    "type": "object",
                    "properties": {
                        "input": {
                            "type": "string",
                            "description": "传给该工具的参数"
                        }
                    },
                    "required": ["input"]
                }
            }
        }