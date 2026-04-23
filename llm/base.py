from abc import ABC, abstractmethod
from typing import Dict, Generator, List


class BaseLLM(ABC):
    @abstractmethod
    def chat(self, messages: List[Dict], tools: list = None) -> Dict:
        """返回格式约定: {"content": str, "tool_calls": list}"""
        pass

    @abstractmethod
    def chat_stream(self, messages: List[Dict], tools: list = None) -> Generator[Dict, None, None]:
        """流式调用：每次 yield 出一块数据"""
        pass

    @abstractmethod
    def embed(self, texts: List[str], model: str | None = None) -> List[List[float]]:
        """返回文本列表的 embedding 向量"""
        pass
