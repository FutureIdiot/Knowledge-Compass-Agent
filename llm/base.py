# base.py
from abc import ABC, abstractmethod
from typing import List, Dict, Generator

class BaseLLM(ABC):
    @abstractmethod
    def chat(self, messages: List[Dict], tools: list = None) -> Dict:
        """返回格式约定: {"content": str, "tool_calls": list}"""
        """非流式调用（保留备用）"""
        pass
        
    @abstractmethod
    def chat_stream(self, messages: List[Dict], tools: list = None) -> Generator[Dict, None, None]:
        """流式调用：每次 yield 出一块数据"""
        pass

# glm.py (假设你用智谱API，兼容OpenAI格式)
from openai import OpenAI
from llm.base import BaseLLM
import json

class GLMAdapter(BaseLLM):
    def __init__(self, api_key: str, model: str = "glm-4"):
        self.client = OpenAI(api_key=api_key, base_url="https://open.bigmodel.cn/api/paas/v4/")
        self.model = model

    def chat(self, messages, tools=None):
        kwargs = {"model": self.model, "messages": messages}
        if tools:
            kwargs["tools"] = tools
            kwargs["tool_choice"] = "auto"
            
        response = self.client.chat.completions.create(**kwargs)
        msg = response.choices[0].message
        
        result = {"content": msg.content}
        if msg.tool_calls:
            result["tool_calls"] = [
                {"name": tc.function.name, "arguments": tc.function.arguments} 
                for tc in msg.tool_calls
            ]
        return result