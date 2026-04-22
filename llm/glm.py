# llm/glm.py
import os
from openai import OpenAI
from llm.base import BaseLLM

class GLMAdapter(BaseLLM):
    def __init__(self, model: str | None = None, api_key: str | None = None):
        # 1. 从环境变量读取 API Key (吸收了GPT的优点)
        api_key = api_key or os.getenv("GLM_API_KEY")
        if not api_key:
            raise ValueError("请在环境变量中设置 GLM_API_KEY")
            
        # 2. 延迟初始化，只有真正用到时才连接，避免全局报错
        self.client = OpenAI(
            api_key=api_key, 
            base_url="https://open.bigmodel.cn/api/paas/v4/"
        )
        self.model = model or os.getenv("GLM_MODEL_NAME") or "glm-4-flash"

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

    def chat_stream(self, messages, tools=None):
        kwargs = {"model": self.model, "messages": messages, "stream": True}
        if tools:
            kwargs["tools"] = tools
            kwargs["tool_choice"] = "auto"

        stream = self.client.chat.completions.create(**kwargs)
        tool_calls_buffer = {}

        for chunk in stream:
            delta = chunk.choices[0].delta
            if delta.content:
                yield {"type": "text", "content": delta.content}

            if delta.tool_calls:
                for tc_chunk in delta.tool_calls:
                    idx = tc_chunk.index
                    if idx not in tool_calls_buffer:
                        tool_calls_buffer[idx] = {"name": "", "arguments": ""}
                    if tc_chunk.function.name:
                        tool_calls_buffer[idx]["name"] += tc_chunk.function.name
                    if tc_chunk.function.arguments:
                        tool_calls_buffer[idx]["arguments"] += tc_chunk.function.arguments

        if tool_calls_buffer:
            yield {"type": "tool_calls", "tool_calls": list(tool_calls_buffer.values())}
