# llm/glm.py
import os
from openai import OpenAI
from llm.base import BaseLLM

class GLMAdapter(BaseLLM):
    def __init__(self):
        # 1. 从环境变量读取 API Key (吸收了GPT的优点)
        api_key = os.getenv("GLM_API_KEY")
        if not api_key:
            raise ValueError("请在环境变量中设置 GLM_API_KEY")
            
        # 2. 延迟初始化，只有真正用到时才连接，避免全局报错
        self.client = OpenAI(
            api_key=api_key, 
            base_url="https://open.bigmodel.cn/api/paas/v4/"
        )
        self.model = "glm-4-flash" # 推荐先用便宜的 flash 测试

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