# llm/siliconflow.py
import os
import time
from openai import OpenAI
from llm.base import BaseLLM

class SiliconFlowAdapter(BaseLLM):
    def __init__(self, model: str | None = None, api_key: str | None = None):
        api_key = api_key or os.getenv("SILICONFLOW_API_KEY")
        if not api_key:
            raise ValueError("请在环境变量中设置 SILICONFLOW_API_KEY")
            
        self.client = OpenAI(
            api_key=api_key, 
            base_url="https://api.siliconflow.cn/v1"  # 硅基流动的专属接口地址
        )
        self.model = model or os.getenv("SILICONFLOW_MODEL_NAME") or "deepseek-ai/DeepSeek-V3"

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
        
        tool_calls_buffer = {} # 用来攒工具调用的碎片
        
        for chunk in stream:
            delta = chunk.choices[0].delta
            
            # 1. 如果是普通文本，直接吐出去（打字机效果）
            if delta.content:
                yield {"type": "text", "content": delta.content}
                
            # 2. 如果是工具调用，不能直接吐，要偷偷攒起来
            if delta.tool_calls:
                for tc_chunk in delta.tool_calls:
                    idx = tc_chunk.index
                    if idx not in tool_calls_buffer:
                        tool_calls_buffer[idx] = {"name": "", "arguments": ""}
                    if tc_chunk.function.name:
                        tool_calls_buffer[idx]["name"] += tc_chunk.function.name
                    if tc_chunk.function.arguments:
                        tool_calls_buffer[idx]["arguments"] += tc_chunk.function.arguments
                        
        # 3. 流结束后，如果攒了工具调用，一次性丢给 Agent 处理
        if tool_calls_buffer:
            yield {"type": "tool_calls", "tool_calls": list(tool_calls_buffer.values())}

    def embed(self, texts, model=None):
        if not texts:
            return []

        normalized_inputs = [(index, text) for index, text in enumerate(texts) if text and text.strip()]
        if not normalized_inputs:
            return [[] for _ in texts]

        retries = max(int(os.getenv("EMBEDDING_MAX_RETRIES", "2")), 0)
        retry_delay = max(float(os.getenv("EMBEDDING_RETRY_DELAY_SECONDS", "0.8")), 0.0)
        last_error = None

        for attempt in range(retries + 1):
            try:
                response = self.client.embeddings.create(
                    model=model or os.getenv("EMBEDDING_MODEL_NAME") or "BAAI/bge-m3",
                    input=[text for _, text in normalized_inputs],
                    encoding_format="float",
                )
                vectors = [[] for _ in texts]
                for (original_index, _), item in zip(normalized_inputs, response.data):
                    vectors[original_index] = item.embedding
                return vectors
            except Exception as exc:
                last_error = exc
                if attempt >= retries:
                    break
                time.sleep(retry_delay)

        return [[] for _ in texts]
