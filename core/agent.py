# core/agent.py
import json
from llm.base import BaseLLM

class LearningNavigatorAgent:
    def __init__(self, llm: BaseLLM, tools: list):
        self.llm = llm
        # 将工具列表转为字典，方便按名字查找
        self.tools = {t.name: t for t in tools}
        # 调用基类的 get_schema 方法，生成符合 OpenAI 标准的工具描述
        self.tool_schemas = [t.get_schema() for t in tools]

    def _build_system_prompt(self):
        with open("prompts/system.txt", "r", encoding="utf-8") as f:
            return f.read()

    def _build_planner_prompt(self):
        with open("prompts/planner.txt", "r", encoding="utf-8") as f:
            return f.read()

    def run_stream(self, user_input: str, history: list = None):
        if not history: 
            history = []
            
        # 1. 强制读取 Memory (直接在代码层面读取，比让AI自己调工具更稳定)
        memory_context = self.tools["read_worklog"].execute("")

        # 2. 拼装终极 System Prompt
        system_prompt = (
            f"{self._build_system_prompt()}\n\n"
            f"{self._build_planner_prompt()}\n\n"
            f"# 当前用户状态(Memory读取结果):\n{memory_context}"
        )

        messages = [{"role": "system", "content": system_prompt}] + history + [{"role": "user", "content": user_input}]

        final_tool_calls = []
        
        # 3. 开始接收流式数据
        for chunk in self.llm.chat_stream(messages, tools=self.tool_schemas):
            if chunk["type"] == "text":
                # 是文字，直接抛给前端打印
                yield chunk["content"]
            elif chunk["type"] == "tool_calls":
                # 是工具，先存下来
                final_tool_calls = chunk["tool_calls"]

        # 4. 文字流结束了，如果发现有工具要执行
        for tc in final_tool_calls:
            tool_name = tc["name"]
            tool_args = tc["arguments"]
            if tool_name in self.tools:
                try:
                    args_dict = json.loads(tool_args) 
                    # 这里改成提取 "content"
                    exec_result = self.tools[tool_name].execute(args_dict.get("content", ""))
                    yield f"\n[系统: {exec_result}]"
                except Exception as e:
                    yield f"\n[系统: 工具执行解析失败 - {str(e)}]"