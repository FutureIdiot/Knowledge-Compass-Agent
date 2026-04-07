# core/agent.py
from llm.base import BaseLLM

class LearningNavigatorAgent:
    def __init__(self, llm: BaseLLM, tools: list):
        self.llm = llm
        # 将工具列表转为字典，方便按名字查找
        self.tools = {t.name: t for t in tools}
        # 调用基类的 get_schema 方法，生成符合 OpenAI 标准的工具描述
        self.tool_schemas = [t.get_schema() for t in tools]

    # def _build_system_prompt(self):
    #     with open("prompts/system.txt", "r", encoding="utf-8") as f:
    #         return f.read()

    # def _build_planner_prompt(self):
    #     with open("prompts/planner.txt", "r", encoding="utf-8") as f:
    #         return f.read()

    # def run(self, user_input: str, history: list = None) -> str:
    #     if not history:
    #         history = []

    #     # 1. 强制读取 Memory (直接在代码层面读取，比让AI自己调工具更稳定)
    #     memory_context = self.tools["read_worklog"].execute("")

    #     # 2. 拼装终极 System Prompt
    #     system_prompt = (
    #         f"{self._build_system_prompt()}\n\n"
    #         f"{self._build_planner_prompt()}\n\n"
    #         f"# 当前用户状态(Memory读取结果):\n{memory_context}"
    #     )

    #     messages = [{"role": "system", "content": system_prompt}] + history + [{"role": "user", "content": user_input}]

    #     # 3. 调用大模型
    #     try:
    #         llm_response = self.llm.chat(messages, tools=self.tool_schemas)
    #     except Exception as e:
    #         return f"[系统错误: LLM调用失败 - {str(e)}]"

    #     # 4. 处理工具调用 (比如保存状态)
    #     if "tool_calls" in llm_response and llm_response["tool_calls"]:
    #         tool_results = []
    #         for tc in llm_response["tool_calls"]:
    #             tool_name = tc["name"]
    #             tool_args = tc["arguments"]
                
    #             if tool_name in self.tools:
    #                 # 执行工具拿到结果
    #                 exec_result = self.tools[tool_name].execute(tool_args)
    #                 tool_results.append(f"[系统提示: 已执行 {tool_name}，结果: {exec_result}]")

    #         # 将工具执行结果拼接到回复中（MVP阶段简化处理，避免二次请求陷入死循环）
    #         content = llm_response.get("content", "") or ""
    #         if tool_results:
    #             content += "\n\n" + "\n".join(tool_results)
    #         return content if content else "状态已处理，但未生成文本回复。"

    #     # 5. 没有工具调用，直接返回文本
    #     return llm_response.get("content", "抱歉，我无法理解你的意图。")

    # core/agent.py (只改 run 方法部分)
    # ... 前面的 __init__ 和 _build 方法不变 ...

    def run_stream(self, user_input: str, history: list = None):
        if not history: history = []
        memory_context = self.tools["read_worklog"].execute("")
        
        system_prompt = (
            f"{self._build_system_prompt()}\n\n"
            f"{self._build_planner_prompt()}\n\n"
            f"# 当前用户状态:\n{memory_context}"
        )
        messages = [{"role": "system", "content": system_prompt}] + history + [{"role": "user", "content": user_input}]

        final_tool_calls = []
        
        # 开始接收流式数据
        for chunk in self.llm.chat_stream(messages, tools=self.tool_schemas):
            if chunk["type"] == "text":
                # 是文字，直接抛给前端打印
                yield chunk["content"]
            elif chunk["type"] == "tool_calls":
                # 是工具，先存下来
                final_tool_calls = chunk["tool_calls"]

        # 文字流结束了，如果发现有工具要执行
        for tc in final_tool_calls:
            tool_name = tc["name"]
            tool_args = tc["arguments"]
            if tool_name in self.tools:
                exec_result = self.tools[tool_name].execute(tool_args)
                # 把工具执行结果也抛出去显示
                yield f"\n[系统: 已静默执行 {tool_name}]"