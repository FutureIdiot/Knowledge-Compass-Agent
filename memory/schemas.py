import json
from core.base_tool import BaseTool

class WorklogTool(BaseTool):
    name = "save_worklog"
    description = "将用户的最新状态、画像、权重等信息保存到 worklog.md 中。输入必须是 JSON 格式的 AgentState。"

    def execute(self, state_json: str) -> str:
        try:
            # 这里可以做格式化为 markdown 的逻辑
            # 简化版：直接存为 json 以便下次读取
            with open("data/worklog.json", "w", encoding="utf-8") as f:
                f.write(state_json)
            return "状态已成功保存到 worklog。"
        except Exception as e:
            return f"保存失败: {str(e)}"

class ReadWorklogTool(BaseTool):
    name = "read_worklog"
    description = "读取用户当前的学习状态和历史记录。每次对话开始前必须调用。"

    def execute(self, _: str = "") -> str:
        try:
            with open("data/worklog.json", "r", encoding="utf-8") as f:
                return f.read()
        except FileNotFoundError:
            return "未找到历史记录，用户是第一次使用。"