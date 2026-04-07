# tools/file_tool.py
from core.base_tool import BaseTool

class WorklogTool(BaseTool):
    name = "save_worklog"
    description = "将用户的当前学习状态、画像等信息保存到本地日志中。当完成信息收集或状态变更时调用。"

    def execute(self, input_data: str) -> str:
        # 不再做任何复杂的 JSON 校验，拿到什么文本就存什么文本
        if not input_data or input_data.strip() == "":
            return "错误：传入的内容为空。"
        
        try:
            with open("data/worklog.md", "w", encoding="utf-8") as f:
                f.write(input_data)
            return "状态已成功保存到 data/worklog.md。"
        except Exception as e:
            return f"系统错误：保存文件失败 - {str(e)}"

    def get_schema(self) -> dict:
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": {
                    "type": "object",
                    "properties": {
                        "content": {
                            "type": "string",
                            "description": "要写入 worklog.md 的完整 Markdown 格式文本。"
                        }
                    },
                    "required": ["content"]
                }
            }
        }


class ReadWorklogTool(BaseTool):
    name = "read_worklog"
    description = "读取用户当前的 worklog.md 日志内容。"

    def execute(self, input_data: str = "") -> str:
        try:
            with open("data/worklog.md", "r", encoding="utf-8") as f:
                return f.read()
        except FileNotFoundError:
            return "未找到 worklog.md，用户是第一次使用。"