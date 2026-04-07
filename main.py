# main.py
import os
import sys
from dotenv import load_dotenv # 需要 pip install python-dotenv

# 1. 启动时加载 .env 文件
load_dotenv()

from llm.glm import GLMAdapter
from tools.file_tool import WorklogTool, ReadWorklogTool
from core.agent import LearningNavigatorAgent


# ---------- 1. LLM接口 ----------
def get_llm():
    """模型路由工厂：根据环境变量决定用哪个模型"""
    provider = os.getenv("LLM_PROVIDER", "glm")
    
    if provider == "glm":
        return GLMAdapter()
    elif provider == "siliconflow":
        from llm.siliconflow import SiliconFlowAdapter
        return SiliconFlowAdapter()
    else:
        raise ValueError(f"不支持的 LLM 提供商: {provider}")


# ---------- 2. Main ----------
def main():
    try:
        llm = get_llm()
    except ValueError as e:
        print(f"初始化失败: {e}")
        sys.exit(1)

    tools = [WorklogTool(), ReadWorklogTool()]
    agent = LearningNavigatorAgent(llm=llm, tools=tools)
    
    print("=== 学习路径导航系统启动 ===")
    history = []
    
    while True:
        user_input = input("\n你: ")
        if user_input.lower() in ['exit', 'quit']:
            break
            
        print("\n导航员: ", end="", flush=True) # 不换行，准备接流式数据
        
        full_response = ""
        # 使用 run_stream 替代 run
        for text_chunk in agent.run_stream(user_input, history):
            print(text_chunk, end="", flush=True) # 来一点打一点
            full_response += text_chunk
            
        print() # 最后换行
        
        history.append({"role": "user", "content": user_input})
        history.append({"role": "assistant", "content": full_response})

if __name__ == "__main__":
    main()