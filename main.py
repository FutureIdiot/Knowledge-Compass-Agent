# main.py
import sys
from dotenv import load_dotenv # 需要 pip install python-dotenv

# 1. 启动时加载 .env 文件
load_dotenv()

from core.orchestra import MultiAgentOrchestrator


# ---------- 2. Main ----------
def main():
    try:
        orchestrator = MultiAgentOrchestrator()
    except ValueError as e:
        print(f"初始化失败: {e}")
        sys.exit(1)
    
    print("=== 多 Agent 学习路径导航系统启动 ===")
    history = []
    
    while True:
        user_input = input("\n你: ")
        if user_input.lower() in ['exit', 'quit']:
            break
            
        print("\n导航员: ", end="", flush=True)
        
        full_response = ""
        for text_chunk in orchestrator.run_stream(user_input, history):
            print(text_chunk, end="", flush=True)
            full_response += text_chunk
            
        print() # 最后换行
        
        history.append({"role": "user", "content": user_input})
        history.append({"role": "assistant", "content": full_response})

if __name__ == "__main__":
    main()
