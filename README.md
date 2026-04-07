# Knowledge_Compass_Agent (学习路径导航系统)
个人知识罗盘，本地学习路径 Agent 系统。

## 🎯 核心理念
通过分层架构（状态机、工具拦截、冷热记忆分离）来控制 LLM 的行为，确保导航过程的客观与可控。

## 🏗️ 总体架构
Agent Kernel = Policy + Memory + Knowledge + Tools + LLM Router + Orchestrator

┌──────────────────────────────┐
│ Interface Layer              │ ← CLI (当前) / API (规划中)
├──────────────────────────────┤
│ Orchestrator Layer           │ ← 多Agent调度 (开发中)
├──────────────────────────────┤
│ Agent Layer                  │ ← 单Agent闭环 ✅ (当前版本)
├──────────────────────────────┤
│ Policy / Planning Layer      │ ← 决策、路径规划 ✅
├──────────────────────────────┤
│ Memory │ Knowledge │ Skills  │ ← 认知系统 
├──────────────────────────────┤
│ Tool Execution Layer         │ ← 文件读写工具 ✅
├──────────────────────────────┤
│ LLM Adapter Layer            │ ← 多模型抽象 ✅
└──────────────────────────────┘

## 🚀 当前状态 (V0.1 - MVP)
- [x] 单 Agent 闭环跑通
- [x] LLM 多模型适配层 (基于环境变量无缝切换)
- [x] 工具调用拦截机制 (防止 LLM 幻觉写入)
- [x] 热记忆层 (基于 Markdown 的 worklog 读写)
- [] 冷知识层
- [] 技能沉淀系统
- [] 多 Agent 编排调度

## ⚡ 快速开始
1. 克隆仓库
2. 安装依赖：pip install -r requirements.txt
3. 配置 .env 文件（参考 .env.example）
4. 运行：python main.py

## 📝 设计原则
Prompt外部化
Agent不直接操作文件，必须通过 Tools
默认不查知识，由 Gate 拦截

## 🙋‍♂️声明
 
坦白说，在写这个项目之前，我完全不懂什么是 Pydantic、Function Calling，甚至连 Python 的面向对象都不太明白。

这个项目里的每一行代码、每一个架构设计，都是我通过和 AI 不断对话，从“写一段神奇的 Prompt”一步步被“逼”成了现在的工程化架构。那些空着的文件夹，是我目前还没学会怎么写，但 AI 告诉我以后应该长在那里的“预留位”。

我把它开源出来，不是为了炫耀技术（因为我真的不懂），而是想记录一个零基础的人，如何依靠 AI 搭建复杂系统的真实过程。

如果你路过了这个仓库，发现了一段极其愚蠢的代码、一个反模式的设计，或者觉得某个架构完全不合规范——请千万不要客气，直接提 Issue 或 PR！你是我的远程导师，我在这里等你批评指正。

（↑ 这些还是ai写的。感谢CLAUDE,GLM,DEEPSEEK,CHATGPT,MINIMAX,GEMINI。是的我就这样问完你的问你的，一个也不放过。）
