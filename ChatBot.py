import requests
import streamlit as st
from langchain.tools import DuckDuckGoSearchRun
import PyPDF2
from docx import Document
import pandas as pd
import chardet
import base64
import io
from langchain.docstore.document import Document as LC_Document # 新增 langchain 相关依赖
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS  # 替换 Chroma
from langchain.chains import RetrievalQA
from langchain.llms import HuggingFaceHub
import tempfile
import os
from pydub import AudioSegment
from openai import OpenAI
from bs4 import BeautifulSoup
from duckduckgo_search import DDGS
import re
from urllib.parse import urlparse
import pickle
from langchain_community.embeddings import HuggingFaceEmbeddings
from datetime import datetime

# 全局变量定义
CHROMADB_PATH = None
COLLECTION_NAME = "rag_collection"

# 在文件开头添加会话管理相关的初始化
if "chat_history" not in st.session_state:
    st.session_state.chat_history = {}  # 用于存储不同模型的对话历史

def manage_chat_history(model_type, role, content):
    """管理对话历史"""
    if model_type not in st.session_state.chat_history:
        st.session_state.chat_history[model_type] = []
    
    # 添加新消息
    st.session_state.chat_history[model_type].append({
        "role": role,
        "content": content,
        "timestamp": datetime.now().isoformat()
    })
    
    # 保持历史记录在合理范围内（最近10轮对话）
    max_history = 10
    if len(st.session_state.chat_history[model_type]) > max_history * 2:  # 每轮包含用户和助手的消息
        st.session_state.chat_history[model_type] = st.session_state.chat_history[model_type][-max_history * 2:]

def get_chat_history(model_type):
    """获取指定模型的对话历史"""
    return st.session_state.chat_history.get(model_type, [])

def format_messages_for_model(model_type, current_prompt):
    """根据不同模型格式化消息历史"""
    messages = []
    
    # 添加系统提示词
    if st.session_state.selected_assistant:
        domain = next(k for k, v in st.session_state.assistant_market.items() 
                    if st.session_state.selected_assistant in v)
        role_prompt = st.session_state.assistant_market[domain][st.session_state.selected_assistant]
        system_message = {
            "role": "system",
            "content": f"{role_prompt}\n\n请以{st.session_state.selected_assistant}的身份回答问题，保持对话连贯性。"
        }
    else:
        system_message = {
            "role": "system",
            "content": "You are a helpful assistant. Please maintain conversation coherence."
        }
    
    messages.append(system_message)
    
    # 添加历史消息
    history = get_chat_history(model_type)
    for msg in history:
        messages.append({
            "role": msg["role"],
            "content": msg["content"]
        })
    
    # 添加当前问题
    messages.append({
        "role": "user",
        "content": current_prompt
    })
    
    return messages

# ChromaDB 配置函数
def configure_chromadb():
    """配置 ChromaDB 存储路径"""
    st.divider()
    with st.expander("🗄️ RAG知识库设置与管理", expanded=not bool(st.session_state.get("chromadb_path"))):
        st.markdown("### 向量数据库存储路径")
        
        # 默认路径设置
        default_path = os.path.join(os.path.expanduser("~"), "chromadb_data")
        
        # 显示当前路径
        current_path = st.session_state.get("chromadb_path", "")
        if current_path:
            st.info(f"当前路径：{current_path}")
        
        # 路径输入
        new_path = st.text_input(
            "设置存储路径",
            value=current_path or default_path,
            placeholder="例如：/Users/YourName/Documents/chromadb",
            key="chromadb_path_input"
        )
        
        # 确认按钮
        if st.button("✅ 确认路径"):
            try:
                # 确保路径存在
                os.makedirs(new_path, exist_ok=True)
                
                # 测试路径是否可写
                test_file = os.path.join(new_path, "test_write.txt")
                try:
                    with open(test_file, "w") as f:
                        f.write("test")
                    os.remove(test_file)
                except Exception as e:
                    st.error(f"路径无写入权限：{str(e)}")
                    return
                
                # 更新路径
                st.session_state.chromadb_path = new_path
                st.success("✅ 向量数据库路径设置成功！")
                
            except Exception as e:
                st.error(f"路径设置失败：{str(e)}")
        
        # 清空知识库按钮
        if st.button("🗑️ 清空知识库"):
            if clear_vector_store():
                st.success("✅ 知识库已清空")
                st.rerun()
        
        st.markdown("""
        **说明：**
        1. 首次使用请设置存储路径
        2. 路径需要有写入权限
        3. 建议选择本地固定位置
        4. 确保有足够存储空间
        """)

# 初始化会话状态
if "messages" not in st.session_state:
    st.session_state.messages = []
if "search_enabled" not in st.session_state:
    st.session_state.search_enabled = False
if "file_analyzed" not in st.session_state:
    st.session_state.file_analyzed = False
if "file_content" not in st.session_state:
    st.session_state.file_content = ""
if "file_summary" not in st.session_state:
    st.session_state.file_summary = ""
if "selected_model" not in st.session_state:
    st.session_state.selected_model = "豆包"
if "selected_function" not in st.session_state:
    st.session_state.selected_function = "智能问答"
if "api_keys" not in st.session_state:
    st.session_state.api_keys = {}
if "rag_enabled" not in st.session_state:
    st.session_state.rag_enabled = False
if "rag_data" not in st.session_state:
    st.session_state.rag_data = []
if "temperature" not in st.session_state:
    st.session_state.temperature = 0.5
if "max_tokens" not in st.session_state:
    st.session_state.max_tokens = 2048
if "chromadb_path" not in st.session_state:
    st.session_state.chromadb_path = ""
if "vector_store" not in st.session_state:
    st.session_state.vector_store = None
if "selected_assistant" not in st.session_state:
    st.session_state.selected_assistant = None
if "assistant_market" not in st.session_state:
    st.session_state.assistant_market = {
        "金融领域": {
            "财务分析师": """[角色指南] 专业财务分析师，专长领域：
1. 财务报表分析和解读
2. 财务风险评估
3. 企业估值和财务建模
4. 行业对标分析
提供严谨、专业的财务分析建议，注重数据支持和分析依据。""",
            
            "投资策略专家": """[角色指南] 资深投资策略专家，核心能力：
1. 宏观经济分析
2. 资产配置策略
3. 投资组合管理
4. 风险收益评估
基于专业知识提供深入的投资见解。""",
            
            "理财规划师": """[角色指南] 专业理财规划师，专注领域：
1. 个人财务规划
2. 退休计划制定
3. 保险配置建议
4. 税务筹划优化
根据客户情况提供个性化理财方案。""",
            
            "投资顾问专家": """[角色指南] 专业投资顾问，擅长方向：
1. 投资产品分析
2. 市场趋势研判
3. 投资机会识别
4. 风险控制策略
提供专业、负责任的投资建议。""",
            
            "股票分析专家": """[角色指南] 专业股票分析专家，研究领域：
1. 技术面分析
2. 基本面分析
3. 行业研究
4. 个股研究
提供专业的股市分析观点。"""
        },
        "IT领域": {
            "数据科学家": """[角色指南] 资深数据科学家，专业领域：
1. 数据分析和挖掘
2. 机器学习算法
3. 统计建模
4. 数据可视化
专注数据科学解决方案。""",
            
            "全栈程序员": """[角色指南] 资深全栈工程师，技术栈：
1. 前端开发(HTML/CSS/JavaScript)
2. 后端开发(Python/Java/Node.js)
3. 数据库设计
4. 系统架构
提供完整的技术解决方案。""",
            
            "IT架构师": """[角色指南] 专业IT架构师，核心能力：
1. 系统架构设计
2. 技术选型决策
3. 性能优化
4. 安全架构
专注企业级架构设计。""",
            
            "Prompt工程师": """[角色指南] AI交互专家，专长：
1. 提示词优化设计
2. AI交互策略
3. 上下文管理
4. 输出质量控制
专注AI交互效果优化。""",
            
            "数据库管理专家": """[角色指南] 资深数据库专家，专业方向：
1. 数据库设计优化
2. 性能调优
3. 数据安全管理
4. 备份恢复策略
提供数据库专业解决方案。"""
        },
        "商业领域": {
            "产品经理": """[角色指南] 资深产品经理，专业领域：
1. 产品战略规划
2. 用户需求分析
3. 产品生命周期管理
4. 产品路线图制定
5. 跨团队协作管理
专注产品战略与落地。""",
            
            "供应链策略专家": """[角色指南] 供应链管理专家，核心能力：
1. 供应链优化设计
2. 库存管理策略
3. 物流网络规划
4. 供应商管理
5. 风险评估与控制
提供专业供应链解决方案。""",
            
            "数字营销专家": """[角色指南] 数字营销专家，专长领域：
1. 数字营销策略制定
2. 社交媒体营销
3. 内容营销策略
4. 用户增长策略
5. ROI分析与优化
专注数字营销效果提升。""",
            
            "人力专家": """[角色指南] 人力资源专家，专业方向：
1. 人才招聘与培养
2. 绩效管理体系
3. 薪酬福利设计
4. 组织发展规划
5. 员工关系管理
提供专业人力资源解决方案。""",
            
            "社交媒体经理": """[角色指南] 社交媒体运营专家，核心能力：
1. 社交媒体策略规划
2. 内容创作与管理
3. 社区运营与互动
4. 舆情监测与危机处理
5. 影响力数据分析
专注社交媒体效果优化。"""
        },
        "咨询领域": {
            "麦肯锡顾问": """[角色指南] 战略咨询专家，专业领域：
1. 战略咨询
2. 组织转型
3. 运营优化
4. 数字化转型
5. 商业模式创新
采用专业咨询方法论。""",
            
            "行业调研专家": """[角色指南] 行业研究专家，研究方向：
1. 市场规模测算
2. 竞争格局分析
3. 产业链研究
4. 发展趋势预测
5. 商业机会识别
提供深度行业洞察。""",
            
            "战略分析师": """[角色指南] 战略分析专家，核心能力：
1. 战略规划制定
2. 商业模式分析
3. 市场进入策略
4. 竞争战略分析
5. 战略实施路径
专注战略规划与执行。""",
            
            "企业策略专家": """[角色指南] 企业战略专家，专业方向：
1. 企业发展战略
2. 业务组合优化
3. 投资并购策略
4. 风险管理策略
5. 组织变革管理
提供全面战略建议。""",
            
            "法律顾问专家": """[角色指南] 商业法律专家，专业领域：
1. 商业合同审查
2. 知识产权保护
3. 企业合规管理
4. 风险法律评估
5. 争议解决方案
提供专业法律咨询。"""
        },
        "学术领域": {
            "经济学教授": """[角色指南] 资深经济学教授，研究领域：
1. 宏观经济理论
2. 经济政策分析
3. 国际经济学
4. 发展经济学
5. 行为经济学
专注经济学理论研究与实证分析，提供严谨的学术见解。""",
            
            "金融学教授": """[角色指南] 资深金融学教授，专业方向：
1. 公司金融理论
2. 资产定价模型
3. 金融市场研究
4. 金融工程学
5. 行为金融学
提供深入的金融学理论分析与研究方法指导。""",
            
            "统计学家": """[角色指南] 专业统计学家，研究领域：
1. 数理统计
2. 实验设计
3. 多元统计分析
4. 时间序列分析
5. 贝叶斯统计
专注统计方法论与数据分析，提供严谨的统计建议。""",
            
            "历史学家": """[角色指南] 资深历史学家，研究方向：
1. 历史事件分析
2. 历史文献考证
3. 历史比较研究
4. 文化史研究
5. 经济史研究
提供专业的历史学视角与研究方法指导。""",
            
            "期刊审稿人": """[角色指南] 专业学术期刊审稿人，针对提交的学术论文进行全面评审，重点关注：
1. 研究方法评估：分析所采用方法的适用性与严谨性；评估数据收集与分析过程的可靠性
2. 学术创新性判断：判断研究的新颖性及其在领域内的独特贡献；评估研究是否填补现有知识空白或提出新的理论框架
3. 文献综述审查：检查文献引用的全面性、相关性及新颖性；评估文献综述是否有效支持研究背景与目的
4. 实验设计评价：评估实验设计的合理性、可重复性及控制变量的有效性；检查实验方法是否详尽描述，便于他人复现
5. 研究结论验证：确认结论是否由数据充分支持，逻辑是否严密；评估结论的科学性与实际应用价值
提供结构清晰、建设性的改进建议，指出论文中的语言、格式及结构性问题，保持评审意见客观、公正。""",
            
            "课题申报指导": """[角色指南] 国家社科/自科基金评审级顾问，提供全流程精细化指导：
1.选题论证：聚焦理论空白与实践需求，分析国内外研究动态，指导创新切入点筛选与标题打磨
2.方案设计：构建"理论框架-技术路线-实验设计"三位一体方案，强调方法论科学性（如混合研究设计、纵向追踪模型）
3.成果规划：区分理论突破（新模型/范式）与实践价值（政策建议/技术原型），规划专利/数据库等实体成果转化路径
4.文本优化：指导文献综述批判性写作、技术路线甘特图可视化、研究基础与课题衔接策略
5.预算编制：按设备费/测试费/会议费分类编制，指导间接费用合规性测算与绩效支出占比
6.答辩预审：模拟评审视角，针对学科代码选择、团队结构合理性、预期成果可行性等12项常见否决点进行风险诊断
全程提供学科差异化的申报策略（如文科强调理论创新，工科侧重应用验证），协助构建"问题驱动-方法创新-价值闭环"的申报逻辑体系。""",
            
            "课题评审专家": """[角色指南] 资深的课题评审专家，拥有深厚的学术背景和丰富的项目评审经验，评审重点：
1. 选题价值评估：分析课题在当前学术前沿或实际应用中的重要性、必要性及其潜在的社会经济影响
2. 研究方案可行性：审查研究设计的合理性与科学性，包括研究方法、技术路线、时间安排及资源配置是否切实可行
3. 创新性分析：评估课题的新颖性和独创性，明确其相较于现有研究的突破点和独特贡献
4. 研究基础评价：考察申报团队的研究背景、专业能力及以往相关成果，确保团队具备完成课题的实力与经验
5. 预期成果评估：预测研究可能取得的成果及其学术价值或实际应用前景，评估成果的可推广性和影响力
请基于以上评审重点，提供详尽、客观且具有建设性的评审意见与改进建议，语言应严谨专业，逻辑清晰。"""
        }
    }

# 页面配置
st.set_page_config(page_title="多模型智能助手2.10(学术增强版)", layout="wide")

# 初始化/加载 langchain 封装的 Chroma 向量库
def get_vector_store():
    """获取向量数据库实例"""
    try:
        # 检查是否已设置路径
        if not st.session_state.get("chromadb_path"):
            st.error("⚠️ 请先在侧边栏设置向量数据库存储路径！")
            return None
            
        # 如果向量库已经在会话状态中，直接返回
        if st.session_state.get("vector_store") is not None:
            return st.session_state.vector_store
            
        # 使用用户设置的路径
        db_path = os.path.join(st.session_state.chromadb_path, "faiss_index")
        
        # 获取 embeddings
        embeddings = get_embeddings()
        if not embeddings:
            return None
            
        # 如果存在现有索引，则加载
        if os.path.exists(db_path):
            try:
                vectorstore = FAISS.load_local(
                    db_path, 
                    embeddings,
                    allow_dangerous_deserialization=True  # 添加此参数
                )
                st.session_state.vector_store = vectorstore
                return vectorstore
            except Exception as e:
                st.error(f"加载向量库失败：{str(e)}")
                return None
        
        # 如果不存在，创建新的向量库实例
        vectorstore = FAISS.from_texts(
            texts=["初始化文档"],
            embedding=embeddings
        )
        st.session_state.vector_store = vectorstore
        return vectorstore
        
    except Exception as e:
        st.error(f"初始化向量库失败：{str(e)}")
        import traceback
        st.error(f"详细错误：{traceback.format_exc()}")
        return None

# 初始化 DuckDuckGo 搜索工具
search_tool = DuckDuckGoSearchRun()

# 核心功能实现

def handle_web_search(query):
    """联网搜索功能，使用 DuckDuckGo API"""
    if not st.session_state.search_enabled:
        return None
    try:
        search = DuckDuckGoSearchRun()
        results = search.run(query)
        return results
    except Exception as e:
        st.error(f"联网搜索失败: {str(e)}")
        return None

def call_model_api(prompt, model_type, rag_data=None):
    """调用除 RAG 部分外的其他接口"""
    headers = {"Content-Type": "application/json"}
    
    try:
        # 获取格式化后的消息列表
        messages = format_messages_for_model(model_type, prompt)
        
        if model_type == "豆包":
            api_key = st.session_state.api_keys.get("豆包", "")
            if not api_key:
                st.error("请提供豆包 API 密钥！")
                return None
            headers["Authorization"] = f"Bearer {api_key}"
            
            response = requests.post(
                "https://ark.cn-beijing.volces.com/api/v3/chat/completions",
                json={
                    "model": "ep-20250128163906-p4tb5",
                    "messages": messages,
                    "temperature": st.session_state.temperature,
                    "max_tokens": st.session_state.max_tokens
                },
                headers=headers
            )
            result = handle_response(response, rag_data)
            if result:
                manage_chat_history(model_type, "assistant", result)
            return result

        elif model_type == "DeepSeek-V3":
            api_key = st.session_state.api_keys.get("DeepSeek", "")
            if not api_key:
                st.error("请提供 DeepSeek API 密钥！")
                return None
            headers["Authorization"] = f"Bearer {api_key}"
            
            response = requests.post(
                "https://api.deepseek.com/v1/chat/completions",
                json={
                    "model": "deepseek-chat",
                    "messages": messages,
                    "temperature": st.session_state.temperature,
                    "max_tokens": st.session_state.max_tokens
                },
                headers=headers
            )
            result = handle_response(response, rag_data)
            if result:
                manage_chat_history(model_type, "assistant", result)
            return result

        elif model_type == "通义千问":
            api_key = st.session_state.api_keys.get("通义千问", "")
            if not api_key:
                st.error("请提供 通义千问 API 密钥！")
                return None
            headers["Authorization"] = f"Bearer {api_key}"
            
            response = requests.post(
                "https://dashscope.aliyuncs.com/compatible-mode/v1/chat/completions",
                json={
                    "model": "qwen-plus",
                    "messages": messages,
                    "temperature": st.session_state.temperature,
                    "max_tokens": st.session_state.max_tokens
                },
                headers=headers
            )
            result = handle_response(response, rag_data)
            if result:
                manage_chat_history(model_type, "assistant", result)
            return result

        elif model_type == "文心一言":
            api_key = st.session_state.api_keys.get("文心一言", "")
            if not api_key:
                st.error("请提供 文心一言 API 密钥！")
                return None
            headers["Authorization"] = f"Bearer {api_key}"
            # 为文心一言构建增强的提示词
            enhanced_prompt = prompt
            if st.session_state.selected_assistant:
                domain = next(k for k, v in st.session_state.assistant_market.items() 
                            if st.session_state.selected_assistant in v)
                role_prompt = st.session_state.assistant_market[domain][st.session_state.selected_assistant]
                enhanced_prompt = f"{role_prompt}\n\n请以{st.session_state.selected_assistant}的身份回答以下问题：\n{prompt}"
            # 获取历史消息
            history = get_chat_history(model_type)
            if history:
                # 将最近的对话历史添加到提示词中
                recent_history = history[-6:]  # 保留最近3轮对话
                history_text = "\n".join([f"{'用户' if msg['role']=='user' else '助手'}: {msg['content']}" 
                                        for msg in recent_history])
                enhanced_prompt = f"以下是历史对话：\n{history_text}\n\n当前问题：{enhanced_prompt}"
            response = requests.post(
                "https://aip.baidubce.com/rpc/2.0/ai_custom/v1/wenxinworkshop/chat/completions",
                json={
                    "model": "ERNIE-Bot",
                    "messages": [{"role": "user", "content": enhanced_prompt}],
                    "temperature": st.session_state.temperature,
                    "max_tokens": st.session_state.max_tokens
                },
                headers=headers
            )
            result = handle_response(response, rag_data)
            if result:
                manage_chat_history(model_type, "assistant", result)
            return result

        elif model_type == "智谱清言":
            api_key = st.session_state.api_keys.get("智谱清言", "")
            if not api_key:
                st.error("请提供 智谱清言 API 密钥！")
                return None
            headers["Authorization"] = f"Bearer {api_key}"
            
            response = requests.post(
                "https://open.bigmodel.cn/api/paas/v4/chat/completions",
                json={
                    "model": "glm-4",
                    "messages": messages,
                    "temperature": st.session_state.temperature,
                    "max_tokens": st.session_state.max_tokens
                },
                headers=headers
            )
            result = handle_response(response, rag_data)
            if result:
                manage_chat_history(model_type, "assistant", result)
            return result

        elif model_type == "MiniMax":
            api_key = st.session_state.api_keys.get("MiniMax", "")
            if not api_key:
                st.error("请提供 MiniMax API 密钥！")
                return None
            headers["Authorization"] = f"Bearer {api_key}"
            
            response = requests.post(
                "https://api.minimax.chat/v1/text/chatcompletion_v2",
                json={
                    "model": "abab5.5-chat",
                    "messages": messages,
                    "temperature": st.session_state.temperature,
                    "max_tokens": st.session_state.max_tokens
                },
                headers=headers
            )
            result = handle_response(response, rag_data)
            if result:
                manage_chat_history(model_type, "assistant", result)
            return result

        elif model_type == "DALL-E(文生图)":
            api_key = st.session_state.api_keys.get("OpenAI", "")
            if not api_key:
                st.error("请提供 DALL-E(文生图) API 密钥！")
                return None
            headers["Authorization"] = f"Bearer {st.session_state.api_keys['OpenAI']}"
            
            response = requests.post(
                "https://api.openai.com/v1/images/generations",
                json={
                    "prompt": prompt,
                    "n": 1,
                    "size": "512x512"
                },
                headers=headers
            )
            response_json = response.json()
            if "data" in response_json and len(response_json["data"]) > 0:
                image_url = response_json["data"][0]["url"]
                return image_url
            else:
                st.error(f"DALL-E API 返回格式异常: {response_json}")
                return None

        elif model_type == "DeepSeek-R1(深度推理)":
            api_key = st.session_state.api_keys.get("DeepSeek", "")
            if not api_key:
                st.error("请提供 DeepSeek API 密钥！")
                return None
            headers["Authorization"] = f"Bearer {api_key}"
            
            # 构建增强的提示词
            enhanced_prompt = prompt
            if st.session_state.selected_assistant:
                domain = next(k for k, v in st.session_state.assistant_market.items() 
                            if st.session_state.selected_assistant in v)
                role_prompt = st.session_state.assistant_market[domain][st.session_state.selected_assistant]
                enhanced_prompt = f"{role_prompt}\n\n请以{st.session_state.selected_assistant}的身份回答以下问题：\n{prompt}"
            
            # 获取历史消息
            history = get_chat_history(model_type)
            if history:
                # 将最近的对话历史添加到提示词中
                recent_history = history[-6:]  # 保留最近3轮对话
                history_text = "\n".join([f"{'用户' if msg['role']=='user' else '助手'}: {msg['content']}" 
                                        for msg in recent_history])
                enhanced_prompt = f"以下是历史对话：\n{history_text}\n\n当前问题：{enhanced_prompt}"
            
            response = requests.post(
                "https://api.deepseek.com/v1/chat/completions",
                json={
                    "model": "deepseek-reasoner",
                    "messages": [{"role": "user", "content": enhanced_prompt}],
                    "temperature": st.session_state.temperature,
                    "max_tokens": st.session_state.max_tokens
                },
                headers=headers
            )
            result = handle_response(response, rag_data)
            if result:
                manage_chat_history(model_type, "assistant", result)
            return result

        elif model_type == "o1(深度推理)":
            api_key = st.session_state.api_keys.get("OpenAI", "")
            if not api_key:
                st.error("请提供 o1 API 密钥！")
                return None
            headers["Authorization"] = f"Bearer {api_key}"
            
            # 构建增强的提示词
            enhanced_prompt = prompt
            if st.session_state.selected_assistant:
                domain = next(k for k, v in st.session_state.assistant_market.items() 
                            if st.session_state.selected_assistant in v)
                role_prompt = st.session_state.assistant_market[domain][st.session_state.selected_assistant]
                enhanced_prompt = f"{role_prompt}\n\n请以{st.session_state.selected_assistant}的身份回答以下问题：\n{prompt}"
            
            # 获取历史消息
            history = get_chat_history(model_type)
            if history:
                # 将最近的对话历史添加到提示词中
                recent_history = history[-6:]  # 保留最近3轮对话
                history_text = "\n".join([f"{'用户' if msg['role']=='user' else '助手'}: {msg['content']}" 
                                        for msg in recent_history])
                enhanced_prompt = f"以下是历史对话：\n{history_text}\n\n当前问题：{enhanced_prompt}"
            
            response = requests.post(
                "https://api.openai.com/v1/chat/completions",
                json={
                    "model": "o1-mini",
                    "messages": [{"role": "user", "content": enhanced_prompt}],
                    "max_completion_tokens": st.session_state.max_tokens
                },
                headers=headers
            )
            result = handle_response(response, rag_data)
            if result:
                manage_chat_history(model_type, "assistant", result)
            return result

        elif model_type == "Kimi(视觉理解)":
            api_key = st.session_state.api_keys.get("Kimi(视觉理解)", "")
            if not api_key:
                st.error("请提供 Kimi(视觉理解) API 密钥！")
                return None
            headers["Authorization"] = f"Bearer {api_key}"
            
            response = requests.post(
                "https://api.moonshot.cn/v1/chat/completions",
                json={
                    "model": "moonshot-v1-8k-vision-preview",
                    "messages": messages
                },
                headers=headers
            )
            result = handle_response(response)
            if result:
                manage_chat_history(model_type, "assistant", result)
            return result

        elif model_type == "GPTs(聊天、语音识别)":
            api_key = st.session_state.api_keys.get("OpenAI", "")
            if not api_key:
                st.error("请提供 OpenAI API 密钥！")
                return None
            headers["Authorization"] = f"Bearer {api_key}"
            
            response = requests.post(
                "https://api.openai.com/v1/chat/completions",
                json={
                    "model": "gpt-4o",
                    "messages": messages,
                    "temperature": st.session_state.temperature,
                    "max_tokens": st.session_state.max_tokens
                },
                headers=headers
            )
            result = handle_response(response, rag_data)
            if result:
                manage_chat_history(model_type, "assistant", result)
            return result

        elif model_type == "grok2":
            api_key = st.session_state.api_keys.get("xAI", "")
            if not api_key:
                st.error("请提供 xAI API 密钥！")
                return None
            headers["Authorization"] = f"Bearer {api_key}"
            
            response = requests.post(
                "https://api.x.ai/v1/chat/completions",
                json={
                    "model": "grok-2-latest",
                    "messages": messages,
                    "temperature": st.session_state.temperature,
                    "max_tokens": st.session_state.max_tokens
                },
                headers=headers
            )
            result = handle_response(response, rag_data)
            if result:
                manage_chat_history(model_type, "assistant", result)
            return result

        elif model_type == "混元生文":
            api_key = st.session_state.api_keys.get("混元生文", "")
            if not api_key:
                st.error("请提供腾讯混元 API 密钥！")
                return None
            
            try:
                client = OpenAI(
                    api_key=api_key,
                    base_url="https://api.hunyuan.cloud.tencent.com/v1"
                )
                
                response = client.chat.completions.create(
                    model="hunyuan-turbo",
                    messages=messages,
                    temperature=st.session_state.temperature,
                    max_tokens=st.session_state.max_tokens
                )
                
                if response.choices:
                    result = response.choices[0].message.content
                    manage_chat_history(model_type, "assistant", result)
                    return result
                else:
                    st.error("API 返回格式异常")
                    return None
            
            except Exception as e:
                st.error(f"调用混元模型时出错：{str(e)}")
                return None

        else:
            # 默认调用使用 RAG 生成答案
            result = rag_generate_response(prompt)
            if result:
                manage_chat_history(model_type, "assistant", result)
            return result

    except Exception as e:
        st.error(f"API调用失败: {str(e)}")
        return None

def handle_response(response, rag_data=None):
    """处理 API 响应"""
    try:
        if response.status_code == 200:
            response_json = response.json()
            if "choices" in response_json and len(response_json["choices"]) > 0:
                answer = response_json["choices"][0]["message"]["content"]
            elif "result" in response_json:
                # 针对文心一言返回格式处理
                answer = response_json["result"]
            elif "data" in response_json and isinstance(response_json["data"], list) and len(response_json["data"]) > 0:
                # 针对 DALL-E 返回格式处理
                if "url" in response_json["data"][0]:
                    answer = response_json["data"][0]["url"]
                else:
                    st.error(f"API 返回格式异常: {response_json}")
                    return None
            else:
                st.error(f"API 返回格式异常: {response_json}")
                return None

            if rag_data and isinstance(answer, str):  # 确保是文本才添加引用
                answer += "\n\n引用来源：\n" + "\n".join([f"- {source}" for source in rag_data])
            return answer
        elif response.status_code == 503:
            st.error("服务器繁忙，请稍后再试。")
            return None
        else:
            st.error(f"API 请求失败，错误码：{response.status_code}")
            return None
    except ValueError as e:
        st.error(f"响应解析失败: {str(e)}")
        return None

# 使用 langchain 实现 RAG：加载文档、分割、嵌入、索引
def rag_index_document(content, source):
    """将文档添加到向量数据库"""
    try:
        # 检查存储路径
        if not st.session_state.get("chromadb_path"):
            st.error("⚠️ 请先在 RAG知识库设置与管理 中设置存储路径！")
            return False
            
        # 检查内容
        if not content or not isinstance(content, str):
            st.error("⚠️ 文档内容为空或格式不正确")
            return False
            
        # 清理文本内容
        content = clean_text(content)
        if not content:
            st.error("⚠️ 清理后的文本内容为空")
            return False
            
        # 文本分割
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=500,
            chunk_overlap=50,
            length_function=len,
        )
        texts = text_splitter.split_text(content)
        
        if not texts:
            st.error("⚠️ 文本分割后为空")
            return False
            
        # 限制文本块数量
        max_chunks = 100
        if len(texts) > max_chunks:
            st.warning(f"文档过大，将只处理前 {max_chunks} 个文本块")
            texts = texts[:max_chunks]
            
        # 获取向量库实例
        vectorstore = get_vector_store()
        if not vectorstore:
            return False
        
        try:
            # 分批添加文档
            batch_size = 20
            for i in range(0, len(texts), batch_size):
                batch_texts = texts[i:i+batch_size]
                batch_metadatas = [{"source": source} for _ in batch_texts]
                vectorstore.add_texts(
                    texts=batch_texts,
                    metadatas=batch_metadatas
                )
            
            # 保存向量库
            save_path = os.path.join(st.session_state.chromadb_path, "faiss_index")
            vectorstore.save_local(save_path)
            
            # 更新会话状态
            st.session_state.vector_store = vectorstore
            if source not in st.session_state.rag_data:
                st.session_state.rag_data.append(source)
            st.success(f"✅ 成功添加 {len(texts)} 个文本块到知识库")
            return True
            
        except Exception as e:
            st.error(f"添加文档失败：{str(e)}")
            return False
            
    except Exception as e:
        st.error(f"❌ 处理文档失败：{str(e)}")
        import traceback
        st.error(f"详细错误：{traceback.format_exc()}")
        return False

def rag_generate_response(query):
    """生成 RAG 响应"""
    try:
        # 获取向量库实例
        vectorstore = get_vector_store()
        if not vectorstore:
            return "请先上传文件或网址到知识库。"
        
        # 限制相似性搜索的数量
        k = 2  # 减少返回的文档数量
        
        try:
            # 执行相似性搜索
            docs = vectorstore.similarity_search(query, k=k)
            
            if not docs:
                return "未找到相关信息。请尝试调整问题或添加更多相关文档。"
            
            # 限制上下文长度
            max_context_length = 2000
            context = "\n\n".join([doc.page_content[:max_context_length] for doc in docs])
            sources = "\n".join([f"- {doc.metadata.get('source', '未知来源')}" for doc in docs])
            
            # 构建提示词
            prompt = f"""基于以下参考信息回答问题。如果参考信息不足以回答问题，请明确说明。

参考信息：
{context}

问题：{query}

请提供准确、相关的回答。
"""
            # 调用模型生成回答
            response = call_model_api(prompt, st.session_state.selected_model)
            if response:
                return f"{response}\n\n来源：\n{sources}"
            return "生成回答失败，请重试。"
            
        except Exception as e:
            st.error(f"搜索相关文档失败：{str(e)}")
            return "处理查询时出错，请重试。"
            
    except Exception as e:
        st.error(f"❌ 生成回答失败：{str(e)}")
        import traceback
        st.error(f"详细错误：{traceback.format_exc()}")
        return None

def handle_file_upload(uploaded_files):
    """处理上传文件，根据 RAG 状态及文件类型执行不同操作：
       - RAG 模式下：文本、表格类文件加入知识库；
       - 非 RAG 模式下：
           图片文件 -> 视觉分析
           语音文件 -> 语音识别
           文本文件 -> 文本总结
    """
    if not uploaded_files:
        return

    if not isinstance(uploaded_files, list):
        uploaded_files = [uploaded_files]

    for uploaded_file in uploaded_files:
        if not hasattr(uploaded_file, "name"):
            st.error("上传的文件格式错误，缺少名称属性。")
            continue

        file_name = uploaded_file.name
        file_type = uploaded_file.type.split("/")[-1].lower()
        try:
            if st.session_state.rag_enabled:
                # RAG 模式下，仅处理文本、表格类文件加入知识库
                if file_type in ["txt", "pdf", "docx", "doc", "csv", "xlsx", "xls"]:
                    content = extract_text_from_file(uploaded_file)
                    if content:
                        if rag_index_document(content, file_name):
                            st.session_state.rag_data.append(file_name)
                            st.success(f"文件 {file_name} 已成功加入 RAG 知识库")
                else:
                    st.warning(f"RAG 模式下，文件 {file_name} 的类型（{file_type}）不支持加入知识库。")
            else:
                # 非 RAG 模式，根据文件类型调用对应功能
                if file_type in ["jpg", "jpeg", "png"]:
                    with st.spinner("🖼️ 正在分析图片..."):
                        if st.session_state.selected_model == "Kimi(视觉理解)":  # 只使用 Kimi 进行视觉理解
                            if "Kimi(视觉理解)" not in st.session_state.api_keys:
                                st.error("请先配置 Kimi(视觉理解) API 密钥")
                            else:
                                image_content = uploaded_file.getvalue()
                                encoded_image = base64.b64encode(image_content).decode('utf-8')
                                
                                headers = {
                                    "Content-Type": "application/json",
                                    "Authorization": f"Bearer {st.session_state.api_keys['Kimi(视觉理解)']}"
                                }
                                
                                payload = {
                                    "model": "moonshot-v1-8k-vision-preview",
                                    "messages": [
                                        {
                                            "role": "user",
                                            "content": [
                                                {
                                                    "type": "text",
                                                    "text": "请详细分析这张图片的内容，包括主要对象、场景、细节等方面。"
                                                },
                                                {
                                                    "type": "image_url",
                                                    "image_url": {
                                                        "url": f"data:image/jpeg;base64,{encoded_image}"
                                                    }
                                                }
                                            ]
                                        }
                                    ]
                                }
                                
                                response = requests.post(
                                    "https://api.moonshot.cn/v1/chat/completions",
                                    json=payload,
                                    headers=headers
                                )
                                
                                if response.status_code == 200:
                                    result = response.json()["choices"][0]["message"]["content"]
                                    st.success("✅ 图片分析完成")
                                    with st.chat_message("assistant"):
                                        st.markdown(f"**图片分析结果：**\n\n{result}")
                                    st.session_state.messages.append({
                                        "role": "assistant",
                                        "content": f"图片 {uploaded_file.name} 的分析结果：\n\n{result}",
                                        "type": "text"
                                    })
                                else:
                                    st.error(f"❌ 图片分析失败：{response.text}")
                elif file_type in ["mp3", "wav", "m4a", "mpeg"]:
                    st.write(f"正在进行语音识别：{file_name}")
                    speech_result = perform_speech_recognition(uploaded_file.getvalue())
                    if speech_result:
                        st.write("语音识别结果：")
                        st.write(speech_result)
                        st.session_state.messages.append({
                            "role": "assistant",
                            "content": f"语音识别结果：\n{speech_result}",
                            "type": "text"
                        })
                elif file_type in ["txt", "pdf", "docx", "doc"]:
                    content = extract_text_from_file(uploaded_file)
                    if content:
                        st.write(f"正在总结文本：{file_name}")
                        summary_result = perform_text_summary(content)
                        if summary_result:
                            st.write("文本总结结果：")
                            st.write(summary_result)
                            st.session_state.messages.append({
                                "role": "assistant",
                                "content": f"文本总结结果：\n{summary_result}",
                                "type": "text"
                            })
                else:
                    st.warning(f"文件 {file_name} 的类型（{file_type}）不支持处理。")
        except Exception as e:
            st.error(f"文件处理失败 ({file_name}): {str(e)}")

def extract_text_from_file(file):
    """从不同类型的文件中提取文本内容"""
    try:
        file_type = file.name.split('.')[-1].lower()
        content = file.read()
        
        if file_type == 'txt':
            # 处理文本文件
            return content.decode('utf-8')
        elif file_type == 'pdf':
            # 处理 PDF 文件
            pdf_reader = PyPDF2.PdfReader(io.BytesIO(content))
            return "\n".join([page.extract_text() for page in pdf_reader.pages])
        elif file_type in ['docx', 'doc']:
            # 处理 Word 文件
            doc = Document(io.BytesIO(content))
            return "\n".join([para.text for para in doc.paragraphs])
        elif file_type in ['csv', 'xlsx', 'xls']:
            # 处理表格文件
            if file_type == 'csv':
                df = pd.read_csv(io.BytesIO(content))
            else:
                df = pd.read_excel(io.BytesIO(content))
            return df.to_string()
        else:
            st.warning(f"不支持的文件类型：{file_type}")
            return None
    except Exception as e:
        st.error(f"处理文件失败：{str(e)}")
        return None

def perform_speech_recognition(audio_bytes):
    """
    使用当前选择的模型进行语音识别
    """
    api_key = st.session_state.api_keys.get("OpenAI", "")
    if not api_key:
        st.error("请提供 OpenAI API 密钥以进行语音识别！")
        return None
    
    try:
        # 创建 OpenAI 客户端
        client = OpenAI(api_key=api_key)
        
        # 将音频数据转换为临时文件
        with tempfile.NamedTemporaryFile(delete=False, suffix='.wav') as temp_file:
            temp_file.write(audio_bytes)
            temp_file_path = temp_file.name
        
        with open(temp_file_path, 'rb') as audio_file:
            transcript = client.audio.transcriptions.create(
                model="whisper-1",
                file=audio_file
            )
        
        os.unlink(temp_file_path)  # 删除临时文件
        return transcript.text
        
    except Exception as e:
        st.error(f"语音识别失败：{str(e)}")
        return None

def perform_text_summary(text):
    """
    使用当前选择的模型对文本进行总结
    """
    try:
        summary_prompt = f"请对以下文本进行简明扼要的总结：\n\n{text}"
        response = call_model_api(summary_prompt, st.session_state.selected_model)
        return response
    except Exception as e:
        st.error(f"文本总结失败：{str(e)}")
        return None

def retrieve_relevant_content(query):
    """
    利用 langchain 封装的向量库检索与查询相关的文档，
    返回包含来源信息的列表。
    """
    vectorstore = get_vector_store()
    try:
        results = vectorstore.similarity_search(query, k=3)
    except Exception as e:
        st.error(f"检索时出现错误: {str(e)}")
        return []
    # 提取文档 metadata 中的 "source" 信息；如果不存在则返回 "未知来源"
    return [doc.metadata.get("source", "未知来源") for doc in results]

def fetch_url_content(url):
    """获取网页内容并提取有效文本"""
    try:
        # 添加请求头，模拟浏览器访问
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        }
        response = requests.get(url, headers=headers, verify=False, timeout=10)
        response.raise_for_status()
        
        # 使用 BeautifulSoup 提取文本内容
        soup = BeautifulSoup(response.content, 'html.parser')
        
        # 移除脚本和样式元素
        for script in soup(["script", "style"]):
            script.decompose()
        
        # 获取文本并处理
        text = soup.get_text()
        lines = (line.strip() for line in text.splitlines())
        chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
        text = ' '.join(chunk for chunk in chunks if chunk)
        
        return text
    except Exception as e:
        st.error(f"获取网页内容失败：{str(e)}")
        return None

def clear_vector_store():
    """清空向量数据库"""
    try:
        # 检查路径是否存在
        if not st.session_state.get("chromadb_path"):
            st.error("未设置向量库存储路径")
            return False
            
        db_path = st.session_state.chromadb_path
        if os.path.exists(db_path):
            import shutil
            shutil.rmtree(db_path)
            # 重新创建目录
            os.makedirs(db_path, exist_ok=True)
            
        st.session_state.vector_store = None
        st.session_state.rag_data = []
        st.success("✅ 知识库已清空")
        st.rerun()
        return True
    except Exception as e:
        st.error(f"清空知识库失败：{str(e)}")
        return False

def clean_text(text):
    """清理文本内容"""
    if not text:
        return ""
    # 移除多余的空白字符
    text = re.sub(r'\s+', ' ', text).strip()
    # 移除特殊字符，保留中文、英文、数字和基本标点
    text = re.sub(r'[^\w\s\u4e00-\u9fff,.?!，。？！:：;；""''()（）《》<>]', '', text)
    return text

def is_financial_domain(url):
    """判断是否为财经金融相关的高质量域名"""
    try:
        domain = urlparse(url).netloc.lower()
        
        # 财经金融网站优先级
        financial_domains = {
            # 官方机构
            'pbc.gov.cn': 10,     # 中国人民银行
            'csrc.gov.cn': 10,    # 中国证监会
            'safe.gov.cn': 10,    # 外汇管理局
            'stats.gov.cn': 10,   # 国家统计局
            'mof.gov.cn': 10,     # 财政部
            
            # 交易所
            'sse.com.cn': 9,      # 上海证券交易所
            'szse.cn': 9,         # 深圳证券交易所
            'cffex.com.cn': 9,    # 中国金融期货交易所
            
            # 金融门户网站
            'eastmoney.com': 8,   # 东方财富
            'finance.sina.com.cn': 8,  # 新浪财经
            'caixin.com': 8,      # 财新网
            'yicai.com': 8,       # 第一财经
            'stcn.com': 8,        # 证券时报网
            'cnstock.com': 8,     # 中国证券网
            '21jingji.com': 8,    # 21世纪经济网
            
            # 财经媒体
            'bloomberg.cn': 8,     # 彭博
            'ftchinese.com': 8,   # FT中文网
            'nbd.com.cn': 7,      # 每日经济新闻
            'ce.cn': 7,           # 中国经济网
            'jrj.com.cn': 7,      # 金融界
            'hexun.com': 7,       # 和讯网
            
            # 研究机构
            'cfets.org.cn': 7,    # 中国外汇交易中心
            'chinabond.com.cn': 7, # 中国债券信息网
            'shibor.org': 7,      # Shibor官网
            
            # 国际金融网站
            'reuters.com': 8,      # 路透社
            'bloomberg.com': 8,    # 彭博
            'wsj.com': 8,         # 华尔街日报
            'ft.com': 8,          # 金融时报
            'economist.com': 8,    # 经济学人
            
            # 其他相关网站
            'investing.com': 7,    # 英为财情
            'marketwatch.com': 7,  # 市场观察
            'cnfol.com': 6,       # 中金在线
            'stockstar.com': 6,   # 证券之星
            '10jqka.com.cn': 6,   # 同花顺财经
        }
        
        # 检查域名优先级
        for known_domain, priority in financial_domains.items():
            if known_domain in domain:
                return priority
                
        return 0  # 非金融网站返回0优先级
    except:
        return 0

def perform_web_search(query, max_results=10):
    """执行优化的财经金融搜索"""
    try:
        # 优化搜索查询
        financial_keywords = ['金融', '财经', '经济', '股市', '基金', '债券', '外汇', 
                            '期货', '理财', '投资', '证券', '银行', '保险', '金价']
        
        # 检查是否需要添加财经关键词
        if not any(keyword in query for keyword in financial_keywords):
            # 添加财经相关关键词以提高相关性
            optimized_query = query + ' 财经'
        else:
            optimized_query = query
        
        # 使用 DuckDuckGoSearchRun 进行主搜索
        search_tool = DuckDuckGoSearchRun()
        initial_results = search_tool.run(optimized_query)
        
        # 使用 DDGS 进行补充搜索
        with DDGS() as ddgs:
            detailed_results = list(ddgs.text(
                optimized_query,
                max_results=max_results,
                region='cn-zh',
                safesearch='moderate',
                timelimit='m'  # 限制最近一个月的结果，保证信息时效性
            ))
        
        # 结果处理和排序
        processed_results = []
        seen_content = set()
        
        if detailed_results:
            for result in detailed_results:
                title = clean_text(result.get('title', ''))
                snippet = clean_text(result.get('body', ''))
                link = result.get('link', '')
                
                # 内容去重检查
                content_hash = f"{title}_{snippet}"
                if content_hash in seen_content:
                    continue
                seen_content.add(content_hash)
                
                # 计算域名质量分数
                domain_score = is_financial_domain(link)
                
                # 计算内容相关性分数
                relevance_score = sum(1 for word in query.lower().split() 
                                    if word in title.lower() or word in snippet.lower())
                
                # 检查是否包含财经关键词
                financial_relevance = sum(1 for keyword in financial_keywords 
                                        if keyword in title or keyword in snippet)
                
                # 综合评分
                total_score = domain_score * 3 + relevance_score * 2 + financial_relevance * 2
                
                if domain_score > 0 or financial_relevance > 0:  # 只保留金融相关网站的内容
                    processed_results.append({
                        'title': title,
                        'snippet': snippet,
                        'link': link,
                        'score': total_score
                    })
        
        # 按综合评分排序
        processed_results.sort(key=lambda x: x['score'], reverse=True)
        
        # 构建最终响应
        final_response = "财经相关搜索结果：\n\n"
        
        # 添加初步搜索结果
        if initial_results and any(keyword in initial_results.lower() for keyword in financial_keywords):
            final_response += f"{initial_results}\n\n"
        
        # 添加高质量补充结果
        if processed_results:
            final_response += "补充信息：\n"
            for idx, result in enumerate(processed_results[:5], 1):
                if result['score'] > 4:  # 提高显示阈值，确保高质量结果
                    final_response += f"{idx}. **{result['title']}**\n"
                    final_response += f"   {result['snippet']}\n"
                    final_response += f"   来源：[{urlparse(result['link']).netloc}]({result['link']})\n\n"
        
        return final_response.strip()
    
    except Exception as e:
        st.error(f"财经信息搜索失败: {str(e)}")
        return None

def get_search_response(query):
    """生成优化的财经搜索响应，并由大模型总结"""
    try:
        # 获取搜索结果
        search_results = perform_web_search(query)
        if not search_results:
            return "抱歉，没有找到相关的财经信息。"
        
        # 构建提示词，让大模型进行总结
        summary_prompt = f"""
请针对以下用户问题和搜索结果，进行专业的总结分析：

用户问题：{query}

搜索结果：
{search_results}

请你作为金融专家：
1. 提取要点，直接回答用户的核心问题
2. 确保信息的准确性和时效性
3. 如有必要，给出专业的建议或风险提示
4. 保持简洁清晰，突出重点

请以专业、客观的口吻回答。
"""
        # 调用大模型进行总结
        summary = call_model_api(summary_prompt, st.session_state.selected_model)
        
        # 构建最终响应
        response = "📊 **核心回答：**\n\n"
        response += f"{summary}\n\n"
        response += "---\n"
        response += "🔍 **详细搜索结果：**\n\n"
        response += f"{search_results}\n\n"
        response += "---\n"
        response += "*以上信息来自权威财经金融网站，并经AI分析整理。请注意信息时效性，建议进一步核实具体数据。*"
        
        return response

    except Exception as e:
        st.error(f"生成回答失败：{str(e)}")
        return None

def process_urls(urls_input):
    """处理输入的网址，提取内容并添加到 RAG 知识库"""
    urls = [url.strip() for url in urls_input.split('\n') if url.strip()]
    
    for url in urls:
        with st.spinner(f"正在处理网址：{url}"):
            try:
                # 发送 HTTP 请求获取网页内容
                response = requests.get(url, timeout=10)
                response.raise_for_status()  # 检查请求是否成功
                
                # 使用 BeautifulSoup 解析网页内容
                soup = BeautifulSoup(response.content, 'html.parser')
                
                # 移除脚本和样式元素
                for script in soup(["script", "style"]):
                    script.decompose()
                
                # 提取文本内容
                text = soup.get_text()
                
                # 清理文本（移除多余的空白字符）
                lines = (line.strip() for line in text.splitlines())
                chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
                text = ' '.join(chunk for chunk in chunks if chunk)
                
                if text:
                    # 将网页内容添加到 RAG 知识库
                    if rag_index_document(text, url):
                        st.session_state.rag_data.append(url)
                        st.success(f"✅ 网址 {url} 已成功加入知识库")
                else:
                    st.warning(f"⚠️ 网址 {url} 未提取到有效内容")
                    
            except requests.RequestException as e:
                st.error(f"❌ 访问网址 {url} 失败：{str(e)}")
            except Exception as e:
                st.error(f"❌ 处理网址 {url} 时出错：{str(e)}")

def get_embeddings():
    """获取 embeddings 实例"""
    try:
        # 使用中文基础模型
        embeddings = HuggingFaceEmbeddings(
            model_name="shibing624/text2vec-base-chinese",
            cache_folder="models"
        )
        return embeddings
    except Exception as e:
        st.error(f"初始化 embeddings 失败：{str(e)}")
        import traceback
        st.error(f"详细错误：{traceback.format_exc()}")
        return None

# ====================
# 侧边栏配置
# ====================
with st.sidebar:
    st.header("⚙️ 系统设置")

    # API 密钥管理
    st.subheader("API密钥管理")
    api_key_input = st.text_input(
        "输入 API 密钥",
        help="输入一个API密钥，用于访问所选模型",
        type="password"
    )
    api_keys_to_set = {
        "豆包": api_key_input,
        "Kimi(视觉理解)": api_key_input,
        "DeepSeek": api_key_input,
        "通义千问": api_key_input,
        "混元生文": api_key_input,
        "文心一言": api_key_input,
        "智谱清言": api_key_input,
        "MiniMax": api_key_input,
        "OpenAI": api_key_input,
        "xAI": api_key_input
    }
    if api_key_input:
        for key, value in api_keys_to_set.items():
            st.session_state.api_keys[key] = value
        st.success("API 密钥已保存！")

    # 模型选择
    model_options = {
        "豆包": ["ep-20250128163906-p4tb5"],
        "DeepSeek-V3": ["deepseek-chat"],
        "通义千问": ["qwen-plus"],
        "混元生文": ["hunyuan-turbo"],
        "文心一言": ["ERNIE-Bot"],
        "智谱清言": ["glm-4"],
        "MiniMax": ["abab5.5-chat"],
        "DALL-E(文生图)": ["dall-e-3"],
        "DeepSeek-R1(深度推理)": ["deepseek-reasoner"],
        "o1(深度推理)": ["o1-mini"],
        "Kimi(视觉理解)": ["moonshot-v1-8k-vision-preview"],
        "GPTs(聊天、语音识别)": ["gpt-4"],
        "grok2": ["grok-2-latest"]
    }

    st.session_state.selected_model = st.selectbox(
        "选择大模型",
        list(model_options.keys()),
        index=0
    )

    # 功能选择
    function_options = [
        "智能问答",
        "文本翻译",
        "文本总结",
        "文生图",
        "深度推理",
        "视觉理解",
        "语音识别"
    ]
    st.session_state.selected_function = st.selectbox(
        "选择功能",
        function_options,
        index=0
    )

    # 通用参数
    col1, col2 = st.columns(2)
    with col1:
        st.session_state.temperature = st.slider("创意度", 0.0, 1.0, 0.5, 0.1)
    with col2:
        st.session_state.max_tokens = st.slider("响应长度", 100, 4096, 2048, 100)

    # 联网搜索功能按钮
    if st.button(
        f"🌏 联网搜索[{('on' if st.session_state.search_enabled else 'off')}]",
        use_container_width=True
    ):
        st.session_state.search_enabled = not st.session_state.search_enabled
        st.rerun()

    # RAG 功能按钮
    if st.button(
        f"📚 RAG 功能[{('on' if st.session_state.rag_enabled else 'off')}]",
        use_container_width=True
    ):
        st.session_state.rag_enabled = not st.session_state.rag_enabled
        st.rerun()

    # API 测试功能
    st.subheader("API 测试")
    if st.button("🔍 测试 API 连接"):
        if not st.session_state.api_keys:
            st.error("请先输入 API 密钥！")
        else:
            with st.spinner("正在测试 API 连接..."):
                try:
                    test_prompt = "您好，请回复'连接成功'。"
                    response = call_model_api(test_prompt, st.session_state.selected_model)
                    if response:
                        st.success(f"API 连接成功！模型回复：{response}")
                    else:
                        st.error("API 连接失败，请检查密钥和网络设置。")
                except Exception as e:
                    st.error(f"API 测试失败：{str(e)}")

    if st.button("🧹 清空对话历史"):
        st.session_state.messages = []
        st.rerun()

    # 在主界面的侧边栏添加 ChromaDB 配置
    if st.session_state.rag_enabled:
        configure_chromadb()

    # 在API测试功能之前添加助手市场配置
    with st.expander("👥 助手市场", expanded=True):
        st.markdown("### 专业助手选择")
        
        domain = st.selectbox(
            "选择领域",
            options=list(st.session_state.assistant_market.keys()),
            key="domain_selector"
        )

        assistant = st.selectbox(
            "选择专业助手",
            options=["无"] + list(st.session_state.assistant_market[domain].keys()),
            key="assistant_selector"
        )

        if assistant != "无":
            st.session_state.selected_assistant = assistant
            st.markdown(f"**当前助手角色：** {assistant}")
            # 使用可折叠的容器替代expander
            with st.container():
                st.markdown("**助手详细说明：**")
                st.markdown(st.session_state.assistant_market[domain][assistant])
        else:
            st.session_state.selected_assistant = None

        st.markdown("""
        **使用说明：**
        1. 选择专业领域和具体角色
        2. 助手将以专业角色身份回答
        3. 可随时切换或取消角色
        4. 专业建议仅供参考
        """)

    # 在侧边栏最下方添加更新说明
    # 建议放在所有侧边栏内容之后
    st.sidebar.markdown("<br>" * 2, unsafe_allow_html=True)  # 添加一些空行作为间隔
    st.sidebar.markdown("""
    <div style='font-size: 0.8em; color: #666666; border-top: 1px solid #e6e6e6; padding-top: 10px;'>
    <b>更新说明：</b>
    <br>1. 增加腾讯混元大模型支持
    <br>2. 增加"助手市场"功能
    <br>3. 强化所有模型上下文记忆功能
    </div>
    """, unsafe_allow_html=True)

# ====================
# 主界面布局
# ====================
st.title("🤖 多模型智能助手2.10(学术增强版)")

# 文件和网址上传区域
st.markdown("### 📁 文件上传")

# RAG 模式：多文件上传和网址输入
if st.session_state.rag_enabled:
    # 文件上传
    uploaded_files = st.file_uploader(
        "支持多个文件上传（建议不超过5个）",
        accept_multiple_files=True,
        type=["txt", "pdf", "docx", "doc", "csv", "xlsx", "xls"],
        key="multi_file_uploader"
    )
    
    # 网址输入
    st.markdown("### 🔗 网址上传")
    urls_input = st.text_area(
        "每行输入一个网址（建议不超过5个）",
        height=100,
        key="urls_input",
        placeholder="https://example1.com\nhttps://example2.com"
    )
    
    # 提交按钮
    if st.button("📤 提交文件和网址"):
        if not uploaded_files and not urls_input.strip():
            st.warning("请至少上传一个文件或输入一个网址。")
        else:
            success_count = 0
            # 处理文件
            if uploaded_files:
                if len(uploaded_files) > 5:
                    st.warning("⚠️ 文件数量超过5个，建议减少文件数量以获得更好的处理效果。")
                
                for file in uploaded_files:
                    with st.spinner(f"正在处理文件：{file.name}"):
                        try:
                            content = extract_text_from_file(file)
                            if content:
                                if rag_index_document(content, file.name):
                                    success_count += 1
                                    st.session_state.rag_data.append(file.name)
                                    st.success(f"✅ 文件 {file.name} 已成功加入知识库")
                            else:
                                st.error(f"❌ 无法提取文件内容：{file.name}")
                        except Exception as e:
                            st.error(f"❌ 处理文件失败：{str(e)}")
            
            # 处理网址
            if urls_input.strip():
                process_urls(urls_input)

            if success_count > 0:
                st.success(f"✅ 共成功处理 {success_count} 个文件/网址")
                # 强制刷新向量库实例
                st.session_state.vector_store = None
                # 重新加载向量库
                get_vector_store()
            else:
                st.error("❌ 未能成功处理任何文件或网址")

# 非 RAG 模式：单文件上传并立即处理
else:
    uploaded_file = st.file_uploader(
        "上传单个文件进行分析",
        accept_multiple_files=False,
        type=["txt", "pdf", "docx", "doc", "jpg", "jpeg", "png", "mp3", "wav", "m4a"],
        key="single_file_uploader"
    )
    
    if uploaded_file:
        file_type = uploaded_file.name.split('.')[-1].lower()
        
        try:
            # 1. 语音识别（GPTs）
            if file_type in ["mp3", "wav", "m4a"]:
                with st.spinner("🎵 正在进行语音识别..."):
                    if "OpenAI" not in st.session_state.api_keys:
                        st.error("请先配置 OpenAI API 密钥")
                    else:
                        client = OpenAI(api_key=st.session_state.api_keys["OpenAI"])
                        with tempfile.NamedTemporaryFile(delete=False, suffix=f".{file_type}") as tmp_file:
                            tmp_file.write(uploaded_file.getvalue())
                            tmp_file.flush()
                            
                            with open(tmp_file.name, "rb") as audio_file:
                                transcription = client.audio.transcriptions.create(
                                    model="whisper-1",
                                    file=audio_file,
                                    language="zh"
                                )
                        
                        st.success("✅ 语音识别完成")
                        with st.chat_message("assistant"):
                            st.markdown(f"**语音识别结果：**\n\n{transcription.text}")
                        st.session_state.messages.append({
                            "role": "assistant",
                            "content": f"语音文件 {uploaded_file.name} 的识别结果：\n\n{transcription.text}",
                            "type": "text"
                        })
            
            # 2. 图片分析（moonshot-v1-8k-vision-preview）
            elif file_type in ["jpg", "jpeg", "png"]:
                with st.spinner("🖼️ 正在分析图片..."):
                    if st.session_state.selected_model == "Kimi(视觉理解)":  # 只使用 Kimi 进行视觉理解
                        if "Kimi(视觉理解)" not in st.session_state.api_keys:
                            st.error("请先配置 Kimi(视觉理解) API 密钥")
                        else:
                            image_content = uploaded_file.getvalue()
                            encoded_image = base64.b64encode(image_content).decode('utf-8')
                            
                            headers = {
                                "Content-Type": "application/json",
                                "Authorization": f"Bearer {st.session_state.api_keys['Kimi(视觉理解)']}"
                            }
                            
                            payload = {
                                "model": "moonshot-v1-8k-vision-preview",
                                "messages": [
                                    {
                                        "role": "user",
                                        "content": [
                                            {
                                                "type": "text",
                                                "text": "请详细分析这张图片的内容，包括主要对象、场景、细节等方面。"
                                            },
                                            {
                                                "type": "image_url",
                                                "image_url": {
                                                    "url": f"data:image/jpeg;base64,{encoded_image}"
                                                }
                                            }
                                        ]
                                    }
                                ]
                            }
                            
                            response = requests.post(
                                "https://api.moonshot.cn/v1/chat/completions",
                                json=payload,
                                headers=headers
                            )
                            
                            if response.status_code == 200:
                                result = response.json()["choices"][0]["message"]["content"]
                                st.success("✅ 图片分析完成")
                                with st.chat_message("assistant"):
                                    st.markdown(f"**图片分析结果：**\n\n{result}")
                                st.session_state.messages.append({
                                    "role": "assistant",
                                    "content": f"图片 {uploaded_file.name} 的分析结果：\n\n{result}",
                                    "type": "text"
                                })
                            else:
                                st.error(f"❌ 图片分析失败：{response.text}")
            
            # 3. 文档总结
            elif file_type in ["txt", "pdf", "docx", "doc"]:
                with st.spinner("📄 正在总结文档..."):
                    content = extract_text_from_file(uploaded_file)
                    if content:
                        summary_prompt = f"""请对以下文本进行专业的总结分析：

文本内容：
{content}

请从以下几个方面进行总结：
1. 核心要点（最重要的2-3个关键信息）
2. 主要内容概述
3. 重要结论或发现
4. 相关建议（如果适用）

请用清晰、专业的语言组织回答。"""

                        summary = call_model_api(summary_prompt, st.session_state.selected_model)
                        if summary:
                            st.success("✅ 文档总结完成")
                            with st.chat_message("assistant"):
                                st.markdown(f"**文档总结结果：**\n\n{summary}")
                            st.session_state.messages.append({
                                "role": "assistant",
                                "content": f"文档 {uploaded_file.name} 的总结：\n\n{summary}",
                                "type": "text"
                            })
            
            else:
                st.warning(f"⚠️ 不支持的文件类型：{file_type}")
        
        except Exception as e:
            st.error(f"❌ 处理文件失败：{str(e)}")
            import traceback
            st.error(f"详细错误：{traceback.format_exc()}")

# ====================
# 用户问题输入区域
with st.container():
    # 初始提示（仅在对话记录为空时显示）
    if not st.session_state.messages:
        with st.chat_message("assistant"):
            st.write("您好！我是多模型智能助手，请选择模型和功能开始交互。")
            
    # 在主界面聊天部分，修改用户输入区域的代码
    # 在用户输入前添加当前助手提示
    if st.session_state.selected_assistant:
        st.markdown(
            f"<p style='color: #666666; font-size: 0.8em; margin-bottom: 5px;'> 👨 当前助手：{st.session_state.selected_assistant}</p>", 
            unsafe_allow_html=True
        )

    # 用户输入
    user_input = st.chat_input(
        "请输入您的问题",
        key="user_input"
    )
    
    if user_input:
        # 记录用户输入到历史记录
        manage_chat_history(st.session_state.selected_model, "user", user_input)
        
        with st.chat_message("user"):
            st.write(user_input)
        
        st.session_state.messages.append({"role": "user", "content": user_input, "type": "text"})
        
        with st.spinner("🧠 正在思考..."):
            combined_response = ""
            
            # 联网搜索部分
            if st.session_state.search_enabled:
                try:
                    search_response = get_search_response(user_input)
                    if search_response:
                        combined_response += search_response + "\n\n"
                except Exception as e:
                    st.error(f"搜索过程出错：{str(e)}")
            
            # RAG 检索部分
            if st.session_state.rag_enabled:
                try:
                    rag_response = rag_generate_response(user_input)
                    if rag_response:
                        combined_response += "📚 **知识库检索结果：**\n\n" + rag_response + "\n\n"
                except Exception as e:
                    st.error(f"RAG 检索出错：{str(e)}")
            
            # 如果两个功能都未开启，使用普通对话模式
            if not (st.session_state.search_enabled or st.session_state.rag_enabled):
                response = call_model_api(user_input, st.session_state.selected_model)
                if response:
                    combined_response = response
            
            # 显示组合后的回答
            if combined_response:
                with st.chat_message("assistant"):
                    st.markdown(combined_response)
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": combined_response,
                    "type": "text"
                })
                # 记录助手回答到历史记录
                manage_chat_history(st.session_state.selected_model, "assistant", combined_response)
            else:
                st.error("未能获取到任何结果，请重试。")

# ====================
# 显示历史对话记录
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        if msg.get("type") == "image":
            st.image(msg["content"])
        else:
            st.write(msg["content"])