import streamlit as st
import os
import json
import uuid
import re
from datetime import datetime
import asyncio

# --- Agno & Qdrant Imports ---
from agno.agent import Agent
from agno.models.openai import OpenAIChat
from agno.knowledge.embedder.openai import OpenAIEmbedder
from agno.tools.tavily import TavilyTools 
from qdrant_client import QdrantClient, models

# ==========================================
# 0. 全局配置 (请在此处填入你的 Key)
# ==========================================
TAVILY_API_KEY = "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" 

# ==========================================
# 1. 页面基础配置与疗愈系 UI
# ==========================================
st.set_page_config(
    page_title="心灵港湾 | Mind Harbor",
    page_icon="🌿",
    layout="wide",
    initial_sidebar_state="expanded"
)

# 注入自定义 CSS
st.markdown("""
<style>
    .stApp { background-color: #F9F7F2; }
    .stChatMessage { background-color: transparent; border: none; padding: 15px 0; }
    div[data-testid="stChatMessage"] {
        padding: 1.2rem; border-radius: 18px; margin-bottom: 12px;
        box-shadow: 0 2px 5px rgba(0,0,0,0.03);
        font-family: 'Helvetica Neue', sans-serif; line-height: 1.6;
    }
    div[data-testid="stChatMessage"][data-test-role="assistant"] {
        background-color: #FFFFFF; border-left: 4px solid #A3B18A; color: #4A4A4A;
    }
    div[data-testid="stChatMessage"][data-test-role="user"] {
        background-color: #DAD7CD; color: #3A3A3A; flex-direction: row-reverse; text-align: right;
    }
    h1 { color: #588157; font-weight: 300; text-align: center; margin-bottom: 30px; }
    section[data-testid="stSidebar"] { background-color: #F3F1EB; }
    .clinical-note {
        background-color: #EDF6F9; color: #457B9D; padding: 12px;
        border-radius: 8px; font-size: 0.85em; margin-top: 8px; border: 1px dashed #A8DADC;
    }
    .search-result {
        font-size: 0.8em; color: #666; border-left: 2px solid #E9C46A; padding-left: 10px; margin-bottom: 5px;
    }
</style>
""", unsafe_allow_html=True)

# ==========================================
# 2. Vector DB 逻辑 (本地模式)
# ==========================================
COLLECTION_NAME = "mind_harbor_memories"

def get_qdrant_client():
    return QdrantClient(path="./qdrant_local_storage")

def get_embedder():
    if not st.session_state.get('openai_api_key'):
        return None
    return OpenAIEmbedder(
        id="text-embedding-3-small",
        api_key=st.session_state['openai_api_key'],
        base_url=st.session_state.get('base_url', "https://api.zhizengzeng.com/v1")
    )

def ensure_collection_exists():
    client = get_qdrant_client()
    if not client: return
    if not client.collection_exists(COLLECTION_NAME):
        client.create_collection(
            collection_name=COLLECTION_NAME,
            vectors_config=models.VectorParams(size=1536, distance=models.Distance.COSINE)
        )

def search_memory(username, query_text, limit=5):
    client = get_qdrant_client()
    embedder = get_embedder()
    if not client or not embedder: return "System: 记忆模块未连接(请检查OpenAI Key)。"

    try:
        ensure_collection_exists()
        query_vector = embedder.get_embedding(query_text)
        search_result = client.search(
            collection_name=COLLECTION_NAME,
            query_vector=query_vector,
            query_filter=models.Filter(
                must=[models.FieldCondition(key="username", match=models.MatchValue(value=username))]
            ),
            limit=limit
        )
        if not search_result: return "暂无相关历史记忆。"
        return "\n".join([f"- [{hit.payload['timestamp']}] {hit.payload['text']}" for hit in search_result])
    except Exception as e:
        return f"记忆检索出错: {e}"

def save_memory_fragment(username, memory_text):
    client = get_qdrant_client()
    embedder = get_embedder()
    if not client or not embedder: return False

    try:
        ensure_collection_exists()
        vector = embedder.get_embedding(memory_text)
        payload = {
            "username": username, "text": memory_text,
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"), "type": "summary"
        }
        client.upsert(
            collection_name=COLLECTION_NAME,
            points=[models.PointStruct(id=str(uuid.uuid4()), vector=vector, payload=payload)]
        )
        return True
    except Exception as e:
        st.error(f"记忆保存失败: {e}")
        return False

# ==========================================
# 3. 辅助函数：鲁棒的 JSON 解析
# ==========================================
def robust_json_parse(text):
    """防止 Agent 输出 Markdown 或不规范格式导致崩溃"""
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    try:
        match = re.search(r'\{.*\}', text, re.DOTALL)
        if match:
            return json.loads(match.group(0))
    except Exception:
        pass
    return {"insight": "数据解析暂不可用", "stress_score": 5, "distortion": "Unknown"}

# ==========================================
# 4. Session State 初始化
# ==========================================
DEFAULT_BASE_URL = "https://api.zhizengzeng.com/v1"

if 'messages' not in st.session_state: st.session_state['messages'] = []
if 'analysis_logs' not in st.session_state: st.session_state['analysis_logs'] = []
if 'emotion_scores' not in st.session_state: st.session_state['emotion_scores'] = [5]
if 'current_user' not in st.session_state: st.session_state['current_user'] = None
if 'temp_pill_input' not in st.session_state: st.session_state['temp_pill_input'] = None
if 'search_logs' not in st.session_state: st.session_state['search_logs'] = []
if 'dynamic_suggestions' not in st.session_state: 
    st.session_state['dynamic_suggestions'] = ["最近感觉很累", "我想聊聊人际关系", "怎么缓解焦虑？", "我不知道该怎么办"]

if 'openai_api_key' not in st.session_state: st.session_state['openai_api_key'] = ''
if 'base_url' not in st.session_state: st.session_state['base_url'] = DEFAULT_BASE_URL

# ==========================================
# 5. 侧边栏
# ==========================================
with st.sidebar:
    st.title("🌿 咨询室接待处")
    
    if not st.session_state['current_user']:
        st.info("请登录以读取您的专属档案")
        username_input = st.text_input("请输入您的名字", placeholder="例如: Ashley")
        if st.button("进入咨询室"):
            if username_input:
                st.session_state['current_user'] = username_input
                welcome_text = f"你好 {username_input}，我是你的AI心理伙伴。这里很安全，你可以畅所欲言。"
                st.session_state['messages'] = [{"role": "assistant", "content": welcome_text}]
                st.rerun()
    else:
        st.success(f"当前用户: **{st.session_state['current_user']}**")
        if st.button("🚪 退出 / 切换账号"):
            st.session_state['current_user'] = None
            st.session_state['messages'] = []
            st.session_state['emotion_scores'] = [5]
            st.session_state['analysis_logs'] = []
            st.session_state['search_logs'] = []
            st.session_state['temp_pill_input'] = None
            st.session_state['dynamic_suggestions'] = ["最近感觉很累", "我想聊聊人际关系", "怎么缓解焦虑？"]
            st.rerun()
    
    st.divider()
    st.subheader("📊 心理压力监测")
    if len(st.session_state['emotion_scores']) > 1:
        st.line_chart(st.session_state['emotion_scores'], height=150)
        curr = st.session_state['emotion_scores'][-1]
        prev = st.session_state['emotion_scores'][-2]
        st.metric("当前压力指数 (0-10)", f"{curr}", f"{curr-prev}", delta_color="inverse")
    else:
        st.caption("暂无足够数据，请开始对话。")

    st.divider()
    with st.expander("⚙️ 系统设置"):
        st.session_state['openai_api_key'] = st.text_input("OpenAI Key", type="password", value=st.session_state['openai_api_key'])
        st.session_state['base_url'] = st.text_input("Base URL", value=st.session_state['base_url'])
        
        st.info("🧠 记忆库状态: **本地内置 (Local)**")
        st.info("🔍 搜索插件: **Tavily (已内置)**")
        
        if st.button("🧹 清空当前对话"):
            if st.session_state['current_user']:
                st.session_state['messages'] = [{"role": "assistant", "content": "好的，我们重新开始。此刻你感觉如何？"}]
                st.session_state['emotion_scores'] = [5]
                st.session_state['analysis_logs'] = []
                st.session_state['dynamic_suggestions'] = ["说说你现在的想法", "可以做个深呼吸", "最近睡眠怎么样？"]
                st.rerun()

if not st.session_state['openai_api_key']: st.warning("🔒 请输入 OpenAI API Key"); st.stop()
if not st.session_state['current_user']: st.stop()

os.environ["OPENAI_API_KEY"] = st.session_state['openai_api_key']
os.environ["OPENAI_BASE_URL"] = st.session_state['base_url']

# ==========================================
# 6. Agent 定义
# ==========================================
def get_model(model_id="gpt-4o"):
    return OpenAIChat(
        id=model_id,
        api_key=st.session_state['openai_api_key'],
        base_url=st.session_state['base_url']
    )

triage_agent = Agent(
    name="Guardian",
    model=get_model("gpt-4o-mini"),
    instructions=[
        "你是一个心理危机干预的安全守门员。",
        "任务：分析用户输入是否包含：自杀意念、自残计划、严重暴力倾向。",
        "输出格式：'CRISIS_ALERT: <原因>' 或 'SAFE: <情绪关键词>'。",
        "不要输出其他任何内容。"
    ],
    markdown=False
)

analyst_agent = Agent(
    name="Logic",
    model=get_model("gpt-4o-mini"),
    instructions=[
        "你是一位专业的临床心理分析师。",
        "任务：分析用户输入并返回纯 JSON 数据。",
        "JSON 字段必须包含：",
        "1. 'insight': 简短临床观察。",
        "2. 'stress_score': 0-10 的整数（10为最高压力）。",
        "3. 'distortion': 认知扭曲类型（无则填'None'）。",
        "**重要**：不要使用 Markdown 代码块，直接返回 JSON 字符串。",
        "Example: {\"insight\": \"用户感到焦虑\", \"stress_score\": 7, \"distortion\": \"过度概括\"}"
    ],
    markdown=False
)

router_agent = Agent(
    name="Router",
    model=get_model("gpt-4o-mini"),
    instructions=[
        "任务：判断用户意图是否需要外部搜索。",
        "如果询问具体药物、地址、科学定义、统计数据 -> 输出 'SEARCH'。",
        "如果是情绪发泄、寻求安慰、闲聊 -> 输出 'CHAT'。",
        "只输出一个单词。"
    ],
    markdown=False
)

# [修改点] 使用内置的 Tavily Key
navigator_tools = []
if TAVILY_API_KEY and "tvly-" in TAVILY_API_KEY:
    navigator_tools = [TavilyTools(api_key=TAVILY_API_KEY)]
else:
    # 如果没填 Key，给一个警告 (仅在控制台)
    print("Warning: TAVILY_API_KEY not set in code.")

navigator_agent = Agent(
    name="Navigator",
    model=get_model("gpt-4o-mini"),
    tools=navigator_tools,
    instructions=[
        "你是一个研究助手。使用 Tavily 搜索用户需要的资源。",
        "用中文简洁总结搜索结果，优先提供事实性信息。"
    ],
    # show_tool_calls=False,
    markdown=True
)

therapist_agent = Agent(
    name="Therapist",
    model=get_model("gpt-4o"),
    instructions=[
        "你现在是‘心灵港湾’的专业心理咨询师‘小安’。",
        "风格：人本主义、温暖支持、同理心。",
        "如果提供了[RESOURCE SEARCH RESULTS]，自然地融入对话。",
        "参考[RELEVANT MEMORIES]让对话有连续性。",
        "用中文回答。"
    ],
    markdown=True
)

archivist_agent = Agent(
    name="Archivist",
    model=get_model("gpt-4o-mini"),
    instructions=[
        "任务：将对话总结为简洁的长期记忆片段。",
        "输出纯文本摘要。"
    ],
    markdown=True
)

suggester_agent = Agent(
    name="Suggester",
    model=get_model("gpt-4o-mini"),
    instructions=[
        "任务：根据上下文生成 3 个简短的用户后续回复建议。",
        "格式：用英文逗号分隔的纯文本。例如：'我想多聊聊, 好的谢谢, 还有别的方法吗'。"
    ],
    markdown=False
)

# ==========================================
# 7. 对话流编排 (修复 Error 逻辑)
# ==========================================
async def run_parallel_analysis(user_input):
    task_triage = asyncio.to_thread(triage_agent.run, f"用户输入: {user_input}")
    task_analyst = asyncio.to_thread(analyst_agent.run, f"用户输入: {user_input}")
    task_router = asyncio.to_thread(router_agent.run, f"用户输入: {user_input}")
    
    return await asyncio.gather(task_triage, task_analyst, task_router)

def process_conversation_turn(user_input):
    # A. Memory Retrieval
    with st.spinner("🧠 正在回溯记忆..."):
        relevant_memories = search_memory(st.session_state['current_user'], user_input)
    
    short_term_history = "\n".join([f"{m['role']}: {m['content']}" for m in st.session_state['messages'][-10:]])
    
    with st.status("🍃 正在用心倾听...", expanded=False) as status:
        
        # B. Parallel Analysis
        status.write("⚡ 正在感知情绪与意图...")
        try:
            triage_resp, analyst_resp, router_resp = asyncio.run(run_parallel_analysis(user_input))
        except Exception as e:
            return f"系统分析模块出错: {str(e)}"

        # 1. Check Safety
        if "CRISIS_ALERT" in triage_resp.content:
            status.update(label="⚠️ 安全拦截", state="error")
            return f"🚨 **紧急安全提示**\n\n检测到潜在高风险。请立即寻求专业帮助。\nReason: {triage_resp.content}"
        
        current_emotion = triage_resp.content.replace("SAFE:", "").strip()

        # 2. Parse Analyst Data (使用 Robust JSON Parse)
        data = robust_json_parse(analyst_resp.content)
        score = int(data.get("stress_score", 5))
        insight = data.get("insight", "")
        distortion = data.get("distortion", "None")
        
        st.session_state['emotion_scores'].append(score)
        st.session_state['analysis_logs'].append(f"压力: {score} | 扭曲: {distortion} | {insight}")

        # 3. Handle Search Intent (Tavily Search)
        search_results = "本次无需外部搜索。"
        intent = router_resp.content.strip()
        
        if "SEARCH" in intent:
            status.write("🌐 正在尝试连接外部网络...")
            if not navigator_tools:
                 search_results = "【系统提示】代码中未正确配置 Tavily Key，无法搜索。"
                 st.session_state['search_logs'].append(f"⚠️ 搜索失败: Key未配置")
            else:
                try:
                    nav_response = navigator_agent.run(f"请搜索: {user_input}")
                    if nav_response and nav_response.content:
                        search_results = nav_response.content
                        st.session_state['search_logs'].append(f"🔍 Tavily搜索成功: {user_input[:10]}...")
                    else:
                        search_results = "搜索未返回结果。"
                        st.session_state['search_logs'].append(f"🔍 搜索无结果")
                except Exception as e:
                    search_results = f"【系统提示】搜索服务连接失败: {str(e)}"
                    st.session_state['search_logs'].append(f"⚠️ 搜索失败: {str(e)}")
        else:
            st.session_state['search_logs'].append("💭 纯对话模式")

        # C. Generate Response
        status.write("🌿 正在生成温暖回复...")
        
        full_prompt = f"""
        [RELEVANT MEMORIES]
        {relevant_memories}
        
        [RESOURCE SEARCH RESULTS]
        {search_results}
        
        [SHORT-TERM HISTORY]
        {short_term_history}
        
        [CURRENT SITUATION]
        用户输入: {user_input}
        当前情绪: {current_emotion}
        分析师观察: {insight}
        
        [INSTRUCTION]
        请自然地回应用户。如果[RESOURCE SEARCH RESULTS]提示缺少 Key 或搜索失败，请根据通用心理学知识进行安抚，不要直接暴露技术错误信息。
        """
        
        therapist_resp = therapist_agent.run(full_prompt)
        response_content = therapist_resp.content

        # D. Generate Dynamic Suggestions
        try:
            sugg_resp = suggester_agent.run(f"用户: {user_input}\nAI: {response_content}\n生成3个简短回复建议，逗号分隔。")
            raw_suggs = sugg_resp.content.replace("，", ",").split(",")
            clean_suggs = [s.strip() for s in raw_suggs if s.strip()][:3]
            if clean_suggs:
                st.session_state['dynamic_suggestions'] = clean_suggs
        except:
            pass

        status.update(label="回复完成", state="complete")
        return response_content

# ==========================================
# 8. UI 主渲染区
# ==========================================
st.title("🌿 心灵港湾")
st.caption(f"Mind Harbor | 当前来访者: {st.session_state['current_user']}")

chat_container = st.container()
with chat_container:
    for msg in st.session_state['messages']:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

def pill_callback():
    if st.session_state.pill_selection:
        st.session_state['temp_pill_input'] = st.session_state.pill_selection
        st.session_state.pill_selection = None

suggestions = st.session_state.get('dynamic_suggestions', ["最近压力有点大", "即使睡着了也很累", "怎么才能控制情绪？"])
st.pills("💬 试试说（点击发送）：", suggestions, selection_mode="single", key="pill_selection", on_change=pill_callback)

user_final_input = None
if st.session_state['temp_pill_input']:
    user_final_input = st.session_state['temp_pill_input']
    st.session_state['temp_pill_input'] = None
chat_input_val = st.chat_input("在此输入您的感受...")
if chat_input_val: user_final_input = chat_input_val

if user_final_input:
    with st.chat_message("user"): st.markdown(user_final_input)
    st.session_state['messages'].append({"role": "user", "content": user_final_input})
    
    try:
        response_text = process_conversation_turn(user_final_input)
        with st.chat_message("assistant"): st.markdown(response_text)
        st.session_state['messages'].append({"role": "assistant", "content": response_text})
        st.rerun()
    except Exception as e:
        st.error(f"连接中断或出错: {e}")

# ==========================================
# 9. 底部功能区
# ==========================================
st.markdown("---")
col1, col2 = st.columns([3, 1])

with col1:
    if st.session_state['analysis_logs']:
        st.markdown("**👩‍⚕️ 咨询手记:**")
        st.markdown(f"<div class='clinical-note'>{st.session_state['analysis_logs'][-1]}</div>", unsafe_allow_html=True)
    if st.session_state['search_logs']:
        st.caption(f"系统状态: {st.session_state['search_logs'][-1]}")

with col2:
    if st.button("💾 结束并保存记忆"):
        if len(st.session_state['messages']) < 2:
            st.warning("对话太短，暂无内容。")
        else:
            with st.spinner("正在保存至本地..."):
                full_text = "\n".join([f"{m['role']}: {m['content']}" for m in st.session_state['messages']])
                summary_resp = archivist_agent.run(f"会话记录:\n{full_text}\n\n任务: 生成中文摘要。")
                success = save_memory_fragment(st.session_state['current_user'], summary_resp.content)
                if success:
                    st.success("✅ 保存成功！")
                    with st.expander("摘要"): st.markdown(summary_resp.content)
                else:

                    st.error("保存失败。")
