"""
Vector Search Engine
Redis Stack + LangChain + HNSW + RAG + Cache + Memory
"""

import os
import time
import streamlit as st
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_redis import RedisVectorStore
from redis import Redis
from redisvl.query.filter import Tag
from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate
from langchain_core.globals import set_llm_cache
from langchain_community.cache import RedisSemanticCache
from langchain_community.chat_message_histories import RedisChatMessageHistory

INDEX_NAME = "rag_knowledge_base"

DEEPSEEK_API_KEY = os.environ.get("DEEPSEEK_API_KEY", "sk-ef3a4e5c1a0c437e8927f18b5a445534")

embeddings = HuggingFaceEmbeddings(model_name="BAAI/bge-small-zh-v1.5")

if DEEPSEEK_API_KEY:
    llm = ChatOpenAI(
        api_key=DEEPSEEK_API_KEY,
        base_url="https://api.deepseek.com",
        model="deepseek-chat",
        streaming=True
    )
else:
    llm = None

@st.cache_resource
def load_ai_models():
    print(">>> 首次启动：正在将 BGE 模型加载到内存，请稍候...")
    embed = HuggingFaceEmbeddings(model_name="BAAI/bge-small-zh-v1.5")
    chat_llm = ChatOpenAI(
        api_key=DEEPSEEK_API_KEY,
        base_url="https://api.deepseek.com",
        model="deepseek-chat",
        streaming=True
    ) if DEEPSEEK_API_KEY else None
    return embed, chat_llm

# 获取缓存好的模型
embeddings, llm = load_ai_models()

# 配置全局语义缓存
try:
    set_llm_cache(RedisSemanticCache(
        redis_url="redis://localhost:6379",
        embedding=embeddings,
        score_threshold=0.3
    ))
except Exception as e:
    pass

# RAG 提示词模板
RAG_PROMPT = PromptTemplate.from_template("""你是一个专业的知识库问答助手。
请结合【历史聊天记录】和【参考资料】来回答用户的【最新问题】。
如果参考资料中没有提及相关内容，请扩充回答。

⚠️【格式严格要求】：
1. 请输出排版干净、自然易读的文本。
2. 绝对不要照抄或输出参考资料中的任何特殊分隔符（如 ///, ***, ---, === 等）。
3. 使用标准的 Markdown 标题或列表进行排版。

【历史聊天记录】：
{chat_history}

【参考资料】：
{context}

【最新问题】：
{query}

请回答：""")

st.set_page_config(
    page_title="Vector Search Engine",
    layout="wide",
)

st.markdown("""
<style>
    [data-testid="stAppViewContainer"], 
    [data-testid="stMainBlockContainer"], 
    [data-testid="stVerticalBlock"], 
    .stApp {
        opacity: 1 !important;
        filter: none !important;
        transition: none !important;
        animation: none !important;
    }
    header[data-testid="stHeader"], [data-testid="stDecoration"], #MainMenu, footer, .stDeployButton { display: none !important; }

    [data-testid="stMainBlockContainer"] { 
        opacity: 1 !important; 
        filter: blur(0px) !important; 
        padding-bottom: 50px !important; 
        padding-top: 1rem !important; 
    }
    [data-testid="stSidebar"] { background: #F9FAFB; border-right: 1px solid #E5E7EB; }
    /* 1. 消除 Streamlit 默认的底色 */
    div[data-testid="stChatMessage"] {
        background-color: transparent !important;
        padding: 0 !important;
        margin-bottom: 24px !important;
    }
    
    /* 2. 给 AI 的文本区套上极简气泡：淡灰蓝 (Slate 50)，左侧小尖角 */
    div[data-testid="stChatMessageContent"] {
        background-color: #F8FAFC !important; /* 淡色区别 1：AI专属淡灰蓝 */
        border: 1px solid #E2E8F0 !important;
        border-radius: 4px 20px 20px 20px !important; /* 左上角为锐角 */
        padding: 18px 24px !important;
        color: #1E293B !important;
        box-shadow: 0 1px 2px rgba(0,0,0,0.02) !important;
        max-width: 85% !important; /* 控制阅读宽度 */
    }

    /* ----- 文本细节排版 ----- */
    .stMarkdown p { line-height: 1.8 !important; letter-spacing: 0.02em !important; color: #334155; font-size: 0.95rem; margin-bottom: 1.2em !important; }
    .stMarkdown li { line-height: 1.8; color: #334155; font-size: 0.95rem; }
    
    /* ----- 深度思考折叠面板样式 (内置在AI气泡中) ----- */
    [data-testid="stExpander"] { border: none !important; background: transparent !important; box-shadow: none !important; margin-bottom: 0px !important; }
    [data-testid="stExpander"] summary { color: #64748B !important; font-size: 0.9rem; padding-left: 0 !important; border-bottom: 1px dashed #E2E8F0; padding-bottom: 6px; }
    [data-testid="stExpander"] summary p { color: #64748B !important; }

    /* ----- 还原：参考资料卡片样式 (使其在AI淡灰背景中凸显) ----- */
    .result-card { background: #FFFFFF; border: 1px solid #E2E8F0; border-radius: 12px; padding: 18px; margin-top: 15px; margin-bottom: 10px; box-shadow: 0 2px 4px -1px rgba(0,0,0,0.02); }
    .card-header { display: flex; align-items: center; justify-content: space-between; margin-bottom: 12px; padding-bottom: 10px; border-bottom: 1px solid #F8FAFC; }
    .card-title { font-weight: 600; font-size: 0.95rem; color: #0F172A; }
    .card-badges { display: flex; gap: 8px; align-items: center; }
    .badge { display: inline-block; padding: 2px 8px; border-radius: 999px; font-size: 0.72rem; font-weight: 500; }
    .badge-topic { background: #F1F5F9; color: #475569; }
    .badge-score { background: #ECFDF5; color: #059669; }
    .card-para { font-size: 0.88rem; color: #475569; line-height: 1.7; text-align: justify; }
    .card-footer { margin-top: 10px; padding-top: 10px; border-top: 1px dashed #F1F5F9; }
    .card-footer a { color: #94A3B8; font-size: 0.8rem; text-decoration: none; }

</style>
""", unsafe_allow_html=True)


# 核心功能与缓存函数
@st.cache_resource
def init_vector_store(_embed_model):
    client = Redis(host="localhost", port=6379)
    client.ping()
    return RedisVectorStore.from_existing_index(
        index_name=INDEX_NAME,
        embedding=embeddings,
        redis_client=client,
    )

vector_store = init_vector_store(embeddings)
client = Redis(host="localhost", port=6379)

@st.cache_data(ttl=30)
def get_dynamic_topics() -> list[str]:
    try:
        raw = client.execute_command("FT.TAGVALS", INDEX_NAME, "topic")
        if raw is None: return []
        return sorted([v.decode() if isinstance(v, bytes) else str(v) for v in raw])
    except Exception:
        return []

@st.cache_data(ttl=30)
def get_index_info() -> dict:
    try:
        raw = client.execute_command("FT.INFO", INDEX_NAME)
        info = {}
        it = iter(raw)
        for k in it:
            key = k.decode() if isinstance(k, bytes) else str(k)
            info[key] = next(it)
        return info
    except Exception:
        return {}

def render_paragraphs(content: str) -> str:
    paragraphs = [p.strip() for p in content.split("\n\n") if p.strip()]
    if not paragraphs: return f'<p class="card-para">{content.strip()}</p>'
    return "\n".join(f'<p class="card-para">{p}</p>' for p in paragraphs)

def parse_image_paths(raw: str) -> list[str]:
    if not raw: return []
    candidates = raw.split(",")
    return [p.strip() for p in candidates if p.strip() and os.path.isfile(p.strip())]
def render_user_bubble(text):
    safe_text = text.replace('\n', '<br>')
    html = f"""
    <div style="display: flex; justify-content: flex-end; margin-bottom: 24px; margin-top: 10px;">
        <div style="background-color: #EFF6FF; border: 1px solid #DBEAFE; color: #1E293B; padding: 14px 20px; border-radius: 20px 4px 20px 20px; max-width: 75%; font-size: 0.95rem; line-height: 1.8; box-shadow: 0 1px 2px rgba(0,0,0,0.02);">
            {safe_text}
        </div>
    </div>
    """
    st.markdown(html, unsafe_allow_html=True)


dynamic_topics = get_dynamic_topics()
topic_options = ["All"] + dynamic_topics

with st.sidebar:
    st.markdown("检索参数")
    top_k = st.slider("Top K", 1, 5, 3)
    if dynamic_topics:
        tag_option = st.selectbox("分类过滤", options=topic_options)
    else:
        tag_option = st.selectbox("分类过滤", options=["All"])

    st.markdown("---")
col1, col2 = st.columns([4, 1])

with col1:
    st.title("Redis Vector智能检索引擎")

with col2:
    st.write("")
    if st.button("🗑️ 清空对话", use_container_width=True):
        if "session_id" in st.session_state:
            RedisChatMessageHistory(st.session_state.session_id, url="redis://localhost:6379").clear()
            st.rerun()

st.markdown("---")

tab_search, tab_dashboard = st.tabs(["智能探索", "系统监控"])

# Tab 1: 智能探索空间
with tab_search:
    import uuid
    if "session_id" not in st.session_state:
        st.session_state.session_id = f"session_{uuid.uuid4()}"
    session_id = st.session_state.session_id

    chat_history = RedisChatMessageHistory(session_id, url="redis://localhost:6379")
    chat_container = st.container(height=400, border=False)
    with chat_container:
        for msg in chat_history.messages[-40:]:
            if msg.type == "human":
                render_user_bubble(msg.content)
            else:
                with st.chat_message("assistant"):
                    st.markdown(msg.content)
    query = st.chat_input("向大模型提问...")
    if query:
        with chat_container:
            render_user_bubble(query)
            tag_filter = None if (tag_option == "All" or not dynamic_topics) else Tag("topic") == tag_option
            with st.chat_message("assistant"):
                status_placeholder = st.empty()
                status_placeholder.markdown(" *正在理解语义并检索...*")

                start_time = time.time()
                raw_results = vector_store.similarity_search_with_score(query=query, k=top_k, filter=tag_filter)
                think_time = time.time() - start_time
                status_placeholder.empty()

                with st.expander(f"💠 已思考 (检索用时 {think_time:.2f} 秒)", expanded=False):
                    if raw_results:
                        st.markdown("已从数据库中提取相关片段作为参考...")
                    else:
                        st.markdown("未检索到强相关数据，将直接生成回答...")

            # 准备提示词
            context_text = ""
            for i, item in enumerate(raw_results):
                doc = item[0] if isinstance(item, (tuple, list)) else item
                source = doc.metadata.get("source", "未知来源")
                clean_content = doc.page_content.replace('///', '').replace('***', '').replace('---', '')
                context_text += f"\n[资料 {i + 1} - {source}]：\n{clean_content}\n"

            history_text = ""
            for msg in chat_history.messages[-40:]:
                role = "用户" if msg.type == "human" else "AI"
                history_text += f"{role}: {msg.content}\n"
            if not history_text: history_text = "无"

            if llm:
                final_prompt = RAG_PROMPT.format(chat_history=history_text, context=context_text, query=query)
                response_stream = llm.stream(final_prompt)
                full_response = st.write_stream(response_stream)

                chat_history.add_user_message(query)
                chat_history.add_ai_message(full_response)
            else:
                st.info("未配置大模型 API Key。")

            if raw_results:
                st.markdown("<br><b style='color:#475569; font-size: 0.95rem;'>📚 溯源参考资料</b>",
                            unsafe_allow_html=True)
                for i, item in enumerate(raw_results):
                    if isinstance(item, (tuple, list)) and len(item) == 2:
                        doc, distance = item
                    else:
                        doc, distance = item, None

                    topic = doc.metadata.get("topic", "unknown")
                    source_url = doc.metadata.get("source", "")
                    title = doc.metadata.get("title", "")
                    image_paths = parse_image_paths(doc.metadata.get("image_paths", ""))
                    score = max(0.0, 1.0 - float(distance)) if distance is not None else 0.0
                    source_label = title if title else source_url
                    body_html = render_paragraphs(doc.page_content.replace('///', '').replace('***', ''))

                    # 渲染纯白资料卡片
                    card_html = f'<div class="result-card"><div class="card-header"><span class="card-title">[{i + 1}] {source_label}</span><div class="card-badges"><span class="badge badge-topic">{topic}</span><span class="badge badge-score">相似度: {score:.4f}</span></div></div><div class="card-body">{body_html}</div>'
                    if source_url.startswith("http"):
                        card_html += f'<div class="card-footer"><a href="{source_url}" target="_blank">查看原文 &rarr;</a></div>'
                    card_html += "</div>"

                    st.markdown(card_html, unsafe_allow_html=True)

                    # 渲染配图
                    if image_paths:
                        cols = st.columns(min(len(image_paths), 3))
                        for j, img_path in enumerate(image_paths):
                            with cols[j % 3]:
                                st.image(img_path, use_container_width=True, caption="匹配到的资料图片")

# Tab 2: 知识引擎监控面板
with tab_dashboard:
    st.markdown("### 知识引擎监控面板")
    index_info = get_index_info()

    if not index_info:
        st.caption("未找到索引。请先运行数据注入脚本构建知识库。")
    else:
        num_docs = int(index_info.get("num_docs", b"0").decode()) if isinstance(index_info.get("num_docs"), bytes) else int(index_info.get("num_docs", 0))
        failures = int(index_info.get("hash_indexing_failures", b"0").decode()) if isinstance(index_info.get("hash_indexing_failures"), bytes) else int(index_info.get("hash_indexing_failures", 0))

        st.markdown("---")
        c1, c2, c3, c4 = st.columns(4)
        with c1:
            st.metric("知识片段总数", num_docs)
        with c2:
            st.metric("唯一主题数", len(dynamic_topics))
        with c3:
            st.metric("向量维度", 512)
        with c4:
            st.metric("索引失败次数", failures)