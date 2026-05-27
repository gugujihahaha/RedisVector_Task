"""
Vector Search Engine — Dynamic Multi-Tab AI Exploration System
Redis Stack + LangChain + HNSW
"""

import os
import re
import streamlit as st
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_redis import RedisVectorStore
from redis import Redis
from redisvl.query.filter import Tag


INDEX_NAME = "rag_knowledge_base"

st.set_page_config(
    page_title="Vector Search Engine",
    layout="wide",
)


# ================================================================
#                   Design System CSS — 现代高定排版
# ================================================================
st.markdown("""
<style>
    /* ----- 全局重置 ----- */
    .stApp {
        background: #FCFCFC;
    }
    header[data-testid="stHeader"],
    [data-testid="stDecoration"],
    #MainMenu, footer, .stDeployButton {
        display: none !important;
    }

    /* ----- 正文排版 ----- */
    .stMarkdown p {
        line-height: 1.95 !important;
        letter-spacing: 0.03em !important;
        color: #334155;
        font-size: 0.95rem;
        margin-bottom: 1.6em !important;
        text-align: justify !important;
        text-justify: inter-word !important;
    }
    .stMarkdown li, .stText {
        line-height: 1.8;
        letter-spacing: 0.03em;
        color: #334155;
        font-size: 0.95rem;
        margin-bottom: 0.8em;
    }
    h1, h2, h3 {
        color: #0F172A !important;
        font-weight: 600 !important;
        letter-spacing: 0.03em;
    }
    h1 { font-size: 1.5rem !important; }
    h2 { font-size: 1.1rem !important; }
    h3 { font-size: 0.95rem !important; }

    /* ----- 侧边栏 ----- */
    [data-testid="stSidebar"] {
        background: #F8FAFC;
        border-right: 1px solid #E2E8F0;
    }
    [data-testid="stSidebar"] .stMarkdown {
        color: #475569;
    }
    [data-testid="stSidebar"] [data-testid="stCaptionContainer"] {
        color: #94A3B8;
    }

    /* ----- 结果卡片 ----- */
    .result-card {
        background: #FAFAFA;
        border: 1px solid #E2E8F0;
        border-radius: 16px;
        padding: 24px;
        margin-bottom: 20px;
        box-shadow: 0 4px 6px -1px rgba(0,0,0,0.05),
                    0 2px 4px -1px rgba(0,0,0,0.03);
        transition: box-shadow 0.15s ease;
    }
    .result-card:hover {
        box-shadow: 0 10px 15px -3px rgba(0,0,0,0.06),
                    0 4px 6px -2px rgba(0,0,0,0.04);
    }

    .card-header {
        display: flex;
        align-items: center;
        justify-content: space-between;
        margin-bottom: 16px;
        padding-bottom: 12px;
        border-bottom: 1px solid #F1F5F9;
        flex-wrap: wrap;
        gap: 8px;
    }
    .card-title {
        font-weight: 600;
        font-size: 0.95rem;
        color: #1E293B;
        letter-spacing: 0.04em;
    }
    .card-badges {
        display: flex;
        gap: 8px;
        align-items: center;
    }
    .badge {
        display: inline-block;
        padding: 3px 10px;
        border-radius: 999px;
        font-size: 0.75rem;
        font-weight: 500;
        letter-spacing: 0.06em;
    }
    .badge-topic {
        background: #F1F5F9;
        color: #64748B;
    }
    .badge-score {
        background: #ECFDF5;
        color: #0F766E;
    }

    .card-body {
        margin-bottom: 20px;
        font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", "PingFang SC",
                     "Microsoft YaHei", "Helvetica Neue", sans-serif;
    }
    .card-para {
        display: block;
        line-height: 2.0 !important;
        letter-spacing: 0.03em !important;
        color: #334155;
        font-size: 0.92rem;
        margin-bottom: 0;
        text-align: justify !important;
        text-justify: inter-word !important;
        text-indent: 2em;
    }

    .card-images {
        margin: 16px 0;
    }
    .card-images img {
        border-radius: 10px;
        border: 1px solid #E2E8F0;
        margin-bottom: 8px;
    }

    .card-footer {
        margin-top: 14px;
        padding-top: 12px;
        border-top: 1px solid #F1F5F9;
    }
    .card-footer a {
        color: #94A3B8;
        font-size: 0.82rem;
        text-decoration: none;
        letter-spacing: 0.04em;
        transition: color 0.15s;
    }
    .card-footer a:hover {
        color: #475569;
        text-decoration: underline;
    }

    /* ----- Chat 消息微调 ----- */
    [data-testid="stChatMessage"] {
        padding: 0.25rem 0;
        background: transparent;
    }

    /* ----- Expander 去除（dashboard 保留基础样式） ----- */
    [data-testid="stExpander"] summary {
        font-weight: 500;
        color: #475569;
    }

    /* ----- 指标卡 ----- */
    [data-testid="stMetric"] {
        background: #FAFAFA;
        border: 1px solid #E2E8F0;
        border-radius: 12px;
        padding: 16px;
    }
    [data-testid="stMetric"] label {
        color: #64748B !important;
        font-size: 0.78rem;
        letter-spacing: 0.06em;
    }
    [data-testid="stMetric"] [data-testid="stMetricValue"] {
        color: #0F172A !important;
        font-size: 1.6rem;
        font-weight: 600;
    }
</style>
""", unsafe_allow_html=True)


# ================================================================
#                Redis 连接与向量存储初始化
# ================================================================
@st.cache_resource
def init_vector_store():
    client = Redis(host="localhost", port=6379)
    client.ping()
    embeddings = HuggingFaceEmbeddings(
        model_name="BAAI/bge-small-zh-v1.5"
    )
    return RedisVectorStore.from_existing_index(
        index_name=INDEX_NAME,
        embedding=embeddings,
        redis_client=client,
    )


vector_store = init_vector_store()
client = Redis(host="localhost", port=6379)


# ================================================================
#                    动态获取真实分类标签
# ================================================================
@st.cache_data(ttl=30)
def get_dynamic_topics() -> list[str]:
    try:
        raw = client.execute_command("FT.TAGVALS", INDEX_NAME, "topic")
        if raw is None:
            return []
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
            val = next(it)
            info[key] = val
        return info
    except Exception:
        return {}


def render_paragraphs(content: str) -> str:
    """
    将文本按 \\n\\n 拆分为段落，每段包裹在 <p class="card-para"> 中，
    实现首行缩进 + 两端对齐的书刊级排版。空段落跳过。
    """
    paragraphs = [p.strip() for p in content.split("\n\n") if p.strip()]
    if not paragraphs:
        return f'<p class="card-para">{content.strip()}</p>'
    return "\n".join(f'<p class="card-para">{p}</p>' for p in paragraphs)


def parse_image_paths(raw: str) -> list[str]:
    if not raw:
        return []
    candidates = raw.split(",")
    return [p.strip() for p in candidates if p.strip() and os.path.isfile(p.strip())]


# ================================================================
#                        侧边栏 (Sidebar)
# ================================================================
dynamic_topics = get_dynamic_topics()
topic_options = ["All"] + dynamic_topics

with st.sidebar:
    st.markdown("## 检索参数")
    top_k = st.slider("Top K", 1, 5, 3)
    if dynamic_topics:
        tag_option = st.selectbox("分类过滤", options=topic_options)
    else:
        tag_option = st.selectbox("分类过滤", options=["All"])
        st.caption("知识库中暂无分类标签。")

# ================================================================
#                       Header
# ================================================================
st.title("向量检索引擎")
st.caption("基于结构化知识库的动态语义搜索。")
st.markdown("---")

# ================================================================
#                       双 Tab 布局
# ================================================================
tab_search, tab_dashboard = st.tabs([
    "智能探索空间",
    "知识引擎监控面板",
])

# ================================================================
#              Tab 1: 智能探索空间
# ================================================================
with tab_search:
    query = st.chat_input("向知识库提问，例如：什么是 ACID？...")

    if query:
        with st.chat_message("user"):
            st.markdown(query)

        if tag_option == "All" or not dynamic_topics:
            tag_filter = None
        else:
            tag_filter = Tag("topic") == tag_option

        with st.spinner("正在搜索向量空间..."):
            raw_results = vector_store.similarity_search_with_score(
                query=query,
                k=top_k,
                filter=tag_filter,
            )

        with st.chat_message("assistant"):
            if not raw_results:
                st.markdown(
                    "*未在向量空间中找到匹配的知识片段，"
                    "请尝试优化查询语句或放宽分类过滤条件。*"
                )
            else:
                st.markdown(
                    "*基于底层向量空间的相似度比对，"
                    "为您检索到以下高相关度知识片段：*"
                )
                st.markdown("---")

                for i, item in enumerate(raw_results):
                    if isinstance(item, (tuple, list)) and len(item) == 2:
                        doc, distance = item
                    else:
                        doc, distance = item, None

                    topic = doc.metadata.get("topic", "unknown")
                    source_url = doc.metadata.get("source", "")
                    title = doc.metadata.get("title", "")
                    raw_paths = doc.metadata.get("image_paths", "")
                    image_paths = parse_image_paths(raw_paths)

                    if distance is not None:
                        score = max(0.0, 1.0 - float(distance))
                    else:
                        score = 0.0

                    source_label = title if title else source_url

                    # ---- 卡片 HTML ----
                    body_html = render_paragraphs(doc.page_content)
                    card_html = f"""
                    <div class="result-card">
                        <div class="card-header">
                            <span class="card-title">{source_label}</span>
                            <div class="card-badges">
                                <span class="badge badge-topic">{topic}</span>
                                <span class="badge badge-score">{score:.4f}</span>
                            </div>
                        </div>
                        <div class="card-body">
                            {body_html}
                        </div>
                    """

                    if source_url.startswith("http"):
                        card_html += f"""
                        <div class="card-footer">
                            <a href="{source_url}" target="_blank">View Source Document &rarr;</a>
                        </div>
                        """

                    card_html += "</div>"

                    st.markdown(card_html, unsafe_allow_html=True)

                    # ---- 渲染抽取的图片 ----
                    if image_paths:
                        st.markdown('<div style="height:20px;"></div>', unsafe_allow_html=True)
                        cols = st.columns(min(len(image_paths), 3))
                        for j, img_path in enumerate(image_paths):
                            with cols[j % 3]:
                                st.image(
                                    img_path,
                                    use_container_width=True,
                                    caption="教材原页扫描图",
                                )

                    st.markdown(
                        '<div style="height:24px;"></div>',
                        unsafe_allow_html=True,
                    )

# ================================================================
#              Tab 2: 知识引擎监控面板
# ================================================================
with tab_dashboard:
    st.markdown("### 知识引擎监控面板")
    st.caption("Redis Stack 实时索引健康度与统计信息。")

    index_info = get_index_info()

    if not index_info:
        st.caption("未找到索引。请先运行 build_knowledge.py 构建知识库。")
    else:
        num_docs = index_info.get("num_docs", 0)
        if isinstance(num_docs, bytes):
            num_docs = int(num_docs.decode())
        else:
            num_docs = int(num_docs)

        failures = index_info.get("hash_indexing_failures", 0)
        if isinstance(failures, bytes):
            failures = int(failures.decode())
        else:
            failures = int(failures)

        num_topics = len(dynamic_topics)

        st.markdown("---")

        c1, c2, c3, c4 = st.columns(4)
        with c1:
            st.metric("知识片段总数", num_docs)
        with c2:
            st.metric("唯一主题数", num_topics)
        with c3:
            st.metric("向量维度", 512)
        with c4:
            st.metric("索引失败次数", failures, delta=None if failures == 0 else f"{failures}")

        st.markdown("---")

        if index_info:
            st.caption("FT.INFO 原始输出：")
            with st.expander("查看完整索引元数据"):
                filtered = {
                    k: (v.decode() if isinstance(v, bytes) else v)
                    for k, v in index_info.items()
                    if not isinstance(v, (bytes, bytearray)) or len(v) < 5000
                }
                st.json(filtered)
