"""
Vector Search Engine — Dynamic Multi-Tab AI Exploration System
Redis Stack + LangChain + HNSW
"""

import streamlit as st
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_redis import RedisVectorStore
from redis import Redis
from redisvl.query.filter import Tag


INDEX_NAME = "rag_knowledge_base"

st.set_page_config(
    page_title="Vector Search Engine",
    page_icon=" ",
    layout="wide",
)


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
    "✨ 智能探索空间",
    "📊 知识引擎监控面板",
])

# ================================================================
#              Tab 1: 智能探索空间 (Perplexity-style)
# ================================================================
with tab_search:
    query = st.chat_input("向知识库提问，例如：什么是 ACID？...")

    if query:
        # ---- User message ----
        with st.chat_message("user"):
            st.markdown(query)

        # ---- Construct filter ----
        if tag_option == "All" or not dynamic_topics:
            tag_filter = None
        else:
            tag_filter = Tag("topic") == tag_option

        # ---- Search ----
        with st.spinner("正在搜索向量空间..."):
            raw_results = vector_store.similarity_search_with_score(
                query=query,
                k=top_k,
                filter=tag_filter,
            )

        # ---- Assistant message ----
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

                    if distance is not None:
                        score = max(0.0, 1.0 - float(distance))
                    else:
                        score = 0.0

                    source_label = title if title else source_url
                    expander_label = f"{source_label}  |  Score: {score:.4f}"

                    with st.expander(expander_label):
                        st.markdown(doc.page_content)
                        if source_url.startswith("http"):
                            st.markdown(
                                f"[View Source Document &nearr;]({source_url})"
                            )

# ================================================================
#              Tab 2: 知识引擎监控面板 (Dashboard)
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
