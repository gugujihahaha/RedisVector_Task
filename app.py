"""
Vector Search Engine — Minimalist Enterprise UI
Redis Stack + LangChain + HNSW
"""

import streamlit as st
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_redis import RedisVectorStore
from redis import Redis
from redisvl.query.filter import Tag


st.set_page_config(
    page_title="Vector Search Engine",
    page_icon=" ",
    layout="wide",
)

# ---- Cache ----
@st.cache_resource
def init_vector_store():
    client = Redis(host="localhost", port=6379)
    client.ping()
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )
    return RedisVectorStore.from_existing_index(
        index_name="rag_knowledge_base",
        embedding=embeddings,
        redis_client=client,
    )


vector_store = init_vector_store()

# ---- Sidebar ----
with st.sidebar:
    st.markdown("## Settings")

    top_k = st.slider("Top K", 1, 5, 3)

    tag_option = st.selectbox(
        "Category",
        options=["All", "database", "ahnu"],
    )

# ---- Header ----
st.title("Vector Search Engine")
st.caption("Semantic search over structured knowledge base.")

st.markdown("---")

query = st.text_input(
    "Search",
    placeholder="Search knowledge base...",
    label_visibility="collapsed",
)

# ---- Search ----
if query.strip():
    if tag_option == "All":
        tag_filter = None
    else:
        tag_filter = Tag("topic") == tag_option

    with st.spinner("Searching..."):
        raw_results = vector_store.similarity_search_with_score(
            query=query,
            k=top_k,
            filter=tag_filter,
        )

    tag_label = f"category = {tag_option}" if tag_option != "All" else "all categories"
    st.caption(f"Results ({tag_label}, top {top_k})")

    if not raw_results:
        st.caption("No results found.")
    else:
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

            with st.container(border=True):
                st.caption(
                    f"RELEVANCE: {score:.4f}  "
                    f"|  DISTANCE: {distance:.4f}  "
                    f"|  CATEGORY: {topic.upper()}"
                    if distance is not None
                    else f"CATEGORY: {topic.upper()}"
                )

                if title:
                    st.markdown(f"**{title}**")

                st.write(doc.page_content)

                if source_url.startswith("http"):
                    st.markdown(f"[View Source Document &nearr;]({source_url})")

else:
    st.caption("Enter a query above to search the knowledge base.")
