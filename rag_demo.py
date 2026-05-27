"""
Redis Vector RAG 混合检索演示 —— 基于 LangChain RedisVectorStore
核心考点：metadata_schema 将元数据映射为 TagField | Tag 过滤 + 向量相似度混合查询
"""

from langchain_core.documents import Document
from langchain_core.embeddings import FakeEmbeddings
from langchain_redis import RedisConfig, RedisVectorStore
from redis import Redis
from redisvl.query.filter import Tag

# ============================================================
# 【论文论述素材】RAG 场景下的混合检索策略
# ============================================================
# 纯向量检索在高维空间中可能找到"语义相近但主题无关"的结果，
# 例如：搜 "数据库索引"，纯 KNN 可能把 "搜索引擎倒排索引" 或
# "Pandas DataFrame index" 都排进来。
# 解决方法是混合检索（Hybrid Search）：
#   1. 先用 TagField 做预过滤（在倒排索引中精确匹配）
#   2. 再在过滤后的候选集上计算向量相似度
# 这相当于在 SQL 层面先 WHERE category='database'，再 ORDER BY
# cosine_similarity DESC。Redis 的 RediSearch 引擎可以在一次
# FT.SEARCH 中同时完成两步，不需要客户端做二次筛选。
# ============================================================


def main():
    # ---- 1. 连接 Redis（清空旧索引，确保幂等运行） ----
    client = Redis(host="localhost", port=6379)
    client.ping()
    print("[OK] Redis 连接成功")

    try:
        client.execute_command("FT.DROPINDEX", "idx:rag_docs", "DD")
        print("[OK] 已清理旧索引")
    except Exception:
        pass

    # ---- 2. 使用 FakeEmbeddings 作为占位 Embedding 模型 ----
    # 避免依赖真实 OpenAI API Key，仅用于演示流程。
    # size=128 表示生成 128 维的虚假向量。
    embeddings = FakeEmbeddings(size=128)

    # ---- 3. 构建带有 metadata 的 Document 对象 ----
    docs = [
        Document(
            page_content="Redis 支持多种数据结构，包括 String、Hash、List、Set、SortedSet。",
            metadata={"source": "Redis入门教程", "topic": "database"},
        ),
        Document(
            page_content="HNSW 是一种高效的近似最近邻搜索算法，广泛用于向量数据库。",
            metadata={"source": "向量数据库综述", "topic": "database"},
        ),
        Document(
            page_content="《百年孤独》是哥伦比亚作家加西亚·马尔克斯的魔幻现实主义代表作。",
            metadata={"source": "文学百科", "topic": "literature"},
        ),
        Document(
            page_content="MySQL InnoDB 存储引擎使用 B+ 树作为主键索引的底层数据结构。",
            metadata={"source": "MySQL深度解析", "topic": "database"},
        ),
    ]
    print(f"[OK] 已构建 {len(docs)} 条 Document 对象")

    # ---- 4. 【核心考点】自定义 metadata_schema，将 metadata 映射为 TagField ----
    #
    # RedisConfig 的 metadata_schema 参数接收一个 list[dict]，每个 dict
    # 描述一个 metadata 字段如何映射为 Redis 索引字段：
    #   • "name": 字段名（必须与 Document.metadata 中的 key 一致）
    #   • "type": Redis 索引类型 —— "tag" 表示 TagField
    #
    # 使用 TAG 类型的好处：
    #   - 倒排索引精确匹配：filter={"topic": "database"} 可以 O(1) 定位
    #   - 支持多值标签：用分隔符（默认 "|"）可存多个标签
    #   - 聚合分组：可按 topic 分组统计
    #
    # 如果不配置 metadata_schema，所有 metadata 默认以 TEXT 类型存储，
    # 只能做全文搜索，无法做高效的精确过滤。
    #
    # RedisVectorStore 会自动为 page_content 创建 TEXT 字段，
    # 为向量创建 VECTOR 字段（维度由 Embedding 模型自动推断），
    # 无需手动声明。
    metadata_schema = [
        {"name": "source", "type": "tag"},
        {"name": "topic", "type": "tag"},
    ]

    config = RedisConfig(
        index_name="idx:rag_docs",
        redis_client=client,
        metadata_schema=metadata_schema,   # <-- 关键：metadata 到 TagField 的映射
        indexing_algorithm="HNSW",         # 使用 HNSW 近似最近邻算法
        distance_metric="COSINE",          # 余弦距离
        embedding_dimensions=128,          # 与 FakeEmbeddings 一致
    )

    print("\n>>> 正在创建向量存储（将 Documents 向量化并写入 Redis）...")
    vector_store = RedisVectorStore.from_documents(
        documents=docs,
        embedding=embeddings,
        config=config,
    )
    print("[OK] 向量存储创建完成，metadata_schema 已将 source/topic 映射为 TAG 字段")

    # ---- 5. 构建 Retriever，演示"Tag 过滤 + 向量相似度"混合查询 ----
    retriever = vector_store.as_retriever(
        search_type="similarity",
        search_kwargs={"k": 3},
    )

    # ---- 5a. 无过滤：纯向量检索 ----
    print("\n===== 场景 A：无过滤（纯向量语义检索） =====")
    results = retriever.invoke("数据库索引原理")
    for i, doc in enumerate(results):
        print(f"  [{i + 1}] topic={doc.metadata['topic']} | "
              f"source={doc.metadata['source']}")
        print(f"       {doc.page_content[:60]}...")

    # ---- 5b. 混合查询：先过滤 topic=database，再做向量检索 ----
    print("\n===== 场景 B：混合查询（先 Tag 过滤 topic='database'，再向量相似度排序） =====")
    # 使用 redisvl 的 Tag 类构造 FilterExpression：
    #   Tag("topic") == "database"  →  Redis 查询语法 @topic:{database}
    tag_filter = Tag("topic") == "database"

    filtered_results = vector_store.similarity_search(
        query="数据库索引原理",
        k=3,
        filter=tag_filter,  # <-- 这就是"混合查询"的关键参数
    )
    for i, doc in enumerate(filtered_results):
        print(f"  [{i + 1}] topic={doc.metadata['topic']} | "
              f"source={doc.metadata['source']}")
        print(f"       {doc.page_content[:60]}...")

    # ---- 5c. 对比分析 ----
    print("\n===== 对比分析 =====")
    print(f"场景 A（无过滤）返回 {len(results)} 条结果，可能包含非 database 主题的内容")
    print(f"场景 B（Tag 过滤）返回 {len(filtered_results)} 条结果，全部限定在 topic='database' 内")

    print("\n[SUCCESS] RAG 混合检索演示完成！")


if __name__ == "__main__":
    main()
