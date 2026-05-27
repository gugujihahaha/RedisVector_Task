"""
本地 PDF 教材批量解析 & 增量灌库脚本
读取 JiaoCai/ 文件夹下的所有 PDF，语义切片后追加到 Redis 知识库。
"""

import os
import sys
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_redis import RedisConfig, RedisVectorStore
from redis import Redis

PDF_DIR = "JiaoCai"
INDEX_NAME = "rag_knowledge_base"


def map_topic(filename: str) -> str:
    """根据文件名映射 topic 标签，供前端 Tag 过滤使用。"""
    lower = filename.lower()
    if "shujvku" in lower or "shujuku" in lower or "数据库" in lower:
        return "database"
    if "caozuoxitong" in lower or "caozuoxitong" in lower or "操作系统" in lower:
        return "os"
    return "textbook"


def main():
    print("=" * 60)
    print("  本地 PDF 教材批量解析 & 增量灌库脚本")
    print("=" * 60)

    # ---- 1. 扫描文件夹 ----
    if not os.path.isdir(PDF_DIR):
        print(f"\n[ERROR] 未找到 '{PDF_DIR}/' 文件夹。")
        print("请在项目根目录创建 JiaoCai/ 文件夹，并将 PDF 教材放入其中。")
        sys.exit(1)

    pdf_files = [
        f for f in os.listdir(PDF_DIR)
        if f.lower().endswith(".pdf")
    ]

    if not pdf_files:
        print(f"\n[ERROR] '{PDF_DIR}/' 文件夹中没有找到任何 PDF 文件。")
        print("请将需要解析的 .pdf 教材放入该文件夹后重试。")
        sys.exit(1)

    print(f"\n[1/4] 扫描 '{PDF_DIR}/' 文件夹...")
    print(f"  [OK] 发现 {len(pdf_files)} 个 PDF 文件:")
    for f in pdf_files:
        print(f"       - {f}")

    # ---- 2. 批量加载 & 元数据注入 ----
    print("\n[2/4] 正在加载 PDF 并注入元数据...")
    all_docs = []

    for pdf_file in pdf_files:
        filepath = os.path.join(PDF_DIR, pdf_file)
        title = os.path.splitext(pdf_file)[0]
        topic = map_topic(title)

        print(f"  正在处理: {pdf_file} (topic={topic}) ...", end=" ", flush=True)

        try:
            loader = PyPDFLoader(filepath)
            pages = loader.load()
        except Exception as e:
            print(f"✗ 加载失败: {type(e).__name__} — {str(e)[:80]}")
            continue

        if not pages:
            print(f"⚠ 未提取到任何文本内容，跳过")
            continue

        for doc in pages:
            doc.metadata["title"] = title
            doc.metadata["topic"] = topic

        all_docs.extend(pages)
        print(f"OK（{len(pages)} 页）")

    if not all_docs:
        print("\n[ERROR] 所有 PDF 均加载失败或内容为空，请检查文件是否损坏。")
        sys.exit(1)

    total_chars = sum(len(doc.page_content) for doc in all_docs)
    print(f"\n  总计加载: {len(pdf_files)} 个文件, {len(all_docs)} 页, {total_chars} 字符")

    # ---- 3. 语义切片 ----
    print("\n[3/4] 正在进行语义切片 (chunk_size=1000, overlap=150)...")
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=150,
        separators=["\n\n", "\n", "。", "；", "，", " ", ""],
    )
    chunks = text_splitter.split_documents(all_docs)
    print(f"  [OK] 切片完成: {len(all_docs)} 页 → {len(chunks)} 个语块")

    # ---- 4. 向量化 & 增量写入 Redis ----
    print("\n[4/4] 正在向量化并增量写入 Redis（使用 BAAI/bge-small-zh-v1.5 模型）...")
    print("  模型: BAAI/bge-small-zh-v1.5 (512 维，专为中文优化)")
    print("  策略: 增量追加，不删除现有索引数据")

    embeddings = HuggingFaceEmbeddings(
        model_name="BAAI/bge-small-zh-v1.5"
    )

    client = Redis(host="localhost", port=6379)
    client.ping()

    # 检查索引是否已存在，若存在则增量追加，否则自动创建
    try:
        client.execute_command("FT.INFO", INDEX_NAME)
        index_exists = True
    except Exception:
        index_exists = False

    if index_exists:
        # 索引已存在 → 增量追加
        print("  检测到现有索引，将以增量方式追加数据...")
        from langchain_redis import RedisVectorStore
        vector_store = RedisVectorStore.from_documents(
            documents=chunks,
            embedding=embeddings,
            redis_url="redis://localhost:6379",
            index_name=INDEX_NAME,
        )
    else:
        # 索引不存在 → 创建新索引
        print("  未检测到现有索引，将创建新索引并写入数据...")
        metadata_schema = [
            {"name": "topic", "type": "tag"},
            {"name": "source", "type": "text"},
            {"name": "title", "type": "tag"},
        ]
        config = RedisConfig(
            index_name=INDEX_NAME,
            redis_client=client,
            metadata_schema=metadata_schema,
            indexing_algorithm="HNSW",
            distance_metric="COSINE",
            embedding_dimensions=512,
        )
        vector_store = RedisVectorStore.from_documents(
            documents=chunks,
            embedding=embeddings,
            config=config,
        )

    print(f"  [OK] 成功将 {len(chunks)} 个语块增量追加到 Redis 索引 {INDEX_NAME}")

    # ---- 5. 验证 ----
    print("\n" + "=" * 60)
    total = client.execute_command("FT.SEARCH", INDEX_NAME, "*", "LIMIT", "0", "0")
    print(f"  灌库完成！索引 {INDEX_NAME} 当前共 {total[0]} 条记录")
    print("=" * 60)


if __name__ == "__main__":
    main()
