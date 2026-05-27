"""
知识库构建脚本 —— 双料数据源爬取、切片、向量化灌库
数据来源：安师大计信学院官网 + Wikipedia/MDN 数据库考点
使用 requests + BeautifulSoup 精准提取正文内容
"""

import re
import sys
import requests
from bs4 import BeautifulSoup
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_redis import RedisConfig, RedisVectorStore
from redis import Redis


# ============================================================
# 定义数据源：URL + metadata
# ============================================================
DATA_SOURCES = [
    # ---- 领域一：安徽师范大学计信学院 ----
    {
        "url": "https://ci.ahnu.edu.cn/xygk/xyjj.htm",
        "title": "安徽师范大学计算机与信息学院简介",
        "topic": "ahnu",
        "source": "安师大官网",
    },
    # ---- 领域二：数据库原理核心考点（Wikipedia 中文版） ----
    {
        "url": "https://zh.wikipedia.org/zh-cn/%E5%85%B3%E7%B3%BB%E6%95%B0%E6%8D%AE%E5%BA%93",
        "title": "关系数据库",
        "topic": "database",
        "source": "维基百科",
    },
    {
        "url": "https://zh.wikipedia.org/zh-cn/SQL",
        "title": "SQL 结构化查询语言",
        "topic": "database",
        "source": "维基百科",
    },
    {
        "url": "https://zh.wikipedia.org/zh-cn/%E6%95%B0%E6%8D%AE%E5%BA%93%E7%B4%A2%E5%BC%95",
        "title": "数据库索引",
        "topic": "database",
        "source": "维基百科",
    },
    {
        "url": "https://zh.wikipedia.org/zh-cn/%E6%95%B0%E6%8D%AE%E5%BA%93%E4%BA%8B%E5%8A%A1",
        "title": "数据库事务",
        "topic": "database",
        "source": "维基百科",
    },
    {
        "url": "https://zh.wikipedia.org/zh-cn/ACID",
        "title": "ACID 事务特性",
        "topic": "database",
        "source": "维基百科",
    },
    {
        "url": "https://zh.wikipedia.org/zh-cn/NoSQL",
        "title": "NoSQL 非关系型数据库",
        "topic": "database",
        "source": "维基百科",
    },
    {
        "url": "https://zh.wikipedia.org/zh-cn/B%2B%E6%A0%91",
        "title": "B+ 树索引结构",
        "topic": "database",
        "source": "维基百科",
    },
]


# ---- 请求头：模拟浏览器，避免被反爬拦截 ----
HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/130.0.0.0 Safari/537.36"
    ),
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    "Accept-Language": "zh-CN,zh;q=0.9,en;q=0.8",
}


def fetch_and_parse(url: str, title: str) -> str:
    """
    使用 requests 抓取网页，BeautifulSoup 提取正文。
    支持多种页面结构的自动适配：
    - Wikipedia: .mw-parser-output
    - 高校官网: .main-content, .container
    - MDN: article, .main-content
    - 通用回退: body 全文本

    返回提取到的纯文本字符串。
    """
    resp = requests.get(url, headers=HEADERS, timeout=20)
    resp.encoding = "utf-8"

    soup = BeautifulSoup(resp.text, "lxml")

    # 移除脚本、样式、导航等无关元素
    for tag in soup(["script", "style", "nav", "footer", "header",
                     "noscript", "iframe", "form", "input", "button",
                     "table", "sup", "link", "meta"]):
        tag.decompose()

    # 移除 Wikipedia 的编辑链接、引用编号等噪音
    for cls in [".reference", ".mw-editsection", ".noprint",
                ".sidebar", ".navbox", ".toc", ".thumb",
                ".mw-jump-link", ".mw-cite-backlink"]:
        for tag in soup.select(cls):
            tag.decompose()

    # ---- 策略 1: Wikipedia 正文区 ----
    content = soup.select_one(".mw-parser-output")
    if content:
        texts = []
        for p in content.find_all(["p", "li", "h2", "h3", "h4", "dd", "dt"]):
            t = p.get_text(separator=" ", strip=True)
            t = re.sub(r"\[\d+\]", "", t)  # 移除引用标记 [1] [2]
            if t and len(t) > 8:
                texts.append(t)
        if texts:
            return f"【{title}】\n\n" + "\n\n".join(texts)

    # ---- 策略 2: 通用文章区 ----
    for sel in ["article", ".main-content", ".content", "#content",
                ".container", ".post-content", ".entry-content"]:
        main = soup.select_one(sel)
        if main:
            t = main.get_text(separator="\n", strip=True)
            t = re.sub(r"\n{3,}", "\n\n", t)
            if len(t) > 200:
                return f"【{title}】\n\n" + t

    # ---- 策略 3: body 全文本回退 ----
    body = soup.find("body")
    if body:
        t = body.get_text(separator="\n", strip=True)
        t = re.sub(r"\n{3,}", "\n\n", t)
        if len(t) > 200:
            return f"【{title}】\n\n" + t

    return ""


def main():
    print("=" * 60)
    print("  Redis Vector 双料知识库构建脚本")
    print("  数据领域：安师大计信学院 + 数据库原理")
    print("=" * 60)

    # ---- 1. 连接 Redis，清理旧索引 ----
    print("\n[1/5] 正在连接 Redis 并清理旧索引...")
    client = Redis(host="localhost", port=6379)
    client.ping()
    print("  [OK] Redis 连接成功")

    try:
        client.execute_command("FT.DROPINDEX", "rag_knowledge_base", "DD")
        print("  [OK] 已删除旧索引 rag_knowledge_base")
    except Exception:
        print("  [OK] 旧索引不存在，无需清理")

    # ---- 2. 逐页爬取网页内容 ----
    print("\n[2/5] 正在爬取网页内容...")
    all_docs = []

    for i, source in enumerate(DATA_SOURCES, 1):
        url = source["url"]
        topic = source["topic"]
        src_name = source["source"]
        title = source["title"]

        print(f"  [{i}/{len(DATA_SOURCES)}] 正在爬取: {title} (topic={topic}) ...", end=" ", flush=True)

        try:
            text = fetch_and_parse(url, title)

            if not text or len(text) < 80:
                print(f"⚠ 文本量不足 ({len(text)} 字符)，跳过")
                continue

            doc = Document(
                page_content=text,
                metadata={"topic": topic, "source": url, "title": title},
            )
            all_docs.append(doc)
            print(f"OK（{len(text)} 字符）")

        except requests.exceptions.Timeout:
            print(f"✗ 请求超时，跳过")
            continue
        except requests.exceptions.ConnectionError:
            print(f"✗ 连接失败，跳过")
            continue
        except Exception as e:
            print(f"✗ 爬取失败: {type(e).__name__} — {str(e)[:80]}")
            continue

    if not all_docs:
        print("\n[ERROR] 所有网页均爬取失败，请检查网络连接。")
        sys.exit(1)

    total_chars = sum(len(doc.page_content) for doc in all_docs)
    print(f"\n  总计爬取: {len(all_docs)} 个页面, {total_chars} 字符")

    # ---- 3. 语义切片 ----
    print("\n[3/5] 正在进行语义切片 (chunk_size=1000, overlap=150)...")
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=150,
        separators=["\n\n", "\n", "。", "；", "，", " ", ""],
    )
    chunks = text_splitter.split_documents(all_docs)
    print(f"  [OK] 切片完成: {len(all_docs)} 页面 → {len(chunks)} 个语块")

    # ---- 4. 向量化并写入 Redis ----
    print("\n[4/5] 正在向量化并写入 Redis（使用 BAAI/bge-small-zh-v1.5 模型）...")
    print("  模型: BAAI/bge-small-zh-v1.5 (512维，专为中文优化)")

    embeddings = HuggingFaceEmbeddings(
        model_name="BAAI/bge-small-zh-v1.5"
    )

    metadata_schema = [
        {"name": "topic", "type": "tag"},
        {"name": "source", "type": "text"},
        {"name": "title", "type": "tag"},
    ]

    config = RedisConfig(
        index_name="rag_knowledge_base",
        redis_client=client,
        metadata_schema=metadata_schema,
        indexing_algorithm="HNSW",
        distance_metric="COSINE",
        embedding_dimensions=512,
    )

    print("  正在逐批写入语块，请耐心等待...")
    vector_store = RedisVectorStore.from_documents(
        documents=chunks,
        embedding=embeddings,
        config=config,
    )
    print(f"  [OK] 成功写入 {len(chunks)} 个语块到 Redis 索引 rag_knowledge_base")

    # ---- 5. 验证与统计 ----
    print("\n[5/5] 正在验证数据完整性...")
    total_keys = client.execute_command(
        "FT.SEARCH", "rag_knowledge_base", "*", "LIMIT", "0", "0"
    )
    print(f"  [OK] 索引中共有 {total_keys[0]} 条记录")

    for t in ["ahnu", "database"]:
        count = client.execute_command(
            "FT.SEARCH", "rag_knowledge_base",
            f"@topic:{{{t}}}", "LIMIT", "0", "0"
        )
        label = "安师大计信学院" if t == "ahnu" else "数据库原理"
        print(f"  [OK] 其中「{label}」(topic={t}) 共 {count[0]} 条")

    print("\n" + "=" * 60)
    print("  知识库构建完成！")
    print(f"  索引名称: rag_knowledge_base")
    print(f"  总语块数: {len(chunks)}")
    print(f"  Embedding 模型: BAAI/bge-small-zh-v1.5 (512维)")
    print("=" * 60)


if __name__ == "__main__":
    main()
