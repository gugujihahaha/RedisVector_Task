"""
本地 PDF 教材批量解析 & 增量灌库脚本 (OCR 图文版)
针对图片扫描版 PDF：每页截图 + pytesseract 中文 OCR → Redis 向量知识库。
"""

import os
import re
import sys
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_redis import RedisConfig, RedisVectorStore
from redis import Redis

PDF_DIR = "JiaoCai"
IMG_DIR = os.path.join(PDF_DIR, "extracted_images")
INDEX_NAME = "rag_knowledge_base"

# ---- 依赖可用性检测 ----
TESSERACT_OK = False
PDF2IMAGE_OK = False

try:
    import pytesseract
    TESSERACT_OK = True
except ImportError:
    pass

try:
    from pdf2image import convert_from_path
    PDF2IMAGE_OK = True
except ImportError:
    pass


def map_topic(filename: str) -> str:
    lower = filename.lower()
    if "shujvku" in lower or "shujuku" in lower or "数据库" in lower:
        return "database"
    if "caozuoxitong" in lower or "caozuoxitong" in lower or "操作系统" in lower:
        return "os"
    return "textbook"


def clean_ocr_text(text: str) -> str:
    """
    清洗 OCR 输出：
    1. 按双换行分割为段落
    2. 段内将单换行替换为空格（修复 OCR 的断行截断）
    3. 合并多余空白
    """
    # 规范化换行
    text = re.sub(r"\r\n", "\n", text)
    text = re.sub(r"\n{3,}", "\n\n", text)

    paragraphs = text.split("\n\n")
    cleaned_paragraphs = []
    for para in paragraphs:
        para = re.sub(r"\n", "", para)  # 段内换行 → 移除
        para = re.sub(r"\s+", " ", para)  # 合并多余空白
        para = para.strip()
        if para:
            cleaned_paragraphs.append(para)

    return "\n\n".join(cleaned_paragraphs)


def check_tesseract_chinese() -> bool:
    """检测 Tesseract 是否安装了中文语言包。"""
    try:
        langs = pytesseract.get_languages()
        return "chi_sim" in langs or "chi_tra" in langs
    except Exception:
        return False


def main():
    print("=" * 60)
    print("  本地 PDF 教材批量解析 & 增量灌库脚本 (OCR)")
    print("=" * 60)

    # ---- 依赖检查 ----
    if not PDF2IMAGE_OK:
        print("\n[ERROR] 缺少 pdf2image 库。请执行: pip install pdf2image")
        print("  此外还需安装 poppler 工具:")
        print("    Windows: 下载 poppler 并添加到 PATH")
        print("    macOS:   brew install poppler")
        print("    Linux:   sudo apt install poppler-utils")
        sys.exit(1)

    if not TESSERACT_OK:
        print("\n[ERROR] 缺少 pytesseract 库。请执行: pip install pytesseract")
        print("  此外还需安装 Tesseract-OCR 引擎:")
        print("    Windows: https://github.com/UB-Mannheim/tesseract/wiki")
        print("    macOS:   brew install tesseract")
        print("    Linux:   sudo apt install tesseract-ocr")
        sys.exit(1)

    if not check_tesseract_chinese():
        print("\n[ERROR] Tesseract 缺少中文语言包 (chi_sim)。")
        print("  请下载 chi_sim.traineddata 放入 tessdata 目录。")
        print("    Windows: https://github.com/tesseract-ocr/tessdata/blob/main/chi_sim.traineddata")
        print("    Linux:   sudo apt install tesseract-ocr-chi-sim")
        sys.exit(1)

    # ---- 1. 扫描文件夹 ----
    if not os.path.isdir(PDF_DIR):
        print(f"\n[ERROR] 未找到 '{PDF_DIR}/' 文件夹。")
        print("请在项目根目录创建 JiaoCai/ 文件夹，并将 PDF 教材放入其中。")
        sys.exit(1)

    pdf_files = [f for f in os.listdir(PDF_DIR)
                 if f.lower().endswith(".pdf")]

    if not pdf_files:
        print(f"\n[ERROR] '{PDF_DIR}/' 文件夹中没有找到任何 PDF 文件。")
        sys.exit(1)

    os.makedirs(IMG_DIR, exist_ok=True)

    print(f"\n[1/5] 扫描 '{PDF_DIR}/' 文件夹...")
    print(f"  [OK] 发现 {len(pdf_files)} 个 PDF 文件:")
    for f in pdf_files:
        print(f"       - {f}")
    print(f"  [OK] 图片输出目录: {IMG_DIR}/")
    print(f"  [OK] OCR 引擎: Tesseract (lang=chi_sim)")

    # ---- 2. 逐页转图 + OCR ----
    print("\n[2/5] 正在将 PDF 转为图像并 OCR 识别...")
    all_docs = []
    total_pages = 0

    for pdf_file in pdf_files:
        filepath = os.path.join(PDF_DIR, pdf_file)
        title = os.path.splitext(pdf_file)[0]
        topic = map_topic(title)

        print(f"  正在处理: {pdf_file} (topic={topic}) ...", flush=True)

        try:
            images = convert_from_path(filepath, dpi=200)
        except Exception as e:
            print(f"    ✗ 转换失败: {type(e).__name__} — {str(e)[:80]}")
            print(f"    (poppler 是否已安装并加入 PATH？)")
            continue

        if not images:
            print(f"    ✗ 未生成任何页面图像")
            continue

        file_pages = 0
        for page_idx, image in enumerate(images):
            page_num = page_idx + 1
            img_filename = f"{title}_page{page_num}.png"
            img_path = os.path.join(IMG_DIR, img_filename)

            # 保存整页截图
            try:
                image.save(img_path, "PNG")
            except Exception as e:
                print(f"    ✗ 保存图片失败 (page {page_num}): {e}")
                continue

            # OCR 识别
            try:
                raw_text = pytesseract.image_to_string(image, lang="chi_sim")
            except Exception as e:
                print(f"    ✗ OCR 失败 (page {page_num}): {type(e).__name__}")
                raw_text = ""

            cleaned = clean_ocr_text(raw_text)

            if not cleaned or len(cleaned) < 10:
                continue

            document = Document(
                page_content=cleaned,
                metadata={
                    "title": title,
                    "topic": topic,
                    "image_paths": img_path,
                },
            )
            all_docs.append(document)
            file_pages += 1

        total_pages += file_pages
        print(f"    OK ({file_pages} 页有效)")

    if not all_docs:
        print("\n[ERROR] 所有 PDF 均未提取到有效文本。")
        print("请确认 PDF 为文字清晰的扫描版，且 Tesseract 中文包已安装。")
        sys.exit(1)

    total_chars = sum(len(doc.page_content) for doc in all_docs)
    print(f"\n  总计: {len(pdf_files)} 文件, {total_pages} 有效 OCR 页, "
          f"{total_chars} 字符, {total_pages} 张整页截图")

    # ---- 3. 语义切片 ----
    print("\n[3/5] 正在进行语义切片 (chunk_size=1000, overlap=150)...")
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=150,
        separators=["\n\n", "\n", "。", "；", "，", " ", ""],
    )
    chunks = text_splitter.split_documents(all_docs)
    print(f"  [OK] 切片完成: {total_pages} 页 → {len(chunks)} 个语块")

    # ---- 4. 向量化 & 增量写入 ----
    print("\n[4/5] 正在向量化并增量写入 Redis...")
    print("  模型: BAAI/bge-small-zh-v1.5 (512 维)")

    embeddings = HuggingFaceEmbeddings(
        model_name="BAAI/bge-small-zh-v1.5"
    )

    client = Redis(host="localhost", port=6379)
    client.ping()

    try:
        client.execute_command("FT.INFO", INDEX_NAME)
        index_exists = True
    except Exception:
        index_exists = False

    if index_exists:
        try:
            client.execute_command(
                "FT.ALTER", INDEX_NAME, "SCHEMA", "ADD",
                "image_paths", "TEXT"
            )
            print("  [OK] 已为现有索引添加 image_paths 字段")
        except Exception:
            pass

        print("  检测到现有索引，增量追加数据...")
        vector_store = RedisVectorStore.from_documents(
            documents=chunks,
            embedding=embeddings,
            redis_url="redis://localhost:6379",
            index_name=INDEX_NAME,
        )
    else:
        print("  未检测到现有索引，创建新索引...")
        metadata_schema = [
            {"name": "topic", "type": "tag"},
            {"name": "source", "type": "text"},
            {"name": "title", "type": "tag"},
            {"name": "image_paths", "type": "text"},
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

    print(f"  [OK] 成功将 {len(chunks)} 个语块增量追加到索引 {INDEX_NAME}")

    # ---- 5. 验证 ----
    print("\n[5/5] 正在验证数据完整性...")
    total = client.execute_command("FT.SEARCH", INDEX_NAME, "*", "LIMIT", "0", "0")
    print(f"  [OK] 索引 {INDEX_NAME} 当前共 {total[0]} 条记录")
    print(f"  [OK] 整页截图已保存至: {IMG_DIR}/ ({total_pages} 张)")

    print("\n" + "=" * 60)
    print("  灌库完成！")
    print(f"  索引名称: {INDEX_NAME}")
    print(f"  新增语块: {len(chunks)}")
    print(f"  全页截图: {total_pages} 张")
    print(f"  Embedding 模型: BAAI/bge-small-zh-v1.5 (512 维)")
    print("=" * 60)


if __name__ == "__main__":
    main()
