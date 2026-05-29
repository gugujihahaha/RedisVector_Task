"""
Redis Vector 基础操作演示 —— 使用底层 redis-py 直接执行 Redis 命令
核心考点：FT.CREATE 显式建索引 | HNSW + COSINE | KNN 检索 | 50条海量数据演示
"""

import numpy as np
import redis
print("redis version:", redis.__version__)
print("redis.Redis:", redis.Redis)
import inspect

# ============================================================
# 【论文关键论述】Redis 原生是 Schema-less 的，
# 为什么 Redis Vector 在这里必须预定义强 Schema？
# ============================================================
# 1. 向量索引是一种高度特化的数据结构：必须预定义维度、距离度量（COSINE）、精度。
# 2. 距离度量的选择不可逆：COSINE 与 L2 在数学上的计算方式完全不同。
# 3. 过滤字段需要类型标注以实现二级索引：TAG 类型用于建立倒排索引，大幅减少计算量。
# ============================================================

def main():
    # ---- 1. 连接 Redis ----
    r = redis.Redis(host="localhost", port=6379, decode_responses=False)
    r.ping()
    print("\n[OK] Redis 连接成功！")

    # ---- 2. 删除可能存在的旧索引 ----
    try:
        r.execute_command("FT.DROPINDEX", "idx:books", "DD")
    except redis.exceptions.ResponseError:
        pass  # 索引不存在则忽略

    # ---- 3. 使用 FT.CREATE 创建向量索引 ----
    create_cmd = [
        "FT.CREATE", "idx:books",
        "ON", "HASH",
        "PREFIX", "1", "book:",
        "SCHEMA",
        "title", "TEXT",
        "category", "TAG",
        "embedding", "VECTOR", "HNSW", "6",
        "TYPE", "FLOAT32",
        "DIM", "4",
        "DISTANCE_METRIC", "COSINE",
    ]
    r.execute_command(*create_cmd)
    print("[OK] 向量索引 idx:books 创建成功")

    # ---- 4. 批量插入 50 条专业图书数据 (Insert) ----
    rng = np.random.default_rng(42)

    # 精心构造的 50 本书，分布在四个领域
    books = [
        # 数据库领域 (15本)
        {"title": "数据库系统概念", "category": "database"}, {"title": "MySQL必知必会", "category": "database"},
        {"title": "高性能MySQL", "category": "database"}, {"title": "Redis深度历险", "category": "database"},
        {"title": "MongoDB权威指南", "category": "database"}, {"title": "PostgreSQL实战", "category": "database"},
        {"title": "Oracle体系结构", "category": "database"}, {"title": "深入理解分布式系统", "category": "database"},
        {"title": "NoSQL精粹", "category": "database"}, {"title": "数据密集型应用系统设计", "category": "database"},
        {"title": "数据库索引设计与优化", "category": "database"}, {"title": "SQL进阶教程", "category": "database"},
        {"title": "数据库事务处理", "category": "database"}, {"title": "向量数据库权威指南", "category": "database"},
        {"title": "图数据库实战", "category": "database"},
        # 操作系统领域 (15本)
        {"title": "现代操作系统", "category": "os"}, {"title": "操作系统概念", "category": "os"},
        {"title": "深入理解计算机系统", "category": "os"}, {"title": "操作系统导论", "category": "os"},
        {"title": "Linux内核设计与实现", "category": "os"}, {"title": "鸟哥的Linux私房菜", "category": "os"},
        {"title": "深入解析Windows操作系统", "category": "os"}, {"title": "操作系统真象还原", "category": "os"},
        {"title": "UNIX环境高级编程", "category": "os"}, {"title": "Linux多线程编程", "category": "os"},
        {"title": "操作系统之哲学原理", "category": "os"}, {"title": "汇编语言", "category": "os"},
        {"title": "计算机体系结构", "category": "os"}, {"title": "编译原理", "category": "os"},
        {"title": "深入理解Linux内核", "category": "os"},
        # 计算机网络领域 (10本)
        {"title": "计算机网络自顶向下", "category": "network"}, {"title": "TCP/IP详解卷1", "category": "network"},
        {"title": "图解TCP/IP", "category": "network"}, {"title": "计算机网络原理", "category": "network"},
        {"title": "深入理解计算机网络", "category": "network"}, {"title": "HTTP权威指南", "category": "network"},
        {"title": "网络编程卷1", "category": "network"}, {"title": "路由与交换技术", "category": "network"},
        {"title": "软件定义网络", "category": "network"}, {"title": "计算机网络系统方法", "category": "network"},
        # 经典文学领域 (10本)
        {"title": "百年孤独", "category": "literature"}, {"title": "围城", "category": "literature"},
        {"title": "三体", "category": "literature"}, {"title": "活着", "category": "literature"},
        {"title": "1984", "category": "literature"}, {"title": "动物农场", "category": "literature"},
        {"title": "挪威的森林", "category": "literature"}, {"title": "梦的解析", "category": "literature"},
        {"title": "乌合之众", "category": "literature"}, {"title": "人类简史", "category": "literature"}
    ]

    print("\n>>> 正在向 Redis 向量数据库中灌入 50 条数据...")
    for i, book in enumerate(books):
        key = f"book:{i + 1}"
        # 模拟大模型生成的 4 维语义向量
        vec = rng.random(4, dtype=np.float32).tobytes()
        r.hset(key, mapping={
            "title": book["title"],
            "category": book["category"],
            "embedding": vec,
        })
    print(f"[OK] 50 条测试数据全部插入完毕！")

    # ---- 5. 执行 KNN 向量检索 (Query) ----
    query_vec = np.array([0.1, 0.9, 0.3, 0.7], dtype=np.float32).tobytes()

    print("\n>>> KNN 向量检索启动：从 50 本书中寻找最相似的 TOP 5...")
    result = r.execute_command(
        "FT.SEARCH", "idx:books",
        "*=>[KNN 5 @embedding $qvec AS score]",  # 这里改成了 KNN 5
        "SORTBY", "score",
        "PARAMS", "2", "qvec", query_vec,
        "RETURN", "3", "title", "category", "score",
        "DIALECT", "2",
    )

    # 解析并展示结果
    total = result[0]
    print(f"向量空间比对完成！为您推荐以下 {total} 本高维相似书籍：")
    for i in range(1, len(result), 2):
        key = result[i]
        fields = result[i + 1]
        info = dict(zip(fields[::2], fields[1::2]))
        title = info.get(b"title", b"").decode()
        category = info.get(b"category", b"").decode()
        score = float(info.get(b"score", b"0"))
        print(f"  📌 [{key.decode()}] 《{title}》 | 领域: {category:<10} | 语义距离: {score:.6f}")

    # ---- 6. 更新与删除操作 (补充调研要求中的 Update & Delete) ----
    print("\n>>> 数据更新与删除操作演示：")

    # 6.1 更新操作：把 book:41 (百年孤独) 的分类更新
    r.hset("book:41", "category", "classic_literature")
    updated_category = r.hget("book:41", "category").decode()
    print(f"  [修改] 已将 book:41 (百年孤独) 的分类更新为: {updated_category}")

    # 6.2 删除操作：踢掉 book:42 (围城)
    r.delete("book:42")
    print("  [删除] 已将 book:42 从数据库中物理删除")

    # 验证最终数量
    count_result = r.execute_command("FT.SEARCH", "idx:books", "*", "LIMIT", "0", "0")
    print(f"  [核对] 删除操作后，当前数据库总记录数: {count_result[0]} 条")
    print("\n[SUCCESS] 增删改查全流程完美闭环！")

if __name__ == "__main__":
    main()