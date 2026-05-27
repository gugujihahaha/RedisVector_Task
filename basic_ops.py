"""
Redis Vector 基础操作演示 —— 使用底层 redis-py 直接执行 Redis 命令
核心考点：FT.CREATE 显式建索引 | HNSW + COSINE | KNN 检索
"""

import numpy as np
import redis

# ============================================================
# 【论文关键论述】Redis 原生是 Schema-less 的，
# 为什么 Redis Vector 在这里必须预定义强 Schema？
# ============================================================
#
# 一句话总结：
# 向量相似度搜索依赖特定的数据结构（向量索引）和距离算法，
# 而这两者都不可能在一个"无 Schema"的环境中自动推断出来。
#
# 详细解释（可直接引用至论文）：
#
# 1. 向量索引是一种高度特化的数据结构
#    Redis 原生 KV 存储只需要知道 key 和 value，它可以接受任意类型
#    的数据而无须预先声明。但向量检索完全不同：Redis 必须在内存中
#    为向量构建 HNSW（Hierarchical Navigable Small World）图结构。
#    构建这个图需要提前知道：(a) 向量的维度是多少；(b) 距离度量
#    用欧几里得、内积还是余弦相似度；(c) 向量值用什么精度存储
#    （FLOAT32 / FLOAT64）。如果不预先声明这些参数，Redis 连
#    内存布局都分配不了。
#
# 2. 距离度量的选择不可逆
#    COSINE（余弦相似度）和 L2（欧几里得距离）在数学上的计算方式
#    和在 HNSW 图中的邻接关系完全不同。如果 Schema 没有提前约定
#    度量方式，检索结果的排序就没有可预期的语义。而且索引一旦创建，
#    度量方式不可更改，这要求建库时必须用 Schema 明确锁定。
#
# 3. 过滤字段需要类型标注以实现二级索引
#    在混合查询（向量 + 过滤条件）场景下，TAG 和 NUMERIC 类型的
#    字段需要建立倒排索引或范围索引，这些同样需要在 Schema 中声明。
#    例如，"分类"字段声明为 TAG 后，Redis 会为它建立单独的倒排索引，
#    在查询时先按 TAG 过滤候选集再算向量距离，大幅减少计算量。
#
# 4. 与纯 KV 的根本差异
#    纯 KV 模式下，Redis 的职责仅仅是"给定 key，返回值"。至于
#    value 内部是什么结构、怎么解释，完全由客户端自己负责。
#    而 RediSearch 向量模块接管了索引构建和查询执行，它必须理解
#    数据内部的语义结构（哪个字段存什么、怎么比较），这就必然要求
#    严格的 Schema 定义。换句话说：
#       • 纯 KV Redis：存储层，Schema-less，只认 key-value
#       • Redis Vector（RediSearch）：搜索引擎，Schema-aware，
#         需要声明字段类型以便构建倒排/向量索引
#
# 综上，Redis Vector 的"强 Schema"不是对 Redis 哲学的反叛，
# 而是向量搜索引擎的天然要求——索引结构、距离度量、过滤字段的
# 类型信息必须在数据入库前就确定下来，引擎才能高效运作。
# ============================================================


def main():
    # ---- 1. 连接 Redis ----
    r = redis.Redis(host="localhost", port=6379, decode_responses=False)
    r.ping()
    print("[OK] Redis 连接成功")

    # ---- 2. 删除可能存在的旧索引（保证幂等运行） ----
    try:
        r.execute_command("FT.DROPINDEX", "idx:books", "DD")
    except redis.exceptions.ResponseError:
        pass  # 索引不存在则忽略

    # ---- 3. 使用 FT.CREATE 显式创建向量索引 ----
    # Schema 组成：
    #   title  → TEXT        (全文可搜索)
    #   category → TAG       (分类过滤/聚合)
    #   embedding → VECTOR   (HNSW 算法, FLOAT32 精度, COSINE 距离)
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

    # ---- 4. 插入 3 条模拟图书数据 ----
    rng = np.random.default_rng(42)
    books = [
        {"title": "数据库系统概论", "category": "database"},
        {"title": "Redis 深度历险", "category": "database"},
        {"title": "百年孤独", "category": "literature"},
    ]

    for i, book in enumerate(books):
        key = f"book:{i + 1}"
        # 生成 4 维随机向量并转为 bytes
        vec = rng.random(4, dtype=np.float32).tobytes()
        r.hset(key, mapping={
            "title": book["title"],
            "category": book["category"],
            "embedding": vec,
        })
        print(f"[OK] 已插入 {key}: {book['title']} (category={book['category']})")

    # ---- 5. 执行 KNN 向量检索 ----
    # 构造一个查询向量（模拟用户想找"数据库相关"书的语义向量）
    query_vec = np.array([0.1, 0.9, 0.3, 0.7], dtype=np.float32).tobytes()

    print("\n>>> KNN 检索（返回最相似的 2 条，按相似度降序）：")
    result = r.execute_command(
        "FT.SEARCH", "idx:books",
        "*=>[KNN 2 @embedding $qvec AS score]",
        "SORTBY", "score",
        "PARAMS", "2", "qvec", query_vec,
        "RETURN", "3", "title", "category", "score",
        "DIALECT", "2",
    )

    # 解析并展示结果
    total = result[0]
    print(f"共命中 {total} 条记录：")
    for i in range(1, len(result), 2):
        key = result[i]
        fields = result[i + 1]
        info = dict(zip(fields[::2], fields[1::2]))
        title = info.get(b"title", b"").decode()
        category = info.get(b"category", b"").decode()
        score = float(info.get(b"score", b"0"))
        print(f"  [{key.decode()}] {title} | 分类: {category} | 相似度得分: {score:.6f}")


if __name__ == "__main__":
    main()
