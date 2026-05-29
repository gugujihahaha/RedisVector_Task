"""
Redis Vector RAG 演示
"""
import warnings
warnings.filterwarnings("ignore")

from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_redis import RedisConfig, RedisVectorStore
from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate
from langchain_core.globals import set_llm_cache
from langchain_community.cache import RedisCache
from langchain_community.chat_message_histories import RedisChatMessageHistory

from redis import Redis

DEEPSEEK_API_KEY = "sk-ef3a4e5c1a0c437e8927f18b5a445534"

# 50 条计算机考点数据
RAW_DATA = [
    ("database", "数据库原理", "事务的ACID特性包括原子性、一致性、隔离性和持久性，是保证数据准确的核心。"),
    ("database", "MySQL底层", "InnoDB存储引擎默认使用B+树作为索引结构，其叶子节点包含完整数据，适合范围查询。"),
    ("database", "并发控制", "MVCC（多版本并发控制）通过保存数据的历史版本，实现读写不阻塞，解决了不可重复读问题。"),
    ("database", "Redis缓存", "Redis是基于内存的KV数据库，单线程架构配合IO多路复用，使其拥有极高的读写性能。"),
    ("database", "数据异常", "脏读是指一个事务读取了另一个未提交事务的数据；幻读则是两次查询得到的记录数量不一致。"),
    ("database", "关系范式", "第三范式（3NF）要求关系表中的每一列都与主键直接相关，而不能存在传递依赖。"),
    ("database", "搜索引擎", "Elasticsearch底层依赖Lucene，通过建立‘倒排索引’，能够实现海量文本的毫秒级全文检索。"),
    ("database", "锁机制", "悲观锁认为冲突大概率会发生，因此在操作前先加锁；乐观锁通常通过版本号机制来实现。"),
    ("database", "Redis持久化", "RDB是Redis的内存快照持久化，AOF则是将执行的写命令追加到日志文件中。"),
    ("database", "向量数据库", "向量数据库通过存储高维浮点数数组，并利用HNSW等算法计算空间距离，实现语义检索。"),

    ("os", "进程管理", "进程是操作系统资源分配的基本单位，拥有独立的内存空间；而线程是CPU调度的基本单位。"),
    ("os", "死锁分析", "产生死锁的四个必要条件：互斥条件、请求和保持条件、不剥夺条件、环路等待条件。"),
    ("os", "内存分配", "分页存储管理将内存划分为固定大小的物理块，解决了内存碎片问题，但可能产生内部碎片。"),
    ("os", "虚拟内存", "虚拟内存允许程序使用比实际物理内存更大的地址空间，其核心依赖于局部性原理和页面置换。"),
    ("os", "页面置换", "LRU（最近最少使用）算法淘汰最长时间未被访问的页面，是一种近似最优的页面置换策略。"),
    ("os", "进程通信", "常见的进程间通信（IPC）方式包括：管道、消息队列、共享内存、信号量和套接字。"),
    ("os", "同步机制", "信号量（Semaphore）是由荷兰学者Dijkstra提出的，用于解决多个进程对临界资源的互斥访问。"),
    ("os", "中断机制", "中断是操作系统并发执行的基础，分为外部硬件中断和内部软件异常（如缺页中断）。"),
    ("os", "文件系统", "FAT32、NTFS和EXT4是常见的文件系统格式，它们负责管理磁盘上的数据组织与存储。"),
    ("os", "并发问题", "临界区是指访问共享资源的那段代码，同一时刻只能允许一个进程进入临界区执行。"),

    ("network", "TCP协议", "TCP是面向连接的可靠传输协议，通过三次握手建立连接，四次挥手断开连接。"),
    ("network", "UDP协议", "UDP是无连接的尽最大努力交付协议，不保证可靠性，但传输速度快，常用于视频直播。"),
    ("network", "网络模型", "OSI七层模型包括：物理层、数据链路层、网络层、传输层、会话层、表示层和应用层。"),
    ("network", "HTTP状态码", "200表示请求成功，404表示资源未找到，500表示服务器内部错误，502为网关错误。"),
    ("network", "DNS解析", "DNS系统负责将人类可读的域名（如baidu.com）转换为计算机可寻址的IP地址。"),
    ("network", "HTTPS原理", "HTTPS在HTTP的基础上加入了SSL/TLS层，通过非对称加密交换密钥，对称加密传输数据。"),
    ("network", "拥塞控制", "TCP的拥塞控制算法包括：慢开始、拥塞避免、快重传和快恢复四个核心阶段。"),
    ("network", "IP地址", "IPv4地址长度为32位，IPv6地址长度为128位，旨在解决全球IP地址枯竭的问题。"),
    ("network", "ARP协议", "ARP（地址解析协议）的作用是在局域网中，通过目标IP地址查询目标设备的MAC地址。"),
    ("network", "长连接", "WebSocket是一种在单个TCP连接上进行全双工通信的协议，非常适合实时聊天室业务。"),

    ("algorithm", "排序算法", "快速排序采用分治策略，通过选取基准值将数组分为两部分，平均时间复杂度为O(n log n)。"),
    ("algorithm", "散列表", "哈希表通过散列函数将键映射到数组索引，处理冲突的常见方法有拉链法和开放寻址法。"),
    ("algorithm", "树结构", "二叉搜索树（BST）的左子树所有节点值小于根节点，右子树所有节点值大于根节点。"),
    ("algorithm", "图论算法", "Dijkstra算法用于求解单源最短路径问题，但它不能处理带有负权边的图。"),
    ("algorithm", "动态规划", "动态规划（DP）的核心思想是把复杂问题分解为子问题，并保存子问题的解来避免重复计算。"),
    ("algorithm", "搜索策略", "DFS（深度优先搜索）通常使用栈来实现，而BFS（广度优先搜索）则利用队列来实现。"),
    ("algorithm", "字符串匹配", "KMP算法利用部分匹配表（Next数组）避免字符串匹配时的指针回溯，时间复杂度为O(m+n)。"),
    ("algorithm", "贪心策略", "贪心算法在每一步选择中都采取当前状态下的最优解，期望从而导致全局最优结果。"),
    ("algorithm", "高级树", "红黑树是一种自平衡的二叉查找树，它通过节点着色和旋转机制，保证最坏查找时间为O(log n)。"),
    ("algorithm", "堆结构", "优先队列通常使用二叉堆来实现，堆的插入和删除最大（小）值操作的时间复杂度都是O(log n)。"),

    ("ai", "大模型原理", "Transformer架构的核心是‘自注意力机制’（Self-Attention），它使得模型能够理解上下文单词的关联。"),
    ("ai", "RAG架构", "RAG（检索增强生成）通过外挂本地知识库，为大模型提供实时、准确的上下文，有效解决了AI幻觉问题。"),
    ("ai", "深度学习", "反向传播（Backpropagation）是神经网络更新权重的基础，它利用链式法则计算损失函数的梯度。"),
    ("ai", "LangChain", "LangChain是一个开源的大模型应用开发框架，它的LCEL链式语法极大简化了提示词拼装和工具调用。"),
    ("ai", "提示词工程", "Prompt Engineering通过精确的指令设计、Few-shot（少样本）示例，引导大模型输出符合预期格式的结果。"),
    ("ai", "Embedding", "文本向量化（Embedding）将人类语言映射为高维数学空间中的坐标点，语义相近的词在空间中距离也更近。"),
    ("ai", "国产模型", "DeepSeek-V3 采用了MoE（混合专家）架构，在保持极高性能的同时大幅降低了推理和训练的算力成本。"),
    ("ai", "机器视觉", "CNN（卷积神经网络）通过卷积核提取图像的局部特征，是目前图像分类和目标检测领域的主流架构。"),
    ("ai", "过拟合", "在模型训练中，如果模型在训练集表现极好但在测试集极差，这被称为‘过拟合’，可通过正则化解决。"),
    ("ai", "微调技术", "LoRA（低秩自适应）是一种高效的大模型微调技术，它只更新极少量的参数，普通显卡也能完成训练。")
]

def main():
    print("=" * 60)
    print("Redis Vector 知识库 RAG 演示")
    print("=" * 60)

    # 1. 初始化
    client = Redis(host="localhost", port=6379)
    try: client.execute_command("FT.DROP INDEX", "idx:rag_docs", "DD")
    except: pass

    # 2. 加载模型
    print("\n[1/3] 正在加载 BGE 语义模型...")
    embeddings = HuggingFaceEmbeddings(model_name="BAAI/bge-small-zh-v1.5")
    print("[1.5/3] 正在挂载 Redis 语义缓存层...")
    set_llm_cache(RedisCache(redis_=Redis(host="localhost", port=6379)))

    # 3. 解析并组装数据
    print(f"[2/3] 正在从系统中读取 {len(RAW_DATA)} 条数据...")
    docs = []
    for topic, source, content in RAW_DATA:
        docs.append(Document(page_content=content, metadata={"source": source, "topic": topic}))

    # 4. 灌入 Redis
    print(f"[3/3] 正在向 Redis 中注入向量并建立 HNSW 图索引...")
    vector_store = RedisVectorStore.from_documents(
        docs, embeddings,
        config=RedisConfig(
            index_name="idx:rag_docs",
            redis_client=client,
            metadata_schema=[{"name": "topic", "type": "tag"}],
            embedding_dimensions=512
        )
    )
    print("✅ 知识库构建完毕！")

    # 5. 设置 LLM
    llm = ChatOpenAI(api_key=DEEPSEEK_API_KEY, base_url="https://api.deepseek.com", model="deepseek-chat") if DEEPSEEK_API_KEY else None

    prompt = PromptTemplate.from_template("""你是一个专业的计算机科学导师。
    请基于【历史聊天记录】和【参考资料】的内容来回答用户提问。
    ⚠️【格式严格要求】：
    1. 请输出排版干净、自然易读的文本。
    2. 绝对不要照抄或输出参考资料中的任何特殊分隔符（如 ///, ***, ---, === 等）。
    3. 使用标准的 Markdown 标题或列表进行排版。

    【历史聊天记录】：
    {chat_history}

    【参考资料】：
    {context}

    【用户提问】：
    {question}

    请回答：""")

    chat_history = RedisChatMessageHistory("rag_cli_session", url="redis://localhost:6379")
    chat_history.clear()

    # 6. 交互循环
    while True:
        query = input("\n 输入问题 (输入'退出'结束): ")
        if query in ['退出', 'exit']: break

        # 混合检索
        print(f"[*] Redis 正在从50条数据中检索并计算余弦相似度距离...")
        results = vector_store.similarity_search_with_score(query, k=4)
        context = ""
        for idx, (doc,score) in enumerate(results):
            context += f"[{idx + 1}] (领域: {doc.metadata['topic']} | 来源: {doc.metadata['source']}) {doc.page_content}\n"

        # G环节：生成
        if llm:
            print(" AI 正在基于历史记忆与检索内容生成解答...")

            # 读取历史记忆
            history_str = ""
            for msg in chat_history.messages[-4:]:
                role = "用户" if msg.type == "human" else "AI"
                history_str += f"{role}: {msg.content}\n"
            if not history_str:
                history_str = "无"

            # 组装发给大模型的Prompt
            final_prompt = prompt.format(
                chat_history=history_str,
                context=context,
                question=query
            )
            response = llm.invoke(final_prompt)

            print("\n" + "=" * 50)
            print(response.content)
            print("=" * 50)

            chat_history.add_user_message(query)
            chat_history.add_ai_message(response.content)
        else:
            print("\n [未配置 LLM，仅展示检索结果]:")
            print(context)

if __name__ == "__main__":
    main()