# Redis Vector 向量数据库调研报告

> **课程：**《数据库原理》期末调研大作业
> **调研方向：** 向量数据库 —— Redis Vector
> **前置条件：** 一台能上网的电脑（Windows / macOS / Linux 均可）

---

## 一、调研背景

随着大语言模型（LLM）的普及，向量数据库作为 RAG（检索增强生成）架构的核心基础设施，成为数据库领域的研究热点。本小组选择 **Redis Vector** 作为调研对象，它是 Redis Stack 的原生向量扩展模块，基于 HNSW（Hierarchical Navigable Small World）近似最近邻算法实现高效向量检索。

本次调研重点关注 Redis Vector 的 **CRUD 操作**（增删改查），包括：
- 索引的创建与删除（`FT.CREATE` / `FT.DROPINDEX`）
- 向量数据的写入与更新（`HSET` + 向量序列化）
- KNN 相似度检索（`FT.SEARCH` 搭配 `=>[KNN $K @vec $BLOB]` 语法）
- Tag 过滤 + 向量混合查询（metadata 映射与联合过滤）

---

## 二、环境要求

| 软件 | 最低版本 | 说明 |
|------|---------|------|
| Docker Desktop | 24.0+ | 用于运行 Redis Stack 容器 |
| Python | 3.10+ | 运行实验脚本 |
| Git | 任意版本 | 拉取代码 |

---

## 三、快速开始（三步走）

### 步骤 1：启动 Redis Stack 容器

打开终端，进入项目目录，执行：

```bash
docker compose up -d
```

这行命令会：
- 自动拉取 `redis/redis-stack:latest` 镜像（约 1.5GB，仅首次需要下载）
- 创建名为 `redis-vector-lab` 的容器
- 暴露两个端口：
  - `6379` —— Redis 服务（Python 脚本连接此端口）
  - `8001` —— RedisInsight 可视化管理面板（浏览器访问 http://localhost:8001）
- 开启 AOF 持久化（数据不会因容器重启而丢失）

验证容器是否正常运行：

```bash
docker ps --filter "name=redis-vector-lab"
```

如果看到 `STATUS` 列显示 `Up`，说明启动成功。

### 步骤 2：安装 Python 依赖

进入项目目录，创建并激活虚拟环境，然后安装依赖：

```bash
# 创建虚拟环境（仅首次）
python -m venv venv

# 激活虚拟环境
# Windows:
venv\Scripts\activate
# macOS / Linux:
source venv/bin/activate

# 安装依赖
pip install redis numpy langchain-redis langchain-community langchain-core
```

### 步骤 3：运行实验脚本

```bash
# 基础 CRUD 操作演示（FT.CREATE 建索引 + HSET 写入 + KNN 检索）
python basic_ops.py

# 混合检索演示（Tag 过滤 + 向量相似度）
python rag_demo.py
```

---

## 四、实验脚本说明

### `basic_ops.py` —— Redis Vector 底层 CRUD 操作

这是本次调研的核心实验脚本，演示了 Redis Vector 最基本的增删改查流程。

**涵盖的知识点：**

1. **FT.CREATE 显式建索引**：使用原始 Redis 命令创建向量索引，Schema 包括：
   - `title`（TEXT）—— 全文搜索字段
   - `category`（TAG）—— 精确分类过滤
   - `embedding`（VECTOR HNSW FLOAT32 COSINE）—— 向量字段

2. **向量数据写入（Create）**：使用 numpy 生成随机向量，通过 `struct.pack` 将 float32 数组序列化为 bytes，再通过 `HSET` 存入 Redis Hash

3. **向量数据读取（Read）**：使用 `HGETALL` 读取 Hash 中存储的完整记录，包括反序列化向量二进制

4. **KNN 检索（Query）**：使用 `FT.SEARCH` 执行 K 近邻查询，语法为 `"*=>[KNN $K @embedding $query_vec AS score]"`，返回按相似度排序的结果

5. **索引删除（Delete）**：使用 `FT.DROPINDEX` 删除索引，并对比 Schema-less 方式下直接 `DEL` key 的区别

**代码中可抄录的论文素材：**
- 文件顶部有约 40 行中文注释，详细论述"Redis 原生 Schema-less 为何 Vector 需要强 Schema"，可直接摘录到调研报告中。

### `rag_demo.py` —— LangChain + Redis Vector 混合检索

本脚本演示了将 Redis Vector 与 LangChain 框架集成后的高级查询能力。

**涵盖的知识点：**

1. **metadata_schema 映射**：通过 `RedisConfig.metadata_schema` 将 Document 的 metadata 字段（如 `topic`、`source`）映射为 Redis 的 TagField，实现元数据索引

2. **FakeEmbeddings 占位**：使用 LangChain 内置的 `FakeEmbeddings` 替代真实 Embedding 模型，无需 API Key 即可跑通流程

3. **混合检索**：演示"Tag 过滤 + 向量相似度"的联合查询
   - 场景 A：纯向量检索（可能混入不相关主题）
   - 场景 B：先过滤 `topic='database'`，再做向量排序（精准命中）

---

## 五、常见问题排查

### Q1: Docker 启动失败，报端口被占用？

```bash
# 检查是谁占用了 6379 端口
# Windows:
netstat -ano | findstr :6379
# macOS / Linux:
lsof -i :6379
```

找到占用进程后，停止它再重试。也可以修改 `docker-compose.yml` 中的宿主机端口映射。

### Q2: Python 脚本连接 Redis 失败？

```bash
# 检查容器是否在运行
docker ps --filter "name=redis-vector-lab"

# 如果容器已停止，重新启动
docker compose up -d
```

### Q3: 想完全清空数据重新开始？

```bash
# 停止并删除容器和数据卷
docker compose down -v

# 重新启动
docker compose up -d
```

### Q4: pip 安装速度慢？

使用国内镜像源：

```bash
pip install -i https://pypi.tuna.tsinghua.edu.cn/simple redis numpy langchain-redis langchain-community langchain-core
```

### Q5: 容器启动后发现库是空的？

**现象：** 容器正常运行，但用 `redis-cli` 连上去 `KEYS *` 发现之前存入的数据全部消失了，索引也不存在。

**根本原因：** Docker Compose 在启动时，如果找不到 `docker-compose.yml` 中声明的外部数据卷，会**自动创建一个带项目前缀的新空数据卷**（例如自动生成 `redisvector_redis_data` 而非你之前使用的 `redis_rag_data`），导致绑定到了一个新的空白存储上。

**解决方案：** 检查并修正 `docker-compose.yml` 中的 volumes 配置，确保显式指定了 `name` 为真实存在的数据卷名称：

```yaml
volumes:
  redis_data:
    external: true
    name: redis_rag_data    # 必须与你之前灌库时使用的数据卷名称一致
```

可以通过以下命令查看当前系统中实际存在的数据卷：

```bash
docker volume ls
```

找到之前灌入数据时使用的卷名（如 `redis_rag_data`），将其填入上述 `name` 字段即可恢复访问原有数据。

---

## 六、项目文件结构

```
RedisVector_Task/
├── docker-compose.yml    # Docker 编排文件
├── basic_ops.py          # 基础 CRUD 操作演示（FT.CREATE + HSET + KNN）
├── rag_demo.py           # 混合检索演示（LangChain + Tag 过滤）
├── build_knowledge.py    # 网页爬取灌库脚本
├── build_pdf_knowledge.py    # 本地 PDF 教材批量解析入库
├── app.py                # Streamlit Web 可视化界面
├── README.md             # 本文件
└── venv/                 # Python 虚拟环境（需自行创建）
```

---

## 七、停止和清理

```bash
# 停止容器（保留数据）
docker compose stop

# 停止并删除容器（保留数据卷）
docker compose down

# 彻底清理（删除数据卷）
docker compose down -v
```

---

## 八、扩展功能：知识库构建

> 以下为调研过程中搭建的辅助实验工具，用于验证 Redis Vector 在实际 RAG 场景中的检索效果。

### 8.1 网页爬取灌库 (`build_knowledge.py`)

自动爬取指定网页 → 文本语义切片 → BGE-m3 向量化 → 存入 Redis Vector。

**新增依赖：**

```bash
pip install beautifulsoup4 lxml requests langchain-huggingface sentence-transformers
```

**运行：**

```bash
python build_knowledge.py
```

**自定义数据源：** 编辑脚本中的 `DATA_SOURCES` 列表，修改 `url`、`topic`、`source` 字段即可。

### 8.2 本地 PDF 教材入库 (`build_pdf_knowledge.py`)

解析 `JiaoCai/` 文件夹下的 PDF 教材，自动识别文件名映射 topic，增量追加到 Redis 知识库中。

**依赖安装：**

```bash
pip install pypdf
```

**运行：**

```bash
python build_pdf_knowledge.py
```

### 8.3 Web 可视化演示 (`app.py`)

基于 Streamlit 的交互式查询界面，用于在课程答辩中进行 Redis Vector 检索效果的现场演示。

**安装与启动：**

```bash
pip install streamlit
streamlit run app.py
```

浏览器打开 http://localhost:8501 即可使用。

---

## 九、项目亮点

### 9.1 Redis CRUD 操作完整覆盖

本次调研的核心成果：从底层 `redis` 库（`basic_ops.py`）到上层 LangChain 封装（`rag_demo.py`），完整演示了 Redis Vector 的索引创建、数据写入、向量检索、结果过滤、索引删除的全生命周期操作。代码中每一处原始 Redis 命令调用均有详细注释，可直接作为调研报告的实验佐证。

### 9.2 Docker 容器化持久化

使用 `docker-compose.yml` 实现数据卷强绑定，通过 `external: true` 声明外部命名卷并指定 `name: redis_rag_data`，确保即使执行 `docker compose down` 删除容器后，向量数据仍然完好地保存在宿主机磁盘上。重新 `docker compose up -d` 即可秒级恢复。

### 9.3 前端交互设计

`app.py` 的 UI 实现包括：
- 输入框固定在页面底部，符合现代对话式 AI 产品的交互习惯
- 优化了 Streamlit 的 `st.rerun()` 触发逻辑，消除搜索时的页面闪烁
- LLM 生成的回答以打字机效果逐字呈现

### 9.4 内网穿透方案

系统采用 **cpolar** 作为内网穿透方案，支持 WebSocket 长连接，提供公网 HTTPS 链接，评审老师可直接打开访问，无需配置本地环境。

**首次配置：**

```bash
cpolar authtoken <您的TOKEN密钥>
```

**启动映射：**

```bash
cpolar http 8501
```

---

### 高频报错排查

#### 问题 A：无法将"cpolar"项识别为 cmdlet

**核心原因：** 安装 cpolar 后未重启终端，或 PATH 未刷新。

**解决方案：**
1. 彻底关闭当前终端窗口，重新打开后再试。
2. 若依旧报错，可直接切入 cpolar 安装路径：`cd "C:\Program Files\cpolar"` 后执行 `.\cpolar http 8501`。

#### 问题 B：无法将"streamlit"项识别为 cmdlet

**核心原因：** 终端未激活 Python 虚拟环境（venv）。

**解决方案：**
1. 先执行虚拟环境激活：`.\venv\Scripts\activate`
2. 或使用 Python 模块化方式绕过：`python -m streamlit run app.py`
