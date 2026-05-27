# Redis Vector 期末调研报告 —— 环境部署与实验指南

> 适用对象：《数据库原理》课程小组成员
> 前置条件：一台能上网的电脑（Windows / macOS / Linux 均可）

---

## 一、环境要求

| 软件 | 最低版本 | 说明 |
|------|---------|------|
| Docker Desktop | 24.0+ | 用于运行 Redis Stack 容器 |
| Python | 3.10+ | 运行实验脚本 |
| Git | 任意版本 | 拉取代码 |

## 二、快速开始（三步走）

### 步骤 1：启动 Redis Stack 容器

打开终端，进入项目目录，执行：

```bash
docker compose up -d
```

这行命令会：
- 自动拉取 `redis/redis-stack:latest` 镜像（约 1.5GB，仅首次需要下载）
- 创建名为 `redis-vector-lab` 的容器
- 暴露两个端口：
  - `6379` —— Redis 服务（我们的 Python 脚本连接这个端口）
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
# 基础操作演示（FT.CREATE 建索引 + KNN 检索）
python basic_ops.py

# RAG 混合检索演示（Tag 过滤 + 向量相似度）
python rag_demo.py
```

---

## 三、实验脚本说明

### `basic_ops.py` —— Redis Vector 底层操作

**涵盖的知识点：**

1. **FT.CREATE 显式建索引**：使用原始 Redis 命令创建向量索引，Schema 包括：
   - `title`（TEXT）—— 全文搜索字段
   - `category`（TAG）—— 精确分类过滤
   - `embedding`（VECTOR HNSW FLOAT32 COSINE）—— 向量字段

2. **向量数据写入**：使用 numpy 生成随机向量，通过 bytes 存入 Redis Hash

3. **KNN 检索**：使用 `FT.SEARCH` 执行 K 近邻查询，返回相似度排序结果

**代码中可抄录的论文素材：**
- 文件顶部有约 40 行中文注释，详细论述"Redis 原生 Schema-less 为何 Vector 需要强 Schema"

### `rag_demo.py` —— LangChain + Redis Vector 混合检索

**涵盖的知识点：**

1. **metadata_schema 映射**：通过 `RedisConfig.metadata_schema` 将 Document 的 metadata 字段（如 `topic`、`source`）映射为 Redis 的 TagField

2. **FakeEmbeddings 占位**：使用 LangChain 内置的 `FakeEmbeddings` 替代真实 OpenAI Embedding，无需 API Key

3. **混合检索**：演示"Tag 过滤 + 向量相似度"的联合查询
   - 场景 A：纯向量检索（可能混入不相关主题）
   - 场景 B：先过滤 `topic='database'`，再做向量排序（精准命中）

---

## 四、常见问题排查

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

---

## 五、项目文件结构

```
RedisVector_Task/
├── docker-compose.yml    # Docker 编排文件
├── basic_ops.py          # 基础操作演示（FT.CREATE + KNN）
├── rag_demo.py           # RAG 混合检索演示（LangChain + Tag 过滤）
├── build_knowledge.py        # 一键爬取灌库脚本（构建专属知识库）
├── build_pdf_knowledge.py    # 本地 PDF 教材批量解析入库
├── app.py                    # Streamlit Web 可视化界面
├── README.md             # 本文件
└── venv/                 # Python 虚拟环境（需自行创建）
```

---

## 六、停止和清理

```bash
# 停止容器（保留数据）
docker compose stop

# 停止并删除容器（保留数据卷）
docker compose down

# 彻底清理（删除数据卷）
docker compose down -v
```

---

## 进阶功能：一键爬取并构建专属知识库

### 功能简介

`build_knowledge.py` 是一个**全自动知识库构建脚本**。它的核心能力是：

> 自动爬取互联网上指定的网页 → 将长文本智能切分成适合检索的语义片段 → 用真实开源 AI 模型将每段文本转化为向量 → 全部存入 Redis 向量数据库。

换句话说，运行它之前，Redis 里空空如也，检索结果是"智障"级别的随机乱序；跑完这个脚本之后，系统就拥有了真实的知识储备，变成了某个领域的"专家"。我们的脚本默认爬取了以下两个领域的数据：

| 领域 | 数据来源 | 内容 |
|------|---------|------|
| 安师大计信学院 | 学院官网 | 学院简介、历史沿革等 |
| 数据库原理 | 维基百科中文版 | 关系数据库、SQL、ACID、事务、B+ 树、NoSQL 等核心考点 |

### 新增依赖安装

这个脚本比基础脚本多了爬虫和真实向量模型的依赖，需要额外安装：

```bash
pip install beautifulsoup4 lxml requests langchain-huggingface sentence-transformers
```

如果你之前已经装过，重复执行不会有副作用。如果你使用的是国内网络，建议加镜像源加速：

```bash
pip install -i https://pypi.tuna.tsinghua.edu.cn/simple beautifulsoup4 lxml requests langchain-huggingface sentence-transformers
```

### 如何运行

确保 Redis Stack 容器正在运行，然后在项目目录下执行：

```bash
python build_knowledge.py
```

### 如何自定义数据源（重点）

如果你想让系统变成其他领域的"专家"（比如爬取算法导论、操作系统等考点），只需做两步：

**第一步：** 用任意文本编辑器打开 `build_knowledge.py`。

**第二步：** 找到文件开头的 `DATA_SOURCES` 列表（大约第 30 行附近），它的结构是这样的：

```python
DATA_SOURCES = [
    {
        "url": "https://ci.ahnu.edu.cn/xygk/xyjj.htm",
        "title": "安徽师范大学计算机与信息学院简介",
        "topic": "ahnu",
        "source": "安师大官网",
    },
    {
        "url": "https://zh.wikipedia.org/zh-cn/SQL",
        "title": "SQL 结构化查询语言",
        "topic": "database",
        "source": "维基百科",
    },
    # ... 更多条目
]
```

修改方法：
- **替换网址**：把 `url` 改成你想爬取的网页地址。
- **调整分类**：把 `topic` 改成你自己的分类标签（如 `"algorithm"`、`"os"`）。这个标签会用于 Streamlit 界面的 Tag 过滤。
- **写上来源**：把 `source` 改成网页来源的名称（会显示在检索结果卡片中）。
- **增加/删除条目**：直接复制粘贴一个 `{...}` 块就能新增数据源，删掉一行就能移除。

> 注意：如果某个网页有反爬机制（如百度百科），脚本会自动跳过它，不会影响其他页面的正常爬取。

### 预期现象

运行 `python build_knowledge.py` 后，你会看到类似以下的分步进度：

```
[1/5] 正在连接 Redis 并清理旧索引...
  [OK] Redis 连接成功
  [OK] 已删除旧索引 rag_knowledge_base

[2/5] 正在爬取网页内容...
  [1/8] 正在爬取: 安徽师范大学计算机与信息学院简介 (topic=ahnu) ... OK（2659 字符）
  [2/8] 正在爬取: 关系数据库 (topic=database) ... OK（1405 字符）
  ...

[3/5] 正在进行语义切片 (chunk_size=400, overlap=50)...
  [OK] 切片完成: 8 页面 → 81 个语块

[4/5] 正在向量化并写入 Redis（使用 all-MiniLM-L6-v2 模型）...
  [OK] 成功写入 81 个语块到 Redis 索引 rag_knowledge_base

[5/5] 正在验证数据完整性...
  [OK] 索引中共有 81 条记录
  [OK] 其中「安师大计信学院」(topic=ahnu) 共 12 条
  [OK] 其中「数据库原理」(topic=database) 共 69 条
```

几点说明：
- **首次运行会下载 AI 模型**：`all-MiniLM-L6-v2` 模型约 80MB，首次运行时自动从 HuggingFace 下载，后续运行时直接使用本地缓存。
- **耐心等待向量化**：81 个语块的向量化通常在 10~30 秒内完成，取决于你的 CPU 性能。
- **幂等运行**：每次运行都会先清空旧的 `rag_knowledge_base` 索引再重新灌入，所以可以放心重复执行。
- **部分失败不影响整体**：如果某个网页因为网络原因爬取失败，脚本会打印 `✗ 爬取失败` 并继续处理下一个，不会中断整个流程。

---

## 终极功能：本地 PDF 教材批量解析入库

### 功能简介

系统现在不仅支持从互联网爬取网页构建知识库，还能直接读取本地的 PDF 教材（如《数据库系统概论》《操作系统概念》等经典课本）。脚本会将 PDF 按页加载，自动识别文件名并映射为分类标签（topic），经过语义切片后**增量追加**到现有的 Redis 向量数据库中。这意味着网页数据和教材数据可以共存于同一个知识库中，互不覆盖、互不干扰。

### 准备工作

1. 在项目根目录下创建 `JiaoCai/` 文件夹。
2. 将需要解析的文字版 PDF 教材放入该文件夹。目前支持的自动标签映射规则如下：

| 文件名包含 | 自动映射 topic | 示例 |
|-----------|--------------|------|
| `ShuJvKu` 或 `数据库` | `database` | `ShuJvKu.pdf` |
| `CaoZuoXiTong` 或 `操作系统` | `os` | `CaoZuoXiTong.pdf` |
| 其他 | `textbook` | `JiSuanJiWangLuo.pdf` |

> 注意：PDF 必须是文字版（非扫描图片版），否则 PyPDFLoader 无法提取文本内容。

### 依赖安装

PDF 解析功能需要额外安装一个轻量级库：

```bash
pip install pypdf
```

### 运行命令

确保 Redis Stack 容器正在运行，然后在项目目录下执行：

```bash
python build_pdf_knowledge.py
```

### 惊艳效果预告

脚本运行完毕后，**完全不需要修改任何前端代码**。只需刷新浏览器中的 Streamlit 页面，你会发现三个令人惊喜的变化：

1. **侧边栏自动生长**：左侧 Category 下拉菜单会自动出现新的分类标签（如 `os`、`textbook`），这些标签是通过 `FT.TAGVALS` 实时从 Redis 中拉取的，零硬编码。
2. **精准定位原文**：在搜索框中输入问题（如"什么是进程？"），系统会直接从教材中揪出最相关的原文段落，并在结果卡片中标注教材文件名和相似度得分。
3. **网页 + 教材混合检索**：之前爬取的维基百科数据库考点和现在导入的教材内容同处一个向量空间，搜索时系统会自动跨数据源找出最优答案。你可以通过 Category 下拉菜单自由切换"只看教材"还是"看全网数据"。

---

## 进阶功能：Web 可视化演示

### 什么是 Streamlit？

Streamlit 是一个开源的 Python Web 框架，**专门用于快速构建数据科学和机器学习的交互式 Web 应用**。你不需要写 HTML、CSS 或 JavaScript，只需用纯 Python 代码就能生成漂亮的 UI 界面。它非常适合用于课程答辩时做现场演示。

### 安装 Streamlit

```bash
pip install streamlit
```

### 启动 Web 页面

确保 Redis Stack 容器正在运行，然后在项目目录下执行：

```bash
streamlit run app.py
```

运行成功后，终端会显示类似以下的提示：

```
  You can now view your Streamlit app in your browser.

  Local URL:            http://localhost:8501
  Network URL:          http://192.168.x.x:8501
```

### 使用说明

1. 在浏览器中打开 **http://localhost:8501**
2. 左侧边栏仅保留两个核心控件：**Top K**（返回数量）和 **Category**（数据分类过滤）
3. 在搜索框中输入问题（如"什么是 ACID？"），按 Enter 即可看到检索结果
4. 每条结果以极简容器卡片呈现：顶部显示相关性得分与分类标签，正文为完整语块内容，底部附可点击的源文档链接

### 界面设计说明

本系统采用**极简企业级 UI 风格**（参考 Vercel / Stripe 设计语言）。页面以大面积留白和克制的黑白灰配色为主，无彩色装饰条、无 Emoji 图标。侧边栏仅保留必要的检索参数控件，主区域仅保留搜索框与结果卡片。每条检索结果卡片包含三项核心信息：`RELEVANCE` 相似度得分、正文内容、`View Source Document` 源网页链接。

系统 V2.6 版本引入了基于 NLP 正则的 OCR 文本断句清洗算法，彻底解决了扫描件提取文本的碎裂换行痛点；配合前端两端对齐（Justify）及首行缩进排版，提供了印刷级的交互式阅读体验。

### 团队协同与多端访问

系统现已原生采用 **cpolar** 作为高稳定性内网穿透方案。cpolar 拥有国内专属节点优化，完美兼容移动端（手机/平板）与 PC 端浏览器的 WebSocket 实时长连接，双端秒开，彻底告别断流与空白页问题。

#### 方案优势

- 国内节点加速，延迟极低
- 原生支持 WebSocket 长连接，Streamlit 无刷新断连风险
- 提供干净的公网 HTTPS 链接，评审老师点击即开

#### 部署步骤

**首次运行（仅需配置一次）：**

```bash
cpolar authtoken <您的TOKEN密钥>
```

**每次启动映射：**

```bash
cpolar http 8501
```

执行后终端会生成一个公网访问地址（形如 `https://xxxx.cpolar.io`），小组成员或评审老师直接在浏览器中打开该链接即可安全访问，无需粘贴公网 IP 或进行任何二次验证。

---

### 💡 核心高频报错排查

以下是 Windows 环境下组员克隆代码后最常遇到的启动报错及保姆级解决方案。

#### 问题 A：无法将"cpolar"项识别为 cmdlet 或可运行程序的名称

**核心原因：** 刚安装完 cpolar 软件后未重启当前终端，或系统 PATH 环境变量未及时刷新。

**解决方案：**

1. 彻底关闭当前的 PyCharm 终端、PowerShell 或 CMD 窗口，重新打开后再试。
2. 若依旧报错，可直接切入 cpolar 的默认物理安装路径下运行：
   - 在终端执行：`cd "C:\Program Files\cpolar"`
   - 随后执行：`.\cpolar http 8501`（PowerShell）或 `cpolar http 8501`（CMD）

---

#### 问题 B：无法将"streamlit"项识别为 cmdlet 或可运行程序的名称

**核心原因：** 重新打开的终端窗口没有激活 Python 虚拟环境（venv），系统无法全局索引 streamlit 命令。

**解决方案：**

1. 确保在项目根目录下先执行虚拟环境激活命令：
   - PowerShell：`.\venv\Scripts\activate`
   - CMD：`venv\Scripts\activate`
2. 若激活后依旧报"系统路径拒绝"类错误，可使用 Python 模块化点名方式强行绕过查找直接启动：
   ```bash
   python -m streamlit run app.py
   ```

### 停止服务

在运行 Streamlit 的终端中按 `Ctrl + C` 即可停止服务。```
