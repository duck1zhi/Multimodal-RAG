My project-Multimodal RAG
================================

本项目把「迪士尼RAG知识库」目录中的文档、图片解析成文本和图片向量，写入 FAISS 索引，再用自然语言提问检索并生成客服式回答。

适用环境：Windows + Python 3.12。解析依赖本机 MinerU 4 API（默认 http://127.0.0.1:8000），向量和问答依赖阿里云 DashScope。


一、下载后目录里有什么
------------------------

迪士尼RAG知识库.py     构建脚本：解析 -> 存档 -> embedding -> 索引
迪士尼RAG查询.py       查询脚本：加载索引 -> 检索 -> 大模型回答
迪士尼RAG知识库\       源文件目录，把 PDF/Word/PPT/图片放这里
parsed_archive\        解析产物存档（运行构建脚本后生成）
parsed_archive\parse_log.txt      本次构建的解析日志
parsed_archive\failed_files.txt   失败/跳过的文件列表
parsed_archive\manifest.json      存档清单
disney_index.faiss     向量索引（构建完成后生成，查询脚本读取）
disney_metadata.json   文本/图片/视频元数据（构建完成后生成）
requirements.txt       Python 依赖
readme.txt             本说明

视频条目写在 迪士尼RAG知识库.py 顶部的 VIDEO_KNOWLEDGE 里，不是文件目录。


二、两个脚本在做什么
--------------------

【迪士尼RAG知识库.py】构建知识库

1. 扫描「迪士尼RAG知识库」文件夹。
2. 所有 PDF、DOC、DOCX、PPT、PPTX、图片都交给本机 MinerU 4 解析：
   - PDF、图片：standard 档（高质量 OCR/版面）
   - DOC/DOCX/PPT/PPTX：MinerU 原生 Office 解析（flash）
3. 失败回退（原生解析失败时）：
   - DOC -> 本机 Word 转 DOCX，再送 MinerU
   - PPT/PPTX -> 本机 PowerPoint 转 PDF，再送 MinerU
   - 某个文件回退仍失败：写入失败列表，继续处理后面的文件，不会整次中断
   - MinerU 队列卡住：不回退（转完还是同一套队列），并跳过后面尚未解析的文件
   - 回退需要本机安装 Word / PowerPoint；pywin32 已在 requirements.txt 中，Office 本身不是 pip 包
4. 解析结果写入 parsed_archive（markdown、抽出的图片、zip）。
   某文件目录下已有 mineru_output 时会复用，不再重复解析。
   每次构建还会写出：
   - parsed_archive/parse_log.txt    解析过程日志
   - parsed_archive/failed_files.txt 失败或跳过的文件
5. 文本按 500 字切分、重叠 50 字；文本和图片调用 DashScope
   模型 tongyi-embedding-vision-plus 做向量。
   图片向量会缓存在对应图片旁的 .npy，下次可复用。
6. 视频按 VIDEO_KNOWLEDGE 里的 url + description 做视频 embedding。
7. 写出 disney_index.faiss 和 disney_metadata.json。

【迪士尼RAG查询.py】提问查询

1. 读取上面两份索引文件。
2. 把用户问题做成文本向量，在 FAISS 里按 L2 距离检索。
3. 取最相近的 Top-3 文本交给 qwen-flash 生成回答。
4. 问题里若出现「图片/海报/看看/长什么样」等，会尝试附加一张相近图片路径。
5. 问题里若出现「视频/播放」等，会尝试附加一条相近视频 URL。
6. 脚本末尾带了 7 道示例题（退款、小朋友项目、年票、客诉、邮轮价目表、园区地图、避坑视频）。


三、下载后怎么用
----------------

第 1 步 安装 Python 依赖

  打开 PowerShell，进入本项目根目录，执行：

    pip install -r requirements.txt

第 2 步 配置阿里云密钥

  需要环境变量 DASHSCOPE_API_KEY（通义 embedding + qwen-flash 都用它）。
  Windows 可在「系统属性 -> 环境变量」里新增用户变量，或在当前 PowerShell 临时设置：

    $env:DASHSCOPE_API_KEY = "你的密钥"

  新开终端后若用「系统环境变量」，一般会自动生效；临时设置只对当前窗口有效。

第 3 步 准备源文件

  把要入库的资料放到「迪士尼RAG知识库」目录（可分子文件夹）。
  支持：.pdf .doc .docx .ppt .pptx 以及常见图片（png/jpg/jpeg 等）。
  文件名含「价格/价目/票价」的图片会按价目表走 MinerU OCR。

第 4 步 启动本机 MinerU 4（构建前必须开着）

  MinerU 不包含在本仓库的 pip 依赖里，需本机单独部署 API。
  默认地址：http://127.0.0.1:8000
  本机若按本环境安装，可在 PowerShell 执行：

    H:\MinerU\start-api.ps1

  若提示无法运行脚本，先执行：

    Set-ExecutionPolicy -Scope Process Bypass

  再重新运行 start-api.ps1。
  等到出现：Uvicorn running on http://127.0.0.1:8000
  这个窗口不要关。浏览器打开根地址出现 404 是正常的，接口在 /v1/health 和 /docs。
  若 API 不在 8000 端口，构建前设置：

    $env:MINERU_API_BASE = "http://127.0.0.1:你的端口"

第 5 步 构建索引

  另开一个 PowerShell，进入项目根目录：

    python 迪士尼RAG知识库.py

  首次会较慢（解析 + embedding）。看到「完成! 文本:xx, 图片:xx, 视频:xx」即成功。
  构建结束后，到 MinerU 那个窗口按 Ctrl+C 停止服务，可释放显存。

第 6 步 提问查询（不需要 MinerU）

  确认根目录已有 disney_index.faiss 和 disney_metadata.json，然后：

    python 迪士尼RAG查询.py

  会依次跑脚本里的示例问题。要改问题，编辑 迪士尼RAG查询.py 最底部的 rag_ask(...) 即可。
  要改视频，编辑 迪士尼RAG知识库.py 里的 VIDEO_KNOWLEDGE，再重新构建。


四、使用注意
------------

- 先构建，再查询。没有索引文件时查询脚本会直接报错。
- 构建期间 MinerU 必须在线。查询只走 DashScope，不必开 MinerU。
- 解析会复用 parsed_archive 里的 MinerU 产物；embedding 每次仍会重建整份 FAISS 索引。
  图片若已有 .npy 缓存则不再调图片向量接口。
- 旧版仅 python-docx 生成的 Word 存档（没有 mineru_output）会重新交给 MinerU 解析。
- 视频链接若带过期签名，失效后视频 embedding 会失败，需换成可访问地址后重建。
- 本机构建脚本主流程是 MinerU 统一解析，没有 Office 也能跑原生解析。
  只有 DOC/PPT 原生失败后的回退才需要本机 Word / PowerPoint。
  某个文件回退失败不会拦住后面的文件；只有 MinerU 队列卡住才会跳过后续解析。
  结果见 parsed_archive/parse_log.txt 和 parsed_archive/failed_files.txt。


五、常见问题
------------

1) 连接 127.0.0.1:8000 被拒绝
   MinerU 没启动，或启动窗口已关闭。先完成第 4 步。

2) 提示未设置 DASHSCOPE_API_KEY
   当前终端读不到该环境变量。设置后重新打开终端再跑。

3) 任务栏看不到 PowerShell
   若 MinerU 是在 Cursor 终端后台开的，不会出现独立窗口。到 Cursor 底部终端里找到
   含 Uvicorn running 的那个，按 Ctrl+C 停止。

4) 构建很慢或卡在「切分为 N 个 chunk」
   正在调用 DashScope 做文本向量，不是死机。每个 chunk 一次网络请求。
