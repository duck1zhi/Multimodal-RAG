# -*- coding: utf-8 -*-
"""
迪士尼RAG助手 - 知识库构建（MinerU解析 + 索引入库）

与 4-disney_build_index.py 独立：不修改原脚本。
MinerU API 负责 PDF、PPT转PDF、价目表截图；其余文件沿用原解析方法。
解析产物先写入项目根目录 parsed_archive，再做 embedding。
"""
import os
import io
import re
import time
import json
import base64
import hashlib
import shutil
import zipfile
import requests
import numpy as np
import faiss
import dashscope
from http import HTTPStatus
from docx import Document as DocxDocument

# 配置
DASHSCOPE_API_KEY = os.getenv("DASHSCOPE_API_KEY")
if not DASHSCOPE_API_KEY:
    raise ValueError("错误：请设置 'DASHSCOPE_API_KEY' 环境变量。")

MINERU_TOKEN = os.getenv("MINERU_TOKEN")
if not MINERU_TOKEN:
    raise ValueError("错误：请设置 'MINERU_TOKEN' 环境变量（在 mineru.net API 管理页创建）。")

dashscope.api_key = DASHSCOPE_API_KEY

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DOCS_DIR = os.path.join(BASE_DIR, "disney_knowledge_base")
ARCHIVE_DIR = os.path.join(BASE_DIR, "parsed_archive")
MULTIMODAL_EMBEDDING_MODEL = "tongyi-embedding-vision-plus"

# 输出文件（与查询脚本共用）
INDEX_FILE = os.path.join(BASE_DIR, "disney_index.faiss")
METADATA_FILE = os.path.join(BASE_DIR, "disney_metadata.json")

# 切分参数
CHUNK_SIZE = 500  # 每个chunk的字符数
CHUNK_OVERLAP = 50  # chunk之间的重叠字符数

# MinerU 接口
MINERU_API_BASE = "https://mineru.net/api/v4"
MINERU_MODEL_VERSION = "vlm"
MINERU_POLL_INTERVAL = 5
MINERU_POLL_TIMEOUT = 1800
MINERU_IN_PROGRESS = {"waiting-file", "pending", "running", "converting"}

# 价目表截图文件名关键词
PRICE_KEYWORDS = ["价格", "价目", "票价", "价目表", "价格表"]
IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".gif", ".bmp", ".webp", ".tiff", ".jp2"}
PPT_EXTS = {".ppt", ".pptx"}

# 视频知识库
VIDEO_KNOWLEDGE = [
    {
        "url": "https://dataset-1255932437.cos.ap-nanjing.myqcloud.com/mp4/car.mp4",
        "description": "汽车剐蹭视频"
    }
]


def parse_docx(file_path):
    """解析 DOCX 文件，提取全部文本"""
    doc = DocxDocument(file_path)
    all_text = []

    for element in doc.element.body:
        if element.tag.endswith('p'):
            paragraph_text = ""
            for run in element.findall('.//w:t', {'w': 'http://schemas.openxmlformats.org/wordprocessingml/2006/main'}):
                paragraph_text += run.text if run.text else ""
            if paragraph_text.strip():
                all_text.append(paragraph_text.strip())

        elif element.tag.endswith('tbl'):
            table = [t for t in doc.tables if t._element is element][0]
            if table.rows:
                md_table = []
                header = [cell.text.strip() for cell in table.rows[0].cells]
                md_table.append("| " + " | ".join(header) + " |")
                md_table.append("|" + "---|" * len(header))
                for row in table.rows[1:]:
                    row_data = [cell.text.strip() for cell in row.cells]
                    md_table.append("| " + " | ".join(row_data) + " |")
                all_text.append("\n".join(md_table))

    return "\n".join(all_text)


def split_text(text, chunk_size=CHUNK_SIZE, overlap=CHUNK_OVERLAP):
    """按固定长度切分文本"""
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunk = text[start:end]
        if chunk.strip():
            chunks.append(chunk.strip())
        start = end - overlap
    return chunks


def get_text_embedding(text):
    """文本embedding"""
    resp = dashscope.MultiModalEmbedding.call(
        model=MULTIMODAL_EMBEDDING_MODEL,
        input=[{'text': text}]
    )
    if resp.status_code != HTTPStatus.OK:
        raise Exception(f"文本Embedding失败: {resp.message}")
    return resp.output['embeddings'][0]['embedding']


def get_image_embedding(image_path):
    """图片embedding"""
    with open(image_path, "rb") as f:
        base64_image = base64.b64encode(f.read()).decode('utf-8')

    ext = os.path.splitext(image_path)[1].lower().lstrip('.')
    if ext == 'jpg':
        ext = 'jpeg'
    image_data = f"data:image/{ext};base64,{base64_image}"

    resp = dashscope.MultiModalEmbedding.call(
        model=MULTIMODAL_EMBEDDING_MODEL,
        input=[{'image': image_data}]
    )
    if resp.status_code != HTTPStatus.OK:
        raise Exception(f"图片Embedding失败: {resp.message}")
    return resp.output['embeddings'][0]['embedding']


def get_video_embedding(video_url):
    """视频embedding（多帧取平均）"""
    resp = dashscope.MultiModalEmbedding.call(
        model=MULTIMODAL_EMBEDDING_MODEL,
        input=[{'video': video_url}]
    )
    if resp.status_code != HTTPStatus.OK:
        raise Exception(f"视频Embedding失败: {resp.message}")

    embeddings = resp.output['embeddings']
    if len(embeddings) > 1:
        vectors = [np.array(e['embedding']) for e in embeddings]
        return np.mean(vectors, axis=0).tolist()
    return embeddings[0]['embedding']


def sanitize_name(name):
    """去掉Windows路径非法字符，用作存档目录名"""
    cleaned = re.sub(r'[<>:"/\\|?*]', "_", name)
    cleaned = cleaned.strip(" .")
    return cleaned or "unnamed"


def make_archive_subdir(category, rel_path):
    """按相对路径生成存档子目录，避免重名覆盖"""
    rel_no_ext = os.path.splitext(rel_path)[0]
    flat = sanitize_name(rel_no_ext.replace("\\", "__").replace("/", "__"))
    return os.path.join(ARCHIVE_DIR, category, flat)


def is_price_image(filename):
    """文件名是否像价目表截图"""
    return any(keyword in filename for keyword in PRICE_KEYWORDS)


def mineru_headers():
    """MinerU API 鉴权头"""
    return {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {MINERU_TOKEN}"
    }


def convert_ppt_to_pdf(ppt_path, pdf_path):
    """用本机 PowerPoint 把 PPT/PPTX 转成 PDF"""
    import pythoncom
    import win32com.client

    ppt_path = os.path.abspath(ppt_path)
    pdf_path = os.path.abspath(pdf_path)
    os.makedirs(os.path.dirname(pdf_path), exist_ok=True)
    if os.path.exists(pdf_path):
        os.remove(pdf_path)

    pythoncom.CoInitialize()
    app = None
    presentation = None
    try:
        app = win32com.client.DispatchEx("PowerPoint.Application")
        app.Visible = 1
        app.DisplayAlerts = 0
        presentation = app.Presentations.Open(
            ppt_path,
            ReadOnly=True,
            Untitled=False,
            WithWindow=False
        )
        # 2 = ppFixedFormatTypePDF
        presentation.ExportAsFixedFormat(pdf_path, 2)
    finally:
        if presentation is not None:
            presentation.Close()
        if app is not None:
            app.Quit()
        pythoncom.CoUninitialize()

    if not os.path.exists(pdf_path):
        raise Exception(f"PPT转PDF失败，未生成文件: {ppt_path}")
    return pdf_path


def mineru_parse_local_file(file_path, archive_dir, data_id):
    """
    调用 MinerU 精准解析 API：申请上传链接 -> PUT 上传 -> 轮询 -> 下载ZIP到存档目录。
    返回 (markdown文本, 抽出的图片路径列表)
    """
    os.makedirs(archive_dir, exist_ok=True)
    ext = os.path.splitext(file_path)[1].lower()
    upload_name = f"{data_id}{ext}"

    apply_url = f"{MINERU_API_BASE}/file-urls/batch"
    payload = {
        "files": [
            {
                "name": upload_name,
                "data_id": data_id,
                "is_ocr": True
            }
        ],
        "model_version": MINERU_MODEL_VERSION,
        "enable_table": True,
        "enable_formula": True,
        "language": "ch"
    }
    apply_resp = requests.post(apply_url, headers=mineru_headers(), json=payload)
    if apply_resp.status_code != 200:
        raise Exception(f"MinerU申请上传链接失败 HTTP {apply_resp.status_code}: {apply_resp.text}")
    apply_result = apply_resp.json()
    if apply_result.get("code") != 0:
        raise Exception(f"MinerU申请上传链接失败: {apply_result.get('msg')}")

    batch_id = apply_result["data"]["batch_id"]
    file_urls = apply_result["data"]["file_urls"]
    print(f"    MinerU batch_id={batch_id}，开始上传")

    with open(file_path, "rb") as f:
        upload_resp = requests.put(file_urls[0], data=f)
    if upload_resp.status_code != 200:
        raise Exception(f"MinerU文件上传失败 HTTP {upload_resp.status_code}: {upload_resp.text}")
    print("    上传完成，等待解析")

    poll_url = f"{MINERU_API_BASE}/extract-results/batch/{batch_id}"
    auth_header = {"Authorization": f"Bearer {MINERU_TOKEN}"}
    started = time.time()
    zip_url = None
    while True:
        if time.time() - started > MINERU_POLL_TIMEOUT:
            raise Exception(f"MinerU解析超时: {os.path.basename(file_path)}")

        poll_resp = requests.get(poll_url, headers=auth_header)
        if poll_resp.status_code != 200:
            raise Exception(f"MinerU查询任务失败 HTTP {poll_resp.status_code}: {poll_resp.text}")
        poll_result = poll_resp.json()
        if poll_result.get("code") != 0:
            raise Exception(f"MinerU查询任务失败: {poll_result.get('msg')}")

        extract_result = poll_result["data"]["extract_result"]
        if isinstance(extract_result, dict):
            extract_result = [extract_result]
        item = extract_result[0]
        state = item.get("state", "")

        if state == "done":
            zip_url = item.get("full_zip_url")
            if not zip_url:
                raise Exception(f"MinerU解析完成但没有 full_zip_url: {item}")
            break
        if state == "failed":
            raise Exception(f"MinerU解析失败: {item.get('err_msg')}")
        if state not in MINERU_IN_PROGRESS:
            raise Exception(f"MinerU未知任务状态: {state}")

        progress = item.get("extract_progress") or {}
        extracted_pages = progress.get("extracted_pages")
        total_pages = progress.get("total_pages")
        if extracted_pages is not None and total_pages is not None:
            print(f"    解析中: {state} {extracted_pages}/{total_pages} 页")
        else:
            print(f"    解析中: {state}")
        time.sleep(MINERU_POLL_INTERVAL)

    print("    下载解析结果ZIP")
    zip_resp = requests.get(zip_url)
    if zip_resp.status_code != 200:
        raise Exception(f"MinerU结果包下载失败 HTTP {zip_resp.status_code}")

    zip_save_path = os.path.join(archive_dir, "mineru_result.zip")
    with open(zip_save_path, "wb") as f:
        f.write(zip_resp.content)

    extract_dir = os.path.join(archive_dir, "mineru_output")
    os.makedirs(extract_dir, exist_ok=True)
    with zipfile.ZipFile(io.BytesIO(zip_resp.content)) as zf:
        zf.extractall(extract_dir)

    markdown_path = None
    for root, _dirs, files in os.walk(extract_dir):
        if "full.md" in files:
            markdown_path = os.path.join(root, "full.md")
            break
    if markdown_path is None:
        md_files = []
        for root, _dirs, files in os.walk(extract_dir):
            for name in files:
                if name.lower().endswith(".md"):
                    md_files.append(os.path.join(root, name))
        if not md_files:
            raise Exception(f"MinerU结果中未找到 Markdown: {archive_dir}")
        markdown_path = md_files[0]

    with open(markdown_path, "r", encoding="utf-8") as f:
        markdown_text = f.read()

    image_paths = []
    for root, _dirs, files in os.walk(extract_dir):
        for name in files:
            if os.path.splitext(name)[1].lower() in IMAGE_EXTS:
                image_paths.append(os.path.join(root, name))
    image_paths.sort()

    print(f"    存档完成: md={len(markdown_text)} 字符, 图片={len(image_paths)} 张")
    return markdown_text, image_paths


def collect_source_files():
    """扫描知识库，按类型分流"""
    if not os.path.isdir(DOCS_DIR):
        raise ValueError(f"错误：找不到知识库目录 {DOCS_DIR}")

    pdf_files = []
    ppt_files = []
    price_images = []
    other_images = []
    docx_files = []

    for root, dirs, files in os.walk(DOCS_DIR):
        dirs[:] = [d for d in dirs if not d.startswith(".")]
        for filename in files:
            if filename.startswith("."):
                continue
            file_path = os.path.join(root, filename)
            rel_path = os.path.relpath(file_path, DOCS_DIR)
            ext = os.path.splitext(filename)[1].lower()

            if ext == ".pdf":
                pdf_files.append((file_path, rel_path))
            elif ext in PPT_EXTS:
                ppt_files.append((file_path, rel_path))
            elif ext in IMAGE_EXTS and is_price_image(filename):
                price_images.append((file_path, rel_path))
            elif ext in IMAGE_EXTS:
                other_images.append((file_path, rel_path))
            elif ext == ".docx":
                docx_files.append((file_path, rel_path))
            else:
                print(f"  跳过未纳入本脚本的文件: {rel_path}")

    return pdf_files, ppt_files, price_images, other_images, docx_files


def append_text_chunks(metadata_store, all_vectors, doc_id, source, text):
    """把文本切分后写入向量和元数据，返回新的 doc_id"""
    chunks = split_text(text)
    print(f"    文档长度: {len(text)} 字符, 切分为 {len(chunks)} 个chunk")
    for chunk in chunks:
        metadata = {
            "id": doc_id,
            "source": source,
            "type": "text",
            "content": chunk
        }
        vector = get_text_embedding(chunk)
        all_vectors.append(vector)
        metadata_store.append(metadata)
        doc_id += 1
    return doc_id


def append_image_record(metadata_store, all_vectors, doc_id, source_label, img_path):
    """按原脚本方式写入一条图片记录，返回新的 doc_id"""
    img_name = os.path.basename(img_path)
    print(f"    - {img_name}")
    metadata = {
        "id": doc_id,
        "source": source_label,
        "type": "image",
        "path": img_path,
        "content": f"[图片] {img_name}"
    }
    vector = get_image_embedding(img_path)
    all_vectors.append(vector)
    metadata_store.append(metadata)
    return doc_id + 1


def write_manifest(entries):
    """写入存档清单，方便核对解析结果"""
    manifest_path = os.path.join(ARCHIVE_DIR, "manifest.json")
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(entries, f, ensure_ascii=False, indent=2)
    print(f"存档清单: {manifest_path}")


def build_and_save():
    """先解析存档，再构建知识库并保存"""
    print("\n--- 构建多模态知识库（MinerU） ---")
    print(f"切分参数: chunk_size={CHUNK_SIZE}, overlap={CHUNK_OVERLAP}")
    print(f"解析存档目录: {ARCHIVE_DIR}")

    if os.path.isdir(ARCHIVE_DIR):
        shutil.rmtree(ARCHIVE_DIR)
    os.makedirs(ARCHIVE_DIR, exist_ok=True)

    pdf_files, ppt_files, price_images, other_images, docx_files = collect_source_files()
    print(
        f"扫描结果: PDF {len(pdf_files)}, PPT {len(ppt_files)}, "
        f"价目表截图 {len(price_images)}, 其他图片 {len(other_images)}, DOCX {len(docx_files)}"
    )

    metadata_store = []
    all_vectors = []
    doc_id = 0
    manifest = []

    # 处理 Word 文档（沿用原 parse_docx）
    for file_path, rel_path in docx_files:
        filename = os.path.basename(file_path)
        print(f"  处理文档: {rel_path}")
        full_text = parse_docx(file_path)
        archive_dir = make_archive_subdir("docx", rel_path)
        os.makedirs(archive_dir, exist_ok=True)
        md_path = os.path.join(archive_dir, "full.md")
        with open(md_path, "w", encoding="utf-8") as f:
            f.write(full_text)
        manifest.append({
            "category": "docx",
            "source": rel_path,
            "archive_dir": archive_dir,
            "markdown": md_path
        })
        doc_id = append_text_chunks(metadata_store, all_vectors, doc_id, filename, full_text)

    # 处理 PDF（MinerU）
    for file_path, rel_path in pdf_files:
        print(f"  处理PDF: {rel_path}")
        archive_dir = make_archive_subdir("pdf", rel_path)
        data_id = "id_" + hashlib.md5(rel_path.encode("utf-8")).hexdigest()[:16]
        markdown_text, image_paths = mineru_parse_local_file(file_path, archive_dir, data_id)
        manifest.append({
            "category": "pdf",
            "source": rel_path,
            "archive_dir": archive_dir,
            "images": image_paths
        })
        doc_id = append_text_chunks(
            metadata_store, all_vectors, doc_id, os.path.basename(file_path), markdown_text
        )
        for img_path in image_paths:
            doc_id = append_image_record(
                metadata_store, all_vectors, doc_id,
                f"图片: {os.path.basename(img_path)}",
                img_path
            )

    # 处理 PPT/PPTX：先转 PDF，再走 MinerU
    for file_path, rel_path in ppt_files:
        print(f"  处理PPT: {rel_path}")
        archive_dir = make_archive_subdir("ppt", rel_path)
        os.makedirs(archive_dir, exist_ok=True)
        converted_pdf = os.path.join(archive_dir, "converted.pdf")
        print("    转换为PDF...")
        convert_ppt_to_pdf(file_path, converted_pdf)
        data_id = "id_" + hashlib.md5(rel_path.encode("utf-8")).hexdigest()[:16]
        markdown_text, image_paths = mineru_parse_local_file(converted_pdf, archive_dir, data_id)
        manifest.append({
            "category": "ppt",
            "source": rel_path,
            "converted_pdf": converted_pdf,
            "archive_dir": archive_dir,
            "images": image_paths
        })
        doc_id = append_text_chunks(
            metadata_store, all_vectors, doc_id, os.path.basename(file_path), markdown_text
        )
        for img_path in image_paths:
            doc_id = append_image_record(
                metadata_store, all_vectors, doc_id,
                f"图片: {os.path.basename(img_path)}",
                img_path
            )

    # 处理价目表截图（MinerU OCR）
    for file_path, rel_path in price_images:
        print(f"  处理价目表截图: {rel_path}")
        archive_dir = make_archive_subdir("price_images", rel_path)
        os.makedirs(archive_dir, exist_ok=True)
        source_copy = os.path.join(archive_dir, "original" + os.path.splitext(file_path)[1].lower())
        shutil.copy2(file_path, source_copy)
        data_id = "id_" + hashlib.md5(rel_path.encode("utf-8")).hexdigest()[:16]
        markdown_text, image_paths = mineru_parse_local_file(file_path, archive_dir, data_id)
        manifest.append({
            "category": "price_images",
            "source": rel_path,
            "original": source_copy,
            "archive_dir": archive_dir,
            "images": image_paths
        })
        doc_id = append_text_chunks(
            metadata_store, all_vectors, doc_id, os.path.basename(file_path), markdown_text
        )

    # 处理其他图片（沿用原整图 embedding）
    print("  处理图片...")
    for file_path, rel_path in other_images:
        archive_dir = make_archive_subdir("images", rel_path)
        os.makedirs(archive_dir, exist_ok=True)
        copied = os.path.join(archive_dir, os.path.basename(file_path))
        shutil.copy2(file_path, copied)
        manifest.append({
            "category": "images",
            "source": rel_path,
            "archive_dir": archive_dir,
            "image": copied
        })
        doc_id = append_image_record(
            metadata_store, all_vectors, doc_id,
            f"图片: {os.path.basename(file_path)}",
            copied
        )

    # 处理视频
    print("  处理视频...")
    for video_info in VIDEO_KNOWLEDGE:
        print(f"    - {video_info['description']}")
        metadata = {
            "id": doc_id,
            "source": f"视频: {video_info['description']}",
            "type": "video",
            "url": video_info["url"],
            "description": video_info["description"],
            "content": f"[视频] {video_info['description']}"
        }
        vector = get_video_embedding(video_info["url"])
        all_vectors.append(vector)
        metadata_store.append(metadata)
        doc_id += 1
        manifest.append({
            "category": "video",
            "source": video_info["description"],
            "url": video_info["url"]
        })

    write_manifest(manifest)

    # 创建FAISS索引
    if all_vectors:
        dim = len(all_vectors[0])
        print(f"\n向量维度: {dim}")

        index = faiss.IndexFlatL2(dim)
        index.add(np.array(all_vectors).astype('float32'))

        faiss.write_index(index, INDEX_FILE)
        print(f"索引已保存: {INDEX_FILE}")

        with open(METADATA_FILE, 'w', encoding='utf-8') as f:
            json.dump(metadata_store, f, ensure_ascii=False, indent=2)
        print(f"元数据已保存: {METADATA_FILE}")

    text_count = sum(1 for m in metadata_store if m["type"] == "text")
    image_count = sum(1 for m in metadata_store if m["type"] == "image")
    video_count = sum(1 for m in metadata_store if m["type"] == "video")
    print(f"\n完成! 文本:{text_count}, 图片:{image_count}, 视频:{video_count}")

    print("\n--- 知识库内容 ---")
    for m in metadata_store:
        print(f"\n[{m['id']:2d}] [{m['type']:5s}] {m.get('source', '')}")
        print(f"    {m['content']}")


if __name__ == "__main__":
    build_and_save()
