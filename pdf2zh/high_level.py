"""Functions that can be used for the most common use-cases for pdf2zh.six"""

import asyncio
import io
import os
import re
import sys
import tempfile
import logging
import time
from asyncio import CancelledError
from pathlib import Path
from string import Template
from typing import Any, BinaryIO, List, Optional, Dict

import numpy as np
import requests
import tqdm
from pdfminer.pdfdocument import PDFDocument
from pdfminer.pdfexceptions import PDFValueError
from pdfminer.pdfinterp import PDFResourceManager
from pdfminer.pdfpage import PDFPage
from pdfminer.pdfparser import PDFParser
from pymupdf import Document, Font

from pdf2zh.converter import TranslateConverter
from pdf2zh.doclayout import OnnxModel
from pdf2zh.pdfinterp import PDFPageInterpreterEx

from pdf2zh.config import ConfigManager
from babeldoc.assets.assets import get_font_and_metadata

from pdf2zh.converter import AnalysisConverter

NOTO_NAME = "noto"

logger = logging.getLogger(__name__)

noto_list = [
    "am",  # Amharic
    "ar",  # Arabic
    "bn",  # Bengali
    "bg",  # Bulgarian
    "chr",  # Cherokee
    "el",  # Greek
    "gu",  # Gujarati
    "iw",  # Hebrew
    "hi",  # Hindi
    "kn",  # Kannada
    "ml",  # Malayalam
    "mr",  # Marathi
    "ru",  # Russian
    "sr",  # Serbian
    "ta",  # Tamil
    "te",  # Telugu
    "th",  # Thai
    "ur",  # Urdu
    "uk",  # Ukrainian
]


def analyze_pdf(
    pdf_bytes: bytes,
    model: OnnxModel,
    pages: Optional[list[int]] = None,
    cancellation_event: asyncio.Event = None,
) -> dict:
    """
    Analyzes a PDF file to gather statistics about its content.

    Args:
        pdf_bytes: The PDF content as bytes.
        model: The ONNX layout analysis model.
        pages: A list of page numbers to analyze. If None, all pages are analyzed.
        cancellation_event: An event to signal cancellation.

    Returns:
        A dictionary containing the analysis results.
    """
    # Create two independent streams from the same bytes to avoid conflicts
    inf_for_pdfminer = io.BytesIO(pdf_bytes)
    inf_for_pymupdf = io.BytesIO(pdf_bytes)

    rsrcmgr = PDFResourceManager()
    layout = {}
    device = AnalysisConverter(rsrcmgr, layout, pages)
    interpreter = PDFPageInterpreterEx(rsrcmgr, device, {})

    parser = PDFParser(inf_for_pdfminer)
    doc = PDFDocument(parser)
    doc_for_render = Document(stream=inf_for_pymupdf)

    pages_to_process = pages
    if pages_to_process is None:
        pages_to_process = list(range(doc_for_render.page_count))

    with tqdm.tqdm(total=len(pages_to_process), desc="Analyzing PDF") as progress:
        for pageno, page in enumerate(PDFPage.create_pages(doc)):
            page.pageno = pageno

            if pageno not in pages_to_process:
                continue

            if cancellation_event and cancellation_event.is_set():
                raise CancelledError("Analysis task cancelled")
            
            # Following the robust pattern from translate_patch:
            # The 'page_xref' attribute is also required by the interpreter.
            page.page_xref = doc_for_render.get_new_xref()
            
            # --- Layout analysis (from translate_patch) ---
            pix = doc_for_render[pageno].get_pixmap()
            image = np.frombuffer(pix.samples, np.uint8).reshape(pix.height, pix.width, 3)[:, :, ::-1]
            page_layout = model.predict(image, imgsz=int(pix.height / 32) * 32)[0]
            box = np.ones((pix.height, pix.width))
            h, w = box.shape
            vcls = ["abandon", "figure", "isolate_formula", "formula_caption"]
            table_regions = []
            
            for i, d in enumerate(page_layout.boxes):
                box_class = page_layout.names[int(d.cls)]
                x0, y0, x1, y1 = d.xyxy.squeeze()
                b_x0, b_y0, b_x1, b_y1 = (
                    np.clip(int(x0 - 1), 0, w - 1),
                    np.clip(int(h - y1 - 1), 0, h - 1),
                    np.clip(int(x1 + 1), 0, w - 1),
                    np.clip(int(h - y0 + 1), 0, h - 1),
                )
                
                if box_class == "table":
                    table_id = -(i + 100)
                    box[b_y0:b_y1, b_x0:b_x1] = table_id
                    table_regions.append({'id': table_id, 'bbox': (b_x0, b_y0, b_x1, b_y1)})
                elif box_class not in vcls:
                    box[b_y0:b_y1, b_x0:b_x1] = i + 2

            if 'table_regions' not in layout:
                layout['table_regions'] = {}
            layout['table_regions'][pageno] = table_regions
            
            for i, d in enumerate(page_layout.boxes):
                if page_layout.names[int(d.cls)] in vcls:
                    x0, y0, x1, y1 = d.xyxy.squeeze()
                    x0, y0, x1, y1 = (
                        np.clip(int(x0 - 1), 0, w - 1),
                        np.clip(int(h - y1 - 1), 0, h - 1),
                        np.clip(int(x1 + 1), 0, w - 1),
                        np.clip(int(h - y0 + 1), 0, h - 1),
                    )
                    box[y0:y1, x0:x1] = 0
            
            layout[pageno] = box
            # --- End Layout analysis ---

            interpreter.process_page(page)
            progress.update()

    device.close()
    return device.stats


def check_files(files: List[str]) -> List[str]:
    files = [
        f for f in files if not f.startswith("http://")
    ]  # exclude online files, http
    files = [
        f for f in files if not f.startswith("https://")
    ]  # exclude online files, https
    missing_files = [file for file in files if not os.path.exists(file)]
    return missing_files


def translate_patch(
    inf: BinaryIO,
    pages: Optional[list[int]] = None,
    vfont: str = "",
    vchar: str = "",
    thread: int = 0,
    doc_zh: Document = None,
    lang_in: str = "",
    lang_out: str = "",
    service: str = "",
    noto_name: str = "",
    noto: Font = None,
    callback: object = None,
    cancellation_event: asyncio.Event = None,
    model: OnnxModel = None,
    envs: Dict = None,
    prompt: Template = None,
    ignore_cache: bool = False,
    **kwarg: Any,
) -> dict:
    rsrcmgr = PDFResourceManager()
    layout = {}
    device = TranslateConverter(
        rsrcmgr,
        vfont,
        vchar,
        thread,
        layout,
        lang_in,
        lang_out,
        service,
        noto_name,
        noto,
        envs,
        prompt,
        ignore_cache,
    )

    assert device is not None
    obj_patch = {}
    interpreter = PDFPageInterpreterEx(rsrcmgr, device, obj_patch)
    if pages:
        total_pages = len(pages)
    else:
        total_pages = doc_zh.page_count

    parser = PDFParser(inf)
    doc = PDFDocument(parser)
    with tqdm.tqdm(total=total_pages) as progress:
        for pageno, page in enumerate(PDFPage.create_pages(doc)):
            if cancellation_event and cancellation_event.is_set():
                raise CancelledError("task cancelled")
            if pages and (pageno not in pages):
                continue
            progress.update()
            if callback:
                callback(progress)
            page.pageno = pageno
            pix = doc_zh[page.pageno].get_pixmap()
            image = np.frombuffer(pix.samples, np.uint8).reshape(
                pix.height, pix.width, 3
            )[:, :, ::-1]
            page_layout = model.predict(image, imgsz=int(pix.height / 32) * 32)[0]
            # kdtree 是不可能 kdtree 的，不如直接渲染成图片，用空间换时间
            box = np.ones((pix.height, pix.width))
            h, w = box.shape
            vcls = ["abandon", "figure", "isolate_formula", "formula_caption"]
            table_regions = []  # 存储表格区域信息
            table_count = 0  # 表格计数
            for i, d in enumerate(page_layout.boxes):
                if page_layout.names[int(d.cls)] not in vcls and page_layout.names[int(d.cls)] != "table":
                    x0, y0, x1, y1 = d.xyxy.squeeze()
                    x0, y0, x1, y1 = (
                        np.clip(int(x0 - 1), 0, w - 1),
                        np.clip(int(h - y1 - 1), 0, h - 1),
                        np.clip(int(x1 + 1), 0, w - 1),
                        np.clip(int(h - y0 + 1), 0, h - 1),
                    )
                    box[y0:y1, x0:x1] = i + 2
                elif page_layout.names[int(d.cls)] == "table":
                    # 表格区域使用负数ID标识，便于后续特殊处理
                    table_count += 1
                    x0, y0, x1, y1 = d.xyxy.squeeze()
                    logger.info(f"[版面分析] 第 {page.pageno+1} 页发现表格 {table_count}: 位置({x0:.1f}, {y0:.1f}, {x1:.1f}, {y1:.1f})")
                    x0, y0, x1, y1 = (
                        np.clip(int(x0 - 1), 0, w - 1),
                        np.clip(int(h - y1 - 1), 0, h - 1),
                        np.clip(int(x1 + 1), 0, w - 1),
                        np.clip(int(h - y0 + 1), 0, h - 1),
                    )
                    table_id = -(i + 100)  # 使用负数ID标识表格区域
                    box[y0:y1, x0:x1] = table_id
                    # 保存表格区域信息供后续处理使用
                    table_regions.append({
                        'id': table_id,
                        'bbox': (x0, y0, x1, y1),
                        'original_bbox': d.xyxy.squeeze()
                    })
                    logger.debug(f"[版面分析] 表格 {table_count} 标记为ID: {table_id}")
            
            # 存储表格区域信息到layout中
            if 'table_regions' not in layout:
                layout['table_regions'] = {}
            layout['table_regions'][page.pageno] = table_regions
            
            if table_count > 0:
                logger.info(f"[版面分析] 第 {page.pageno+1} 页共发现 {table_count} 个表格区域")
            
            for i, d in enumerate(page_layout.boxes):
                if page_layout.names[int(d.cls)] in vcls:
                    x0, y0, x1, y1 = d.xyxy.squeeze()
                    x0, y0, x1, y1 = (
                        np.clip(int(x0 - 1), 0, w - 1),
                        np.clip(int(h - y1 - 1), 0, h - 1),
                        np.clip(int(x1 + 1), 0, w - 1),
                        np.clip(int(h - y0 + 1), 0, h - 1),
                    )
                    box[y0:y1, x0:x1] = 0
            layout[page.pageno] = box
            # 新建一个 xref 存放新指令流
            page.page_xref = doc_zh.get_new_xref()  # hack 插入页面的新 xref
            doc_zh.update_object(page.page_xref, "<<>>")
            doc_zh.update_stream(page.page_xref, b"")
            doc_zh[page.pageno].set_contents(page.page_xref)
            interpreter.process_page(page)

    device.close()
    
    # 收集token统计信息
    token_stats = {}
    if hasattr(device, 'translator') and device.translator:
        token_stats = device.translator.get_token_stats()
        logger.debug(f"[翻译完成] Token统计: {token_stats}")
    
    # 收集段落统计信息
    paragraph_stats = {}
    if hasattr(device, 'get_paragraph_stats'):
        paragraph_stats = device.get_paragraph_stats()
        logger.debug(f"[翻译完成] 段落统计: {paragraph_stats}")
    
    # 收集表格统计信息
    table_stats = {}
    if hasattr(device, 'get_table_stats'):
        table_stats = device.get_table_stats()
        logger.debug(f"[翻译完成] 表格统计: {table_stats}")
    
    return obj_patch, token_stats, paragraph_stats, table_stats


def translate_stream(
    stream: bytes,
    pages: Optional[list[int]] = None,
    lang_in: str = "",
    lang_out: str = "",
    service: str = "",
    thread: int = 0,
    vfont: str = "",
    vchar: str = "",
    callback: object = None,
    cancellation_event: asyncio.Event = None,
    model: OnnxModel = None,
    envs: Dict = None,
    prompt: Template = None,
    skip_subset_fonts: bool = False,
    ignore_cache: bool = False,
    **kwarg: Any,
):
    font_list = [("tiro", None)]

    font_path = download_remote_fonts(lang_out.lower())
    noto_name = NOTO_NAME
    noto = Font(noto_name, font_path)
    font_list.append((noto_name, font_path))

    doc_en = Document(stream=stream)
    stream = io.BytesIO()
    doc_en.save(stream)
    doc_zh = Document(stream=stream)
    page_count = doc_zh.page_count
    # font_list = [("GoNotoKurrent-Regular.ttf", font_path), ("tiro", None)]
    font_id = {}
    for page in doc_zh:
        for font in font_list:
            font_id[font[0]] = page.insert_font(font[0], font[1])
    xreflen = doc_zh.xref_length()
    for xref in range(1, xreflen):
        for label in ["Resources/", ""]:  # 可能是基于 xobj 的 res
            try:  # xref 读写可能出错
                font_res = doc_zh.xref_get_key(xref, f"{label}Font")
                target_key_prefix = f"{label}Font/"
                if font_res[0] == "xref":
                    resource_xref_id = re.search("(\\d+) 0 R", font_res[1]).group(1)
                    xref = int(resource_xref_id)
                    font_res = ("dict", doc_zh.xref_object(xref))
                    target_key_prefix = ""

                if font_res[0] == "dict":
                    for font in font_list:
                        target_key = f"{target_key_prefix}{font[0]}"
                        font_exist = doc_zh.xref_get_key(xref, target_key)
                        if font_exist[0] == "null":
                            doc_zh.xref_set_key(
                                xref,
                                target_key,
                                f"{font_id[font[0]]} 0 R",
                            )
            except Exception:
                pass

    fp = io.BytesIO()

    doc_zh.save(fp)
    obj_patch, token_stats, paragraph_stats, table_stats = translate_patch(fp, **locals())

    for obj_id, ops_new in obj_patch.items():
        # ops_old=doc_en.xref_stream(obj_id)
        # print(obj_id)
        # print(ops_old)
        # print(ops_new.encode())
        doc_zh.update_stream(obj_id, ops_new.encode())

    doc_en.insert_file(doc_zh)
    for id in range(page_count):
        doc_en.move_page(page_count + id, id * 2 + 1)
    if not skip_subset_fonts:
        doc_zh.subset_fonts(fallback=True)
        doc_en.subset_fonts(fallback=True)
    return (
        doc_zh.write(deflate=True, garbage=3, use_objstms=1),
        doc_en.write(deflate=True, garbage=3, use_objstms=1),
        token_stats,
        paragraph_stats,
        table_stats,
    )


def convert_to_pdfa(input_path, output_path):
    """
    Convert PDF to PDF/A format

    Args:
        input_path: Path to source PDF file
        output_path: Path to save PDF/A file
    """
    from pikepdf import Dictionary, Name, Pdf

    # Open the PDF file
    pdf = Pdf.open(input_path)

    # Add PDF/A conformance metadata
    metadata = {
        "pdfa_part": "2",
        "pdfa_conformance": "B",
        "title": pdf.docinfo.get("/Title", ""),
        "author": pdf.docinfo.get("/Author", ""),
        "creator": "PDF Math Translate",
    }

    with pdf.open_metadata() as meta:
        meta.load_from_docinfo(pdf.docinfo)
        meta["pdfaid:part"] = metadata["pdfa_part"]
        meta["pdfaid:conformance"] = metadata["pdfa_conformance"]

    # Create OutputIntent dictionary
    output_intent = Dictionary(
        {
            "/Type": Name("/OutputIntent"),
            "/S": Name("/GTS_PDFA1"),
            "/OutputConditionIdentifier": "sRGB IEC61966-2.1",
            "/RegistryName": "http://www.color.org",
            "/Info": "sRGB IEC61966-2.1",
        }
    )

    # Add output intent to PDF root
    if "/OutputIntents" not in pdf.Root:
        pdf.Root.OutputIntents = [output_intent]
    else:
        pdf.Root.OutputIntents.append(output_intent)

    # Save as PDF/A
    pdf.save(output_path, linearize=True)
    pdf.close()


def translate(
    files: list[str],
    output: str = "",
    pages: Optional[list[int]] = None,
    lang_in: str = "",
    lang_out: str = "",
    service: str = "",
    thread: int = 0,
    vfont: str = "",
    vchar: str = "",
    callback: object = None,
    compatible: bool = False,
    cancellation_event: asyncio.Event = None,
    model: OnnxModel = None,
    envs: Dict = None,
    prompt: Template = None,
    skip_subset_fonts: bool = False,
    ignore_cache: bool = False,
    stats_obj: Optional[Any] = None,  # 添加统计对象参数
    **kwarg: Any,
):
    if not files:
        raise PDFValueError("No files to process.")

    missing_files = check_files(files)

    if missing_files:
        print("The following files do not exist:", file=sys.stderr)
        for file in missing_files:
            print(f"  {file}", file=sys.stderr)
        raise PDFValueError("Some files do not exist.")

    # 初始化统计信息
    total_token_stats = {
        "prompt_tokens": 0,
        "completion_tokens": 0,
        "total_tokens": 0,
        "translation_count": 0,
    }
    
    # 添加总段落统计
    total_paragraph_stats = {
        "total_paragraphs": 0,
        "skipped_empty": 0,
        "skipped_formula": 0,
        "skipped_no_text": 0,
        "translated": 0,
    }
    
    # 添加总表格统计
    total_table_stats = {
        "total_cells": 0,
        "skipped_empty": 0,
        "skipped_no_text": 0,
        "translated": 0,
    }

    # 如果传入了统计对象，则开始跟踪翻译时间
    if stats_obj:
        stats_obj.start_translation_tracking()

    result_files = []
    
    for file in files:
        if type(file) is str and (
            file.startswith("http://") or file.startswith("https://")
        ):
            print("Online files detected, downloading...")
            try:
                r = requests.get(file, allow_redirects=True)
                if r.status_code == 200:
                    with tempfile.NamedTemporaryFile(
                        suffix=".pdf", delete=False
                    ) as tmp_file:
                        print(f"Writing the file: {file}...")
                        tmp_file.write(r.content)
                        file = tmp_file.name
                else:
                    r.raise_for_status()
            except Exception as e:
                raise PDFValueError(
                    f"Errors occur in downloading the PDF file. Please check the link(s).\nError:\n{e}"
                )
        filename = os.path.splitext(os.path.basename(file))[0]

        # If the commandline has specified converting to PDF/A format
        # --compatible / -cp
        if compatible:
            with tempfile.NamedTemporaryFile(
                suffix="-pdfa.pdf", delete=False
            ) as tmp_pdfa:
                print(f"Converting {file} to PDF/A format...")
                convert_to_pdfa(file, tmp_pdfa.name)
                doc_raw = open(tmp_pdfa.name, "rb")
                os.unlink(tmp_pdfa.name)
        else:
            doc_raw = open(file, "rb")
        s_raw = doc_raw.read()
        doc_raw.close()

        temp_dir = Path(tempfile.gettempdir())
        file_path = Path(file)
        try:
            if file_path.exists() and file_path.resolve().is_relative_to(
                temp_dir.resolve()
            ):
                file_path.unlink(missing_ok=True)
                logger.debug(f"Cleaned temp file: {file_path}")
        except Exception as e:
            logger.warning(f"Failed to clean temp file {file_path}", exc_info=True)

        result = translate_stream(
            s_raw,
            **locals(),
        )
        s_mono, s_dual, token_stats, paragraph_stats, table_stats = result
        
        # 累计token统计信息
        for key in total_token_stats:
            total_token_stats[key] += token_stats.get(key, 0)
            
        # 累计段落统计信息
        for key in total_paragraph_stats:
            total_paragraph_stats[key] += paragraph_stats.get(key, 0)
            
        # 累计表格统计信息
        for key in total_table_stats:
            total_table_stats[key] += table_stats.get(key, 0)
        
        file_mono = Path(output) / f"{filename}-mono.pdf"
        file_dual = Path(output) / f"{filename}-dual.pdf"
        doc_mono = open(file_mono, "wb")
        doc_dual = open(file_dual, "wb")
        doc_mono.write(s_mono)
        doc_dual.write(s_dual)
        doc_mono.close()
        doc_dual.close()
        result_files.append((str(file_mono), str(file_dual)))

    # 结束翻译时间跟踪
    if stats_obj:
        stats_obj.end_translation_tracking()

    # 如果没有统计对象，则输出原有的统计报告
    if not stats_obj:
        # 输出token统计报告
        if total_token_stats["translation_count"] > 0:
            logger.info("=" * 20 + " Translation Token Report " + "=" * 20)
            logger.info(f"总翻译调用次数: {total_token_stats['translation_count']}")
            logger.info(f"总输入Token数: {total_token_stats['prompt_tokens']}")
            logger.info(f"总输出Token数: {total_token_stats['completion_tokens']}")
            logger.info(f"总Token使用量: {total_token_stats['total_tokens']}")
            if total_token_stats['translation_count'] > 0:
                logger.info(f"平均每次翻译Token数: {total_token_stats['total_tokens'] / total_token_stats['translation_count']:.1f}")
            logger.info("=" * 67)
        
        # 输出段落统计报告
        if total_paragraph_stats["total_paragraphs"] > 0:
            logger.info("=" * 20 + " Paragraph Statistics Report " + "=" * 20)
            logger.info(f"总段落数: {total_paragraph_stats['total_paragraphs']}")
            logger.info(f"已翻译段落: {total_paragraph_stats['translated']}")
            logger.info(f"跳过的段落:")
            logger.info(f"  - 空白段落: {total_paragraph_stats['skipped_empty']}")
            logger.info(f"  - 公式段落: {total_paragraph_stats['skipped_formula']}")
            logger.info(f"  - 无中英文段落: {total_paragraph_stats['skipped_no_text']}")
            logger.info("=" * 67)
            
        # 输出表格统计报告
        if total_table_stats["total_cells"] > 0:
            logger.info("=" * 20 + " Table Statistics Report " + "=" * 20)
            logger.info(f"总单元格数: {total_table_stats['total_cells']}")
            logger.info(f"已翻译单元格: {total_table_stats['translated']}")
            logger.info(f"跳过的单元格:")
            logger.info(f"  - 空白单元格: {total_table_stats['skipped_empty']}")
            logger.info(f"  - 无中英文单元格: {total_table_stats['skipped_no_text']}")
            logger.info("=" * 67)
    
    # 根据是否有统计对象返回不同格式
    if stats_obj:
        # 返回结果文件和统计信息（用于新的统计系统）
        return result_files, total_token_stats, total_paragraph_stats, total_table_stats
    else:
        # 返回原有格式（兼容原有调用）
        return result_files


def download_remote_fonts(lang: str):
    lang = lang.lower()
    LANG_NAME_MAP = {
        **{la: "GoNotoKurrent-Regular.ttf" for la in noto_list},
        **{
            la: f"SourceHanSerif{region}-Regular.ttf"
            for region, langs in {
                "CN": ["zh-cn", "zh-hans", "zh"],
                "TW": ["zh-tw", "zh-hant"],
                "JP": ["ja"],
                "KR": ["ko"],
            }.items()
            for la in langs
        },
    }
    font_name = LANG_NAME_MAP.get(lang, "GoNotoKurrent-Regular.ttf")

    # docker
    font_path = ConfigManager.get("NOTO_FONT_PATH", Path("/app", font_name).as_posix())
    if not Path(font_path).exists():
        font_path, _ = get_font_and_metadata(font_name)
        font_path = font_path.as_posix()

    logger.info(f"use font: {font_path}")

    return font_path
