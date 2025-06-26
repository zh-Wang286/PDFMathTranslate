#!/usr/bin/env python3
"""A command line tool for extracting text and images from PDF and
output it to plain text, html, xml or tags.
"""

from __future__ import annotations

import argparse
import logging
import sys
from string import Template
from typing import List, Optional
import time
import os
from datetime import datetime

from pdf2zh import __version__, log
from pdf2zh.high_level import translate, download_remote_fonts
from pdf2zh.doclayout import OnnxModel, ModelInstance

from pdf2zh.config import ConfigManager
from babeldoc.translation_config import TranslationConfig as YadtConfig
from babeldoc.high_level import async_translate as yadt_translate
from babeldoc.high_level import init as yadt_init
from babeldoc.main import create_progress_handler

from pdf2zh.high_level import analyze_pdf

# ==================================================
# A constant to represent the approximate number of tokens in the prompt template.
# This value is used for time estimation.
from pdf2zh.translator import TEMPLATE_PROMPT_TOKEN_COUNT
TPS = 60 # general model token per second
AVG_THINK_CONTENT = 450 # thinking model token per second
# ==================================================

logger = logging.getLogger(__name__)


def create_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__, add_help=True)
    parser.add_argument(
        "files",
        type=str,
        default=None,
        nargs="*",
        help="One or more paths to PDF files.",
    )
    parser.add_argument(
        "--version",
        "-v",
        action="version",
        version=f"pdf2zh v{__version__}",
    )
    parser.add_argument(
        "--debug",
        "-d",
        default=False,
        action="store_true",
        help="Use debug logging level.",
    )
    parse_params = parser.add_argument_group(
        "Parser",
        description="Used during PDF parsing",
    )
    parse_params.add_argument(
        "--pages",
        "-p",
        type=str,
        help="The list of page numbers to parse (1-based). Examples: '1' for first page, '1,2,3' for multiple pages, '1-3' for range.",
    )
    parse_params.add_argument(
        "--vfont",
        "-f",
        type=str,
        default="",
        help="The regex to math font name of formula.",
    )
    parse_params.add_argument(
        "--vchar",
        "-c",
        type=str,
        default="",
        help="The regex to math character of formula.",
    )
    parse_params.add_argument(
        "--lang-in",
        "-li",
        type=str,
        default="en",
        help="The code of source language.",
    )
    parse_params.add_argument(
        "--lang-out",
        "-lo",
        type=str,
        default="zh",
        help="The code of target language.",
    )
    parse_params.add_argument(
        "--service",
        "-s",
        type=str,
        default="google",
        help="The service to use for translation.",
    )
    parse_params.add_argument(
        "--output",
        "-o",
        type=str,
        default="",
        help="Output directory for files.",
    )
    parse_params.add_argument(
        "--thread",
        "-t",
        type=int,
        default=4,
        help="The number of threads to execute translation.",
    )
    parse_params.add_argument(
        "--interactive",
        "-i",
        action="store_true",
        help="Interact with GUI.",
    )
    parse_params.add_argument(
        "--share",
        action="store_true",
        help="Enable Gradio Share",
    )
    parse_params.add_argument(
        "--flask",
        action="store_true",
        help="flask",
    )
    parse_params.add_argument(
        "--celery",
        action="store_true",
        help="celery",
    )
    parse_params.add_argument(
        "--authorized",
        type=str,
        nargs="+",
        help="user name and password.",
    )
    parse_params.add_argument(
        "--prompt",
        type=str,
        help="user custom prompt.",
    )

    parse_params.add_argument(
        "--compatible",
        "-cp",
        action="store_true",
        help="Convert the PDF file into PDF/A format to improve compatibility.",
    )

    parse_params.add_argument(
        "--onnx",
        type=str,
        help="custom onnx model path.",
    )

    parse_params.add_argument(
        "--serverport",
        type=int,
        help="custom WebUI port.",
    )

    parse_params.add_argument(
        "--dir",
        action="store_true",
        help="translate directory.",
    )

    parse_params.add_argument(
        "--config",
        type=str,
        help="config file.",
    )

    parse_params.add_argument(
        "--babeldoc",
        default=False,
        action="store_true",
        help="Use experimental backend babeldoc.",
    )

    parse_params.add_argument(
        "--skip-subset-fonts",
        action="store_true",
        help="Skip font subsetting. "
        "This option can improve compatibility "
        "but will increase the size of the output file.",
    )

    parse_params.add_argument(
        "--analysis-only",
        action="store_true",
        help="分析PDF，不翻译",
    )

    parse_params.add_argument(
        "--analysis-report",
        action="store_true",
        help="打印详细的分析报告",
    )

    parse_params.add_argument(
        "--ignore-cache",
        action="store_true",
        help="Ignore cache and force retranslation.",
    )

    parse_params.add_argument(
        "--mcp", action="store_true", help="Launch pdf2zh MCP server in STDIO mode"
    )

    parse_params.add_argument(
        "--sse", action="store_true", help="Launch pdf2zh MCP server in SSE mode"
    )

    parse_params.add_argument(
        "--reasoning",
        type=lambda x: x.lower() == 'true',
        default=False,
        help="Use alternative reasoning mode for time estimation (true/false).",
    )

    return parser


def parse_args(args: Optional[List[str]]) -> argparse.Namespace:
    parsed_args = create_parser().parse_args(args=args)

    if parsed_args.pages:
        pages = []
        for p in parsed_args.pages.split(","):
            if "-" in p:
                start, end = p.split("-")
                pages.extend(range(int(start) - 1, int(end)))
            else:
                pages.append(int(p) - 1)
        parsed_args.raw_pages = parsed_args.pages
        parsed_args.pages = pages

    return parsed_args


def find_all_files_in_directory(directory_path):
    """
    Recursively search all PDF files in the given directory and return their paths as a list.

    :param directory_path: str, the path to the directory to search
    :return: list of PDF file paths
    """
    # Check if the provided path is a directory
    if not os.path.isdir(directory_path):
        raise ValueError(f"The provided path '{directory_path}' is not a directory.")

    file_paths = []

    # Walk through the directory recursively
    for root, _, files in os.walk(directory_path):
        for file in files:
            # Check if the file is a PDF
            if file.lower().endswith(".pdf"):
                # Append the full file path to the list
                file_paths.append(os.path.join(root, file))

    return file_paths


def estimate_time_default(stats: dict) -> float:
    """
    估算处理时间（默认模式）
    总时间 = 输入时间 + 输出时间
    输入时间 = (段落token总数 + 段落数量*模板token + 表格token总数 + 表格单元格数量*模板token) / 100
    输出时间 = (段落token总数 + 表格token总数) / TPS / 3
    """
    total_paragraph_tokens = stats.get('total_paragraph_tokens', 0)
    total_table_tokens = stats.get('total_table_tokens', 0)
    total_paragraph_count = stats.get('total_paragraph_count', 0)
    total_table_cell_count = stats.get('total_table_cell_count', 0)
    
    # 计算输入时间
    input_tokens = (total_paragraph_tokens + total_paragraph_count * TEMPLATE_PROMPT_TOKEN_COUNT + 
                   total_table_tokens + total_table_cell_count * TEMPLATE_PROMPT_TOKEN_COUNT)
    input_time = input_tokens / 100
    
    # 计算输出时间
    output_tokens = total_paragraph_tokens + total_table_tokens
    output_time = output_tokens / TPS / 3
    
    # 总时间
    estimated_time_seconds = input_time + output_time
    
    logger.debug(f"Input time: {input_time:.2f}s, Output time: {output_time:.2f}s")
    return estimated_time_seconds


def estimate_time_reasoning(stats: dict) -> float:
    """
    估算处理时间（推理模式）
    总时间 = 输入时间 + 输出时间
    输入时间 = (段落token总数 + 段落数量*模板token + 表格token总数 + 表格单元格数量*模板token) / 100
    输出时间 = (段落token总数 + 段落数量*思考token + 表格token总数 + 表格单元格数量*思考token) / TPS / 2
    """
    total_paragraph_tokens = stats.get('total_paragraph_tokens', 0)
    total_table_tokens = stats.get('total_table_tokens', 0)
    total_paragraph_count = stats.get('total_paragraph_count', 0)
    total_table_cell_count = stats.get('total_table_cell_count', 0)
    
    # 计算输入时间
    input_tokens = (total_paragraph_tokens + total_paragraph_count * TEMPLATE_PROMPT_TOKEN_COUNT + 
                   total_table_tokens + total_table_cell_count * TEMPLATE_PROMPT_TOKEN_COUNT)
    input_time = input_tokens / 100
    
    # 计算输出时间（包含思考token）
    output_tokens = (total_paragraph_tokens + total_paragraph_count * AVG_THINK_CONTENT + 
                    total_table_tokens + total_table_cell_count * AVG_THINK_CONTENT)
    output_time = output_tokens / TPS / 2
    
    # 总时间
    estimated_time_seconds = input_time + output_time
    
    logger.debug(f"Input time: {input_time:.2f}s, Output time: {output_time:.2f}s")
    return estimated_time_seconds


def print_analysis_report(stats: dict, estimated_time: float, to_console: bool = False):
    """打印分析报告。如果 to_console 为 True，则打印到控制台，否则使用 debug 日志。"""
    output_func = print if to_console else logger.debug

    # 写入标题
    title = " PDF Translation Analysis Report "
    separator = "=" * ((59 - len(title)) // 2)
    header = f"{separator}{title}{separator}"
    output_func(header)
    
    # 写入基本统计信息
    output_func("\n1. Content Statistics")
    output_func("-" * 59)
    output_func(f"Total Pages: {stats.get('page_count', 0)}")
    output_func(f"Total Paragraphs: {stats.get('total_paragraph_count', 0)}")
    output_func(f"Total Paragraph Tokens: {stats.get('total_paragraph_tokens', 0)}")
    output_func(
        f"Actual Paragraph Tokens (for estimation): {stats.get('total_paragraph_count', 0) + stats.get('total_paragraph_count', 0) * TEMPLATE_PROMPT_TOKEN_COUNT}"
    )
    output_func(f"Total Table Cells: {stats.get('total_table_cell_count', 0)}")
    output_func(f"Total Table Tokens: {stats.get('total_table_tokens', 0)}")
    output_func(f"Total Estimated Content Tokens: {stats.get('total_paragraph_tokens', 0) + stats.get('total_table_tokens', 0)}")
    
    # 写入页面详细信息
    output_func("\n2. Page Details")
    output_func("-" * 59)
    for page_num, page_data in stats.get("pages", {}).items():
        paragraph_count = page_data.get("paragraph_count", 0)
        table_count = page_data.get("table_count", 0)
        table_cell_count = page_data.get("table_cell_count", 0)
        table_token_count = page_data.get("table_token_count", 0)
        output_func(
            f"Page {page_num + 1}: "
            f"{paragraph_count} paragraphs ({page_data.get('paragraph_token_count', 0)} tokens) | "
            f"{table_count} tables "
            f"({table_cell_count} cells, {table_token_count} tokens)"
        )

    if to_console:
        output_func("=" * 59)


def save_time_log(file_path: str, estimated_time: float, actual_time: float) -> None:
    """保存时间日志到文件

    Args:
        file_path: PDF文件路径
        estimated_time: 预估时间
        actual_time: 实际时间
    """
    try:
        # 获取PDF文件名（不含路径和扩展名）
        pdf_name = os.path.splitext(os.path.basename(file_path))[0]
        # 生成日志文件名：PDF文件名_时间戳.log
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_filename = f"{pdf_name}_{timestamp}.log"
        
        # 计算时间差异
        time_diff = actual_time - estimated_time
        diff_percentage = (time_diff / estimated_time) * 100 if estimated_time > 0 else 0
        
        # 写入日志文件
        with open(log_filename, "w", encoding="utf-8") as f:
            f.write("PDF翻译时间分析报告\n")
            f.write("=" * 50 + "\n")
            f.write(f"文件名: {file_path}\n")
            f.write(f"开始时间: {datetime.fromtimestamp(time.time() - actual_time).strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"结束时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write("-" * 50 + "\n")
            f.write(f"预估时间: {estimated_time:.2f} 秒\n")
            f.write(f"实际时间: {actual_time:.2f} 秒\n")
            f.write(f"时间偏差: {time_diff:+.2f} 秒\n")
            f.write(f"偏差百分比: {diff_percentage:+.1f}%\n")
            f.write("=" * 50 + "\n")
        
        logger.info(f"时间分析报告已保存到: {log_filename}")
    except Exception as e:
        logger.error(f"保存时间日志时发生错误: {e}")


def save_analysis_report(stats: dict, estimated_time: float, actual_time: float, filename: str):
    """保存分析报告到文件。"""
    with open(filename, 'w', encoding='utf-8') as f:
        # 写入标题
        title = " PDF Translation Analysis Report "
        separator = "=" * ((59 - len(title)) // 2)
        header = f"{separator}{title}{separator}\n"
        f.write(header)
        
        # 写入基本统计信息
        f.write("\n1. Content Statistics\n")
        f.write("-" * 59 + "\n")
        f.write(f"Total Pages: {stats.get('page_count', 0)}\n")
        f.write(f"Total Paragraphs: {stats.get('total_paragraph_count', 0)}\n")
        f.write(f"Total Paragraph Tokens: {stats.get('total_paragraph_tokens', 0)}\n")
        f.write(
            f"Actual Paragraph Tokens (for estimation): {stats.get('total_paragraph_count', 0) + stats.get('total_paragraph_count', 0) * TEMPLATE_PROMPT_TOKEN_COUNT}\n"
        )
        f.write(f"Total Table Cells: {stats.get('total_table_cell_count', 0)}\n")
        f.write(f"Total Table Tokens: {stats.get('total_table_tokens', 0)}\n")
        f.write(f"Total Estimated Content Tokens: {stats.get('total_paragraph_tokens', 0) + stats.get('total_table_tokens', 0)}\n")
        
        # 写入页面详细信息
        f.write("\n2. Page Details\n")
        f.write("-" * 59 + "\n")
        for page_num, page_data in stats.get("pages", {}).items():
            paragraph_count = page_data.get("paragraph_count", 0)
            table_count = page_data.get("table_count", 0)
            table_cell_count = page_data.get("table_cell_count", 0)
            table_token_count = page_data.get("table_token_count", 0)
            f.write(
                f"Page {page_num + 1}: "
                f"{paragraph_count} paragraphs ({page_data.get('paragraph_token_count', 0)} tokens) | "
                f"{table_count} tables "
                f"({table_cell_count} cells, {table_token_count} tokens)\n"
            )
            
        # 写入时间分析
        f.write("\n3. Time Analysis\n")
        f.write("-" * 59 + "\n")
        f.write(f"Estimated Processing Time: {estimated_time:.2f} seconds\n")
        f.write(f"Actual Processing Time: {actual_time:.2f} seconds\n")
        difference = actual_time - estimated_time
        percentage = abs(difference) / estimated_time * 100
        if difference > 0:
            f.write(f"Time Difference: +{difference:.2f} seconds ({percentage:.1f}% longer than estimated)\n")
        else:
            f.write(f"Time Difference: {difference:.2f} seconds ({percentage:.1f}% shorter than estimated)\n")
            
        # 写入结尾分隔符
        f.write("\n" + "=" * 59 + "\n")
        f.write(f"Report generated at: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")


def main(args: Optional[List[str]] = None) -> int:
    import time
    start_time = time.time()  # 记录开始时间
    estimated_time = 0  # 初始化预估时间变量
    
    from rich.logging import RichHandler

    logging.basicConfig(level=logging.INFO, handlers=[RichHandler()])

    # disable httpx, openai, httpcore, http11 logs
    logging.getLogger("httpx").setLevel("CRITICAL")
    logging.getLogger("httpx").propagate = False
    logging.getLogger("openai").setLevel("CRITICAL")
    logging.getLogger("openai").propagate = False
    logging.getLogger("httpcore").setLevel("CRITICAL")
    logging.getLogger("httpcore").propagate = False
    logging.getLogger("http11").setLevel("CRITICAL")
    logging.getLogger("http11").propagate = False

    parsed_args = parse_args(args)

    if parsed_args.config:
        ConfigManager.custome_config(parsed_args.config)

    if parsed_args.debug:
        log.setLevel(logging.DEBUG)

    if parsed_args.onnx:
        ModelInstance.value = OnnxModel(parsed_args.onnx)
    else:
        ModelInstance.value = OnnxModel.load_available()

    if parsed_args.interactive:
        from pdf2zh.gui import setup_gui

        if parsed_args.serverport:
            setup_gui(
                parsed_args.share, parsed_args.authorized, int(parsed_args.serverport)
            )
        else:
            setup_gui(parsed_args.share, parsed_args.authorized)
        return 0

    if parsed_args.flask:
        from pdf2zh.backend import flask_app

        flask_app.run(port=11008)
        return 0

    if parsed_args.celery:
        from pdf2zh.backend import celery_app

        celery_app.start(argv=sys.argv[2:])
        return 0

    if parsed_args.prompt:
        try:
            with open(parsed_args.prompt, "r", encoding="utf-8") as file:
                content = file.read()
            parsed_args.prompt = Template(content)
        except Exception:
            raise ValueError("prompt error.")

    if parsed_args.mcp:
        logging.getLogger("mcp").setLevel(logging.ERROR)
        from pdf2zh.mcp_server import create_mcp_app, create_starlette_app

        mcp = create_mcp_app()
        if parsed_args.sse:
            import uvicorn

            starlette_app = create_starlette_app(mcp._mcp_server)
            uvicorn.run(starlette_app)
            return 0
        mcp.run()
        return 0

    print(parsed_args)

    # Perform analysis before translation
    if not parsed_args.babeldoc and parsed_args.files:
        logger.info("="*20 + " PDF Analysis Report " + "="*20)
        try:
            with open(parsed_args.files[0], "rb") as f:
                pdf_bytes = f.read()

            stats = analyze_pdf(
                pdf_bytes=pdf_bytes,
                model=ModelInstance.value,
                pages=parsed_args.pages,
            )

            # 根据不同的推理模式选择时间估算方法
            if parsed_args.reasoning:
                estimated_time = estimate_time_reasoning(stats)
            else:
                estimated_time = estimate_time_default(stats)

            # 输出估算时间
            logger.info(f"Estimated processing time: {estimated_time:.2f} seconds")
            
            # 打印详细分析报告
            print_analysis_report(stats, estimated_time, parsed_args.analysis_report)

        except Exception as e:
            logger.error(f"An error occurred during PDF analysis: {e}", exc_info=True)
        logger.info("="*59)

    # 如果只需要分析，则在这里返回
    if parsed_args.analysis_only:
        logger.info("Analysis completed. Skipping translation as --analysis-only was specified.")
        return 0

    result = 0
    if parsed_args.babeldoc:
        result = yadt_main(parsed_args)
    elif parsed_args.dir:
        untranlate_file = find_all_files_in_directory(parsed_args.files[0])
        parsed_args.files = untranlate_file
        translate(model=ModelInstance.value, **vars(parsed_args))
    else:
        translate(model=ModelInstance.value, **vars(parsed_args))

    # 计算并输出总运行时间
    end_time = time.time()
    actual_time = end_time - start_time
    
    # 输出时间对比报告
    logger.info("="*20 + " Time Analysis Report " + "="*20)
    if estimated_time > 0:
        logger.info(f"Estimated time: {estimated_time:.2f} seconds")
        logger.info(f"Actual time: {actual_time:.2f} seconds")
        time_diff = actual_time - estimated_time
        diff_percentage = (time_diff / estimated_time) * 100
        if time_diff > 0:
            logger.info(f"Difference: +{time_diff:.2f} seconds ({diff_percentage:.1f}% longer than estimated)")
        else:
            logger.info(f"Difference: {time_diff:.2f} seconds ({abs(diff_percentage):.1f}% shorter than estimated)")
        
        # 保存时间日志
        if parsed_args.files and len(parsed_args.files) > 0:
            save_time_log(parsed_args.files[0], estimated_time, actual_time)
    else:
        logger.info(f"Total execution time: {actual_time:.2f} seconds")
    logger.info("="*59)
    
    # 保存时间分析报告
    if parsed_args.analysis_report:
        try:
            # 使用第一个文件名作为基础
            base_filename = os.path.splitext(os.path.basename(parsed_args.files[0]))[0]
            # 生成报告文件名：原文件名_YYYYMMDD_HHMMSS_analysis.txt
            report_filename = f"{base_filename}_{time.strftime('%Y%m%d_%H%M%S')}_analysis.txt"
            save_analysis_report(stats, estimated_time, actual_time, report_filename)
            logger.info(f"分析报告已保存到: {report_filename}")
        except Exception as e:
            logger.error(f"保存分析报告时发生错误: {e}")

    return result


def yadt_main(parsed_args) -> int:
    if parsed_args.dir:
        untranlate_file = find_all_files_in_directory(parsed_args.files[0])
    else:
        untranlate_file = parsed_args.files
    lang_in = parsed_args.lang_in
    lang_out = parsed_args.lang_out
    ignore_cache = parsed_args.ignore_cache
    outputdir = None
    if parsed_args.output:
        outputdir = parsed_args.output

    # yadt require init before translate
    yadt_init()
    font_path = download_remote_fonts(lang_out.lower())

    param = parsed_args.service.split(":", 1)
    service_name = param[0]
    service_model = param[1] if len(param) > 1 else None

    envs = {}
    prompt = []

    if parsed_args.prompt:
        try:
            with open(parsed_args.prompt, "r", encoding="utf-8") as file:
                content = file.read()
            prompt = Template(content)
        except Exception:
            raise ValueError("prompt error.")

    from pdf2zh.translator import (
        AzureOpenAITranslator,
        GoogleTranslator,
        BingTranslator,
        DeepLTranslator,
        DeepLXTranslator,
        OllamaTranslator,
        OpenAITranslator,
        ZhipuTranslator,
        ModelScopeTranslator,
        SiliconTranslator,
        GeminiTranslator,
        AzureTranslator,
        TencentTranslator,
        DifyTranslator,
        AnythingLLMTranslator,
        XinferenceTranslator,
        ArgosTranslator,
        GrokTranslator,
        GroqTranslator,
        DeepseekTranslator,
        OpenAIlikedTranslator,
        QwenMtTranslator,
    )

    for translator in [
        GoogleTranslator,
        BingTranslator,
        DeepLTranslator,
        DeepLXTranslator,
        OllamaTranslator,
        XinferenceTranslator,
        AzureOpenAITranslator,
        OpenAITranslator,
        ZhipuTranslator,
        ModelScopeTranslator,
        SiliconTranslator,
        GeminiTranslator,
        AzureTranslator,
        TencentTranslator,
        DifyTranslator,
        AnythingLLMTranslator,
        ArgosTranslator,
        GrokTranslator,
        GroqTranslator,
        DeepseekTranslator,
        OpenAIlikedTranslator,
        QwenMtTranslator,
    ]:
        if service_name == translator.name:
            translator = translator(
                lang_in,
                lang_out,
                service_model,
                envs=envs,
                prompt=prompt,
                ignore_cache=ignore_cache,
            )
            break
    else:
        raise ValueError("Unsupported translation service")
    import asyncio

    for file in untranlate_file:
        file = file.strip("\"'")
        yadt_config = YadtConfig(
            input_file=file,
            font=font_path,
            pages=",".join((str(x) for x in getattr(parsed_args, "raw_pages", []))),
            output_dir=outputdir,
            doc_layout_model=None,
            translator=translator,
            debug=parsed_args.debug,
            lang_in=lang_in,
            lang_out=lang_out,
            no_dual=False,
            no_mono=False,
            qps=parsed_args.thread,
        )

        async def yadt_translate_coro(yadt_config):
            progress_context, progress_handler = create_progress_handler(yadt_config)
            # 开始翻译
            with progress_context:
                async for event in yadt_translate(yadt_config):
                    progress_handler(event)
                    if yadt_config.debug:
                        logger.debug(event)
                    if event["type"] == "finish":
                        result = event["translate_result"]
                        logger.info("Translation Result:")
                        logger.info(f"  Original PDF: {result.original_pdf_path}")
                        logger.info(f"  Time Cost: {result.total_seconds:.2f}s")
                        logger.info(f"  Mono PDF: {result.mono_pdf_path or 'None'}")
                        logger.info(f"  Dual PDF: {result.dual_pdf_path or 'None'}")
                        break

        asyncio.run(yadt_translate_coro(yadt_config))
    return 0


if __name__ == "__main__":
    sys.exit(main())
