#!/usr/bin/env python3
"""A command line tool for extracting text and images from PDF and
output it to plain text, html, xml or tags.
"""

from __future__ import annotations

import argparse
import logging
import sys
from string import Template
from typing import List, Optional, Any
import time
import os
from datetime import datetime

from .doclayout import OnnxModel, ModelInstance
from .high_level import translate, download_remote_fonts
from .config import ConfigManager
from .statistics import perform_pre_analysis, collect_runtime_stats, PDFTranslationStatistics

from babeldoc.translation_config import TranslationConfig as YadtConfig
from babeldoc.high_level import async_translate as yadt_translate
from babeldoc.high_level import init as yadt_init
from babeldoc.main import create_progress_handler

from pdf2zh.high_level import analyze_pdf

# 导入版本信息
try:
    from pdf2zh import __version__
except ImportError:
    __version__ = "unknown"

logger = logging.getLogger(__name__)

# ==================================================
# A constant to represent the approximate number of tokens in the prompt template.
# This value is used for time estimation.
# from pdf2zh.translator import TEMPLATE_PROMPT_TOKEN_COUNT
# TPS = 60 # general model token per second
# AVG_THINK_CONTENT = 450 # thinking model token per second
# ==================================================

# =================== 统计功能通用接口 ===================

def initialize_statistics(
    needs_statistics: bool,
    files: Optional[List[str]] = None,
    service: str = "",
    thread_count: int = 4
) -> Optional[PDFTranslationStatistics]:
    """
    初始化统计功能
    
    Args:
        needs_statistics: 是否需要统计功能
        files: 输入文件列表
        service: 翻译服务名称
        thread_count: 线程数量
        
    Returns:
        PDFTranslationStatistics对象或None
    """
    if not needs_statistics:
        return None
        
    logger.debug("初始化统计功能...")
    stats_obj = PDFTranslationStatistics()
    stats_obj.start_runtime_tracking()
    
    if files:
        stats_obj.set_input_files(files)
        
    # 设置运行时配置
    if service or thread_count:
        stats_obj.set_runtime_config(service=service, thread_count=thread_count)
        
    logger.debug("统计功能初始化完成")
    return stats_obj


def perform_pdf_pre_analysis(
    stats_obj: Optional[PDFTranslationStatistics],
    files: List[str],
    model,
    pages: Optional[List[int]] = None,
    service: str = "",
    reasoning: bool = False
) -> Optional[PDFTranslationStatistics]:
    """
    执行PDF预分析
    
    Args:
        stats_obj: 统计对象，如果为None则跳过分析
        files: 输入文件列表
        model: ONNX模型实例
        pages: 页面列表
        service: 翻译服务名称
        reasoning: 是否为推理模式
        
    Returns:
        更新后的统计对象或None
    """
    if not stats_obj or not files:
        return stats_obj
        
    logger.info("执行预处理分析...")
    
    # 检查是否为推理模式
    is_reasoning = reasoning or (
        service and any(reasoning_service in service.lower() 
                       for reasoning_service in ['r1', 'reasoning', 'think', 'o1'])
    )
    
    try:
        # 读取文件内容
        file_contents = read_inputs(files)
        
        # 执行预分析（使用第一个文件）
        for i, pdf_bytes in enumerate(file_contents):
            logger.info(f"分析文件 {i+1}/{len(file_contents)}: {files[i]}")
            
            current_stats = perform_pre_analysis(
                pdf_bytes=pdf_bytes,
                model=model,
                pages=pages,
                is_reasoning=is_reasoning,
                cancellation_event=None
            )
            
            if i == 0:
                # 使用第一个文件的分析结果
                stats_obj = current_stats
                stats_obj.set_input_files(files)
                # 确保开始运行时跟踪
                stats_obj.start_runtime_tracking()
                # 重新设置运行时配置
                if hasattr(stats_obj, '_service') or hasattr(stats_obj, '_thread_count'):
                    service_val = getattr(stats_obj, '_service', service)
                    thread_val = getattr(stats_obj, '_thread_count', 4)
                    stats_obj.set_runtime_config(service=service_val, thread_count=thread_val)
            else:
                # 这里可以扩展为合并多个文件的统计信息
                logger.debug(f"跳过文件 {i+1} 的统计合并（简化处理）")
                
        logger.info("PDF预分析完成")
        return stats_obj
        
    except Exception as e:
        logger.error(f"预分析过程中发生错误: {e}")
        return stats_obj


def collect_translation_statistics(
    stats_obj: Optional[PDFTranslationStatistics],
    translation_result: Any
) -> Optional[PDFTranslationStatistics]:
    """
    收集翻译统计信息
    
    Args:
        stats_obj: 统计对象
        translation_result: 翻译结果
        
    Returns:
        更新后的统计对象
    """
    if not stats_obj or not translation_result:
        return stats_obj
        
    try:
        if isinstance(translation_result, tuple) and len(translation_result) >= 4:
            result_files, token_stats, paragraph_stats, table_stats = translation_result[:4]
            # 收集运行时统计
            stats_obj = collect_runtime_stats(stats_obj, token_stats, paragraph_stats, table_stats)
            logger.debug("翻译统计信息收集完成")
        else:
            logger.warning("翻译结果格式不符合预期，跳过统计收集")
    except Exception as e:
        logger.error(f"收集翻译统计时发生错误: {e}")
        
    return stats_obj


def finalize_statistics(
    stats_obj: Optional[PDFTranslationStatistics],
    output_dir: str = ".",
    generate_report: bool = True
) -> Optional[str]:
    """
    结束统计功能并生成报告
    
    Args:
        stats_obj: 统计对象
        output_dir: 输出目录
        generate_report: 是否生成报告
        
    Returns:
        生成的报告文件路径或None
    """
    if not stats_obj:
        return None
        
    try:
        # 结束运行时统计跟踪
        stats_obj.end_runtime_tracking()
        logger.debug("统计跟踪已结束")
        
        # 生成统计报告
        if generate_report:
            # 处理输出目录路径，移除可能的.pdf后缀
            clean_output_dir = output_dir
            if clean_output_dir.endswith('.pdf'):
                clean_output_dir = os.path.splitext(clean_output_dir)[0]
                
            log_file = stats_obj.generate_report_log(clean_output_dir)
            logger.info(f"详细统计报告已保存到: {log_file}")
            return log_file
    except Exception as e:
        logger.error(f"结束统计功能时发生错误: {e}")
        
    return None

# ==================================================

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

    parse_params.add_argument(
        "--no-concurrent-table-translation",
        dest="use_concurrent_table_translation",
        default=True,
        action="store_false",
        help="Disable concurrent translation for tables.",
    )

    parse_params.add_argument(
        "--draw-layout-boxes",
        action="store_true",
        help="Draw YOLO layout detection boxes on the output PDF for debugging.",
    )

    return parser


def parse_args(args: Optional[List[str]] = None) -> argparse.Namespace:
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


def main(args: Optional[List[str]] = None) -> int:
    import time
    
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
        logger.setLevel(logging.DEBUG)

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

    # 检查是否需要统计功能
    needs_statistics = (
        parsed_args.analysis_only or 
        parsed_args.analysis_report or 
        getattr(parsed_args, 'statistics', False)  # 为未来可能的统计参数预留
    )

    # 初始化统计功能
    stats_obj = initialize_statistics(
        needs_statistics=needs_statistics,
        files=parsed_args.files,
        service=getattr(parsed_args, 'service', ''),
        thread_count=getattr(parsed_args, 'thread', 4)
    )

    result = 0

    try:
        if parsed_args.analysis_only:
            # 仅分析模式
            if not parsed_args.files:
                logger.error("分析模式需要指定输入文件")
                return 1
                
            logger.info("开始PDF分析模式...")
            
            # 执行预分析
            stats_obj = perform_pdf_pre_analysis(
                stats_obj=stats_obj,
                files=parsed_args.files,
                model=ModelInstance.value,
                pages=parsed_args.pages,
                service=getattr(parsed_args, 'service', ''),
                reasoning=getattr(parsed_args, 'reasoning', False)
            )
                    
            logger.info("PDF分析完成")
            
        elif parsed_args.babeldoc:
            result = yadt_main(parsed_args)
        elif parsed_args.dir:
            untranlate_file = find_all_files_in_directory(parsed_args.files[0])
            parsed_args.files = untranlate_file
            
            # 执行预分析
            stats_obj = perform_pdf_pre_analysis(
                stats_obj=stats_obj,
                files=parsed_args.files,
                model=ModelInstance.value,
                pages=parsed_args.pages,
                service=getattr(parsed_args, 'service', ''),
                reasoning=getattr(parsed_args, 'reasoning', False)
            )
            
            # 执行翻译
            if needs_statistics:
                translation_result = translate(model=ModelInstance.value, stats_obj=stats_obj, **vars(parsed_args))
                # 收集翻译统计
                stats_obj = collect_translation_statistics(stats_obj, translation_result)
            else:
                # 标准调用，不需要统计
                translate(model=ModelInstance.value, **vars(parsed_args))
            
        else:
            # 标准翻译模式
            # 执行预分析
            stats_obj = perform_pdf_pre_analysis(
                stats_obj=stats_obj,
                files=parsed_args.files,
                model=ModelInstance.value,
                pages=parsed_args.pages,
                service=getattr(parsed_args, 'service', ''),
                reasoning=getattr(parsed_args, 'reasoning', False)
            )
            
            # 执行翻译
            if needs_statistics:
                translation_result = translate(model=ModelInstance.value, stats_obj=stats_obj, **vars(parsed_args))
                # 收集翻译统计
                stats_obj = collect_translation_statistics(stats_obj, translation_result)
            else:
                # 标准调用，不需要统计
                translate(model=ModelInstance.value, **vars(parsed_args))

    except Exception as e:
        logger.error(f"执行过程中发生错误: {e}")
        result = 1
    finally:
        # 结束统计功能并生成报告
        finalize_statistics(
            stats_obj=stats_obj,
            output_dir=getattr(parsed_args, 'output', '.'),
            generate_report=needs_statistics
        )

    return result


def yadt_main(parsed_args):
    """BabelDoc translation main function"""
    # 初始化babeldoc
    yadt_init()
    from babeldoc.high_level import async_translate as babeldoc_translate
    
    # 检查是否需要统计功能（babeldoc模式下可能也需要统计）
    needs_statistics = (
        getattr(parsed_args, 'analysis_only', False) or 
        getattr(parsed_args, 'analysis_report', False) or 
        getattr(parsed_args, 'statistics', False)
    )
    
    # 初始化统计功能
    stats_obj = initialize_statistics(
        needs_statistics=needs_statistics,
        files=parsed_args.files,
        service=getattr(parsed_args, 'service', ''),
        thread_count=getattr(parsed_args, 'thread', 4)
    )
    
    try:
        # 解析服务参数
        param = parsed_args.service.split(":", 1)
        service_name = param[0]
        service_model = param[1] if len(param) > 1 else None
        
        # 设置环境变量
        envs = getattr(parsed_args, 'envs', {}) or {}
        
        # 处理自定义prompt
        prompt = None
        if parsed_args.prompt:
            try:
                with open(parsed_args.prompt, "r", encoding="utf-8") as file:
                    content = file.read()
                prompt = Template(content)
            except Exception:
                raise ValueError("prompt error.")
        
        # 导入所有翻译器
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
        
        # 查找匹配的翻译器
        translator = None
        for translator_class in [
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
            if service_name == translator_class.name:
                translator = translator_class(
                    parsed_args.lang_in,
                    parsed_args.lang_out,
                    service_model,
                    envs=envs,
                    prompt=prompt,
                    ignore_cache=getattr(parsed_args, 'ignore_cache', False),
                )
                break
        
        if not translator:
            raise ValueError("Unsupported translation service")
        
        # 打印翻译服务信息
        service_info = translator.get_service_info()
        logger.info("==================== 翻译服务信息 ====================")
        logger.info(f"服务名称: {service_info['name']}")
        logger.info(f"使用模型: {service_info['model']}")
        logger.info(f"源语言: {service_info['lang_in']} -> 目标语言: {service_info['lang_out']}")
        logger.info(f"缓存启用: {'是' if service_info['cache_enabled'] else '否'}")
        if 'envs' in service_info and service_info['envs']:
            logger.info("服务配置:")
            for key, value in service_info['envs'].items():
                # 对于API密钥等敏感信息，只显示部分内容
                if any(sensitive in key.lower() for sensitive in ['key', 'token', 'secret']):
                    if value and len(str(value)) > 8:
                        value = f"{str(value)[:4]}...{str(value)[-4:]}"
                logger.info(f"  {key}: {value}")
        logger.info("===================================================")
        
        # 获取字体路径
        from pdf2zh.high_level import download_remote_fonts
        font_path = download_remote_fonts(parsed_args.lang_out.lower())
        
        # 处理文件列表
        untranlate_file = parsed_args.files
        if parsed_args.dir:
            untranlate_file = find_all_files_in_directory(parsed_args.files[0])
        
        # 执行预分析（如果需要统计）
        if needs_statistics and untranlate_file:
            stats_obj = perform_pdf_pre_analysis(
                stats_obj=stats_obj,
                files=untranlate_file,
                model=ModelInstance.value if hasattr(ModelInstance, 'value') else None,
                pages=getattr(parsed_args, 'pages', None),
                service=getattr(parsed_args, 'service', ''),
                reasoning=getattr(parsed_args, 'reasoning', False)
            )
        
        # 设置输出目录
        outputdir = parsed_args.output or "."
        
        # 开始翻译
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
                lang_in=parsed_args.lang_in,
                lang_out=parsed_args.lang_out,
                no_dual=False,
                no_mono=False,
                qps=parsed_args.thread,
            )
            
            # 使用正确的babeldoc翻译方式
            async def yadt_translate_coro(yadt_config):
                async for event in babeldoc_translate(yadt_config):
                    if yadt_config.debug:
                        logger.debug(event)
                    if event["type"] == "finish":
                        result = event["translate_result"]
                        logger.info("Translation Result:")
                        logger.info(f"  Original PDF: {result.original_pdf_path}")
                        logger.info(f"  Time Cost: {result.total_seconds:.2f}s")
                        logger.info(f"  Mono PDF: {result.mono_pdf_path or 'None'}")
                        logger.info(f"  Dual PDF: {result.dual_pdf_path or 'None'}")
                        return result
            
            result = asyncio.run(yadt_translate_coro(yadt_config))
            logger.info(f"BabelDoc translation completed: {result}")
        
        return 0
    
    except Exception as e:
        logger.error(f"BabelDoc翻译过程中发生错误: {e}")
        return 1
    finally:
        # 结束统计功能并生成报告
        finalize_statistics(
            stats_obj=stats_obj,
            output_dir=getattr(parsed_args, 'output', '.'),
            generate_report=needs_statistics
        )


def translate_file(
    input_file: str,
    output_dir: Optional[str] = None,
    pages: Optional[list[int]] = None,
    lang_in: str = "en",
    lang_out: str = "zh",
    service: str = "google",
    thread: int = 4,
    *,
    vfont: str = "",
    vchar: str = "",
    custom_onnx_path: Optional[str] = None,
    analysis_only: bool = False,
    generate_analysis_report: bool = False,
    reasoning: bool = False,
    ignore_cache: bool = False,
    debug: bool = False,
    use_concurrent_table_translation: bool = False,
    draw_layout_boxes: bool = False,  # 新增：是否绘制YOLO检测框
    **kwargs: Any,
) -> (Optional[str], Optional[PDFTranslationStatistics]):
    """
    Programmatic entry point for PDF translation and analysis.

    This function encapsulates the core functionality of pdf2zh, allowing for
    translation, analysis, and statistics generation through a library call.

    Args:
        input_file: Path to the PDF file. Can be a local path or a URL.
        output_dir: Directory to save the translated file and reports.
                    Defaults to the input file's directory or current dir for URLs.
        pages: List of 0-indexed page numbers to process. None for all pages.
        lang_in: Source language code.
        lang_out: Target language code.
        service: Translation service to use.
        thread: Number of translation threads.
        vfont: Regex for formula font names.
        vchar: Regex for formula characters.
        custom_onnx_path: Path to a custom ONNX model.
        analysis_only: If True, performs only PDF analysis without translation.
        generate_analysis_report: If True, generates a detailed statistics report.
        reasoning: Use alternative reasoning mode for time estimation.
        ignore_cache: Ignore cache and force re-translation.
        debug: If True, enables debug logging level.
        use_concurrent_table_translation: Whether to enable concurrent translation for tables.
        compatible: Convert the PDF file into PDF/A format to improve compatibility.
        skip_subset_fonts: Skip font subsetting to improve compatibility at cost of file size.
        prompt: Custom prompt template for translation.
        babeldoc: Use experimental backend babeldoc.
        **kwargs: Additional arguments for the `translate` function.

    Returns:
        A tuple containing:
        - The path to the translated PDF file (str) or None if translation failed or was skipped.
        - A PDFTranslationStatistics object with the analysis and runtime data, or None if stats were not requested.
    """
    prompt_path = kwargs.pop("prompt", None)
    babeldoc = kwargs.pop("babeldoc", False)
    compatible = kwargs.pop("compatible", False)
    skip_subset_fonts = kwargs.pop("skip_subset_fonts", False)

    # --- Set up logging ---
    if debug:
        logger.setLevel(logging.DEBUG)

    # --- Handle Prompt Template ---
    prompt = None
    if prompt_path:
        try:
            with open(prompt_path, "r", encoding="utf-8") as file:
                prompt = Template(file.read())
        except Exception:
            raise ValueError("Failed to read prompt template.")
    else:
        # 使用默认的 TEMPLATE_PROMPT
        from pdf2zh.translator import TEMPLATE_PROMPT
        prompt = Template(TEMPLATE_PROMPT)

    # --- Handle Backend Selection ---
    if babeldoc:
        from babeldoc.high_level import init as yadt_init
        yadt_init()
        
        # Create config for babeldoc
        from babeldoc.translation_config import TranslationConfig as YadtConfig
        yadt_config = YadtConfig(
            input_file=input_file,
            output_dir=output_dir,
            pages=",".join(str(x + 1) for x in pages) if pages else None,
            lang_in=lang_in,
            lang_out=lang_out,
            doc_layout_model=None,
            debug=debug,
            qps=thread,
            no_dual=False,
            no_mono=False,
        )
        
        # Execute babeldoc translation
        import asyncio
        async def yadt_translate_coro(config):
            from babeldoc.high_level import async_translate
            async for event in async_translate(config):
                if debug:
                    logger.debug(event)
                if event["type"] == "finish":
                    result = event["translate_result"]
                    return result.dual_pdf_path or result.mono_pdf_path
        
        try:
            translated_file_path = asyncio.run(yadt_translate_coro(yadt_config))
            stats_obj = None # stats_obj is not available in this path
            return translated_file_path, stats_obj
        except Exception as e:
            logger.error(f"BabelDoc translation failed: {e}")
            stats_obj = None # stats_obj is not available in this path
            return None, stats_obj

    # --- Model Initialization ---
    if custom_onnx_path:
        model = OnnxModel(custom_onnx_path)
    else:
        if not hasattr(ModelInstance, "value") or not ModelInstance.value:
            ModelInstance.value = OnnxModel.load_available()
        model = ModelInstance.value

    if not model:
        raise RuntimeError(
            "Layout analysis model not found. Ensure a model is available "
            "or provide a path via `custom_onnx_path`."
        )

    # --- Initialize Translator ---
    # 导入所有翻译器类
    from pdf2zh.translator import (
        AzureOpenAITranslator, GoogleTranslator, BingTranslator,
        DeepLTranslator, DeepLXTranslator, OllamaTranslator,
        OpenAITranslator, ZhipuTranslator, ModelScopeTranslator,
        SiliconTranslator, GeminiTranslator, AzureTranslator,
        TencentTranslator, DifyTranslator, AnythingLLMTranslator,
        XinferenceTranslator, ArgosTranslator, GrokTranslator,
        GroqTranslator, DeepseekTranslator, OpenAIlikedTranslator,
        QwenMtTranslator,
    )

    # 解析服务参数
    param = service.split(":", 1)
    service_name = param[0]
    service_model = param[1] if len(param) > 1 else None

    # 查找匹配的翻译器
    translator = None
    translator_classes = [
        GoogleTranslator, BingTranslator, DeepLTranslator,
        DeepLXTranslator, OllamaTranslator, XinferenceTranslator,
        AzureOpenAITranslator, OpenAITranslator, ZhipuTranslator,
        ModelScopeTranslator, SiliconTranslator, GeminiTranslator,
        AzureTranslator, TencentTranslator, DifyTranslator,
        AnythingLLMTranslator, ArgosTranslator, GrokTranslator,
        GroqTranslator, DeepseekTranslator, OpenAIlikedTranslator,
        QwenMtTranslator,
    ]

    for translator_class in translator_classes:
        if service_name == translator_class.name:
            translator = translator_class(
                lang_in, lang_out, service_model,
                ignore_cache=ignore_cache,
                prompt=prompt,
                **kwargs
            )
            break

    if not translator:
        raise ValueError(f"Unsupported translation service: {service_name}")

    # 打印翻译服务信息
    service_info = translator.get_service_info()
    logger.info("==================== 翻译服务信息 ====================")
    logger.info(f"服务名称: {service_info['name']}")
    logger.info(f"使用模型: {service_info['model']}")
    logger.info(f"源语言: {service_info['lang_in']} -> 目标语言: {service_info['lang_out']}")
    logger.info(f"缓存启用: {'是' if service_info['cache_enabled'] else '否'}")
    logger.info("===================================================")

    # --- Path and Argument Handling ---
    if input_file.startswith(("http://", "https://")):
        if not output_dir:
            output_dir = "."
    else:
        if not os.path.exists(input_file):
            raise FileNotFoundError(f"Input file not found: {input_file}")
        if not output_dir:
            output_dir = os.path.dirname(input_file)
    
    os.makedirs(output_dir, exist_ok=True)

    # --- Read Input File ---
    try:
        pdf_bytes_list = read_inputs([input_file])
        if not pdf_bytes_list:
            raise ValueError("Could not read input file.")
        pdf_bytes = pdf_bytes_list[0]
    except Exception as e:
        logger.error(f"Failed to read input file {input_file}: {e}")
        return None, None

    # --- Determine Reasoning Mode ---
    is_reasoning = reasoning or any(
        s in service.lower() for s in ['r1', 'reasoning', 'think', 'o1']
    )

    # --- Statistics and Analysis ---
    needs_statistics = analysis_only or generate_analysis_report
    
    # 初始化统计功能
    stats_obj = initialize_statistics(
        needs_statistics=needs_statistics,
        files=[input_file],
        service=service,
        thread_count=thread
    )
    
    # 执行预分析
    stats_obj = perform_pdf_pre_analysis(
        stats_obj=stats_obj,
        files=[input_file],
        model=model,
        pages=pages,
        service=service,
        reasoning=is_reasoning
    )

    # --- Main Logic: Translation ---
    translated_file_path = None
    try:
        if analysis_only:
            logger.info("Analysis only mode. Skipping translation.")
        else:
            # Perform translation
            logger.info("Starting translation...")
            translation_result = translate(
                files=[input_file],
                output=output_dir,
                pages=pages,
                lang_in=lang_in,
                lang_out=lang_out,
                service=service,
                thread=thread,
                vfont=vfont,
                vchar=vchar,
                model=model,
                ignore_cache=ignore_cache,
                stats_obj=stats_obj,
                use_concurrent_table_translation=use_concurrent_table_translation,
                compatible=compatible,
                skip_subset_fonts=skip_subset_fonts,
                draw_layout_boxes=draw_layout_boxes,
                **kwargs,
            )

            if translation_result and translation_result[0]:
                translated_file_path = translation_result[0][0]
                # 收集翻译统计
                stats_obj = collect_translation_statistics(stats_obj, translation_result)
                logger.info(f"Translation successful. Output: {translated_file_path}")
            else:
                logger.error("Translation failed.")

    except Exception as e:
        logger.error(f"An error occurred during processing: {e}", exc_info=True)
        return None, stats_obj
    finally:
        # --- Finalize Statistics and Reporting ---
        finalize_statistics(
            stats_obj=stats_obj,
            output_dir=output_dir,
            generate_report=generate_analysis_report
        )
    
    return translated_file_path, stats_obj


def read_inputs(files: List[str]) -> List[bytes]:
    """Read input files and return their contents as bytes"""
    file_contents = []
    for file in files:
        if file.startswith(("http://", "https://")):
            # 处理在线文件
            import requests
            try:
                r = requests.get(file, allow_redirects=True)
                if r.status_code == 200:
                    file_contents.append(r.content)
                else:
                    r.raise_for_status()
            except Exception as e:
                logger.error(f"下载文件失败 {file}: {e}")
                raise
        else:
            # 处理本地文件
            with open(file, 'rb') as f:
                file_contents.append(f.read())
    return file_contents


if __name__ == "__main__":
    sys.exit(main())
