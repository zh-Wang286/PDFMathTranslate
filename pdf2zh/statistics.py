"""
PDF翻译统计管理模块

This module provides comprehensive statistics management for PDF translation tasks,
including pre-processing estimation and runtime statistics collection.
"""

import logging
import os
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, Optional, Any

from pdf2zh.translator import TEMPLATE_PROMPT_TOKEN_COUNT
from pdf2zh.utils import count_tokens

logger = logging.getLogger(__name__)

# 全局常量
TPS = 60  # 通用模型每秒token处理速度
AVG_THINK_CONTENT = 450  # 推理模型平均思考内容token数


class PDFTranslationStatistics:
    """
    PDF翻译统计管理类

    管理PDF翻译过程中的所有统计信息，包括：
    1. 项目开始时的估算数据
    2. 项目运行时的实际数据  
    3. 统计报告生成
    """

    def __init__(self):
        # 项目开始时的估算数据
        self.pre_stats = {
            "page_count": 0,
            "total_paragraph_count": 0,
            "total_paragraph_tokens": 0,
            "total_table_cell_count": 0,
            "total_table_tokens": 0,
            "estimated_input_tokens": 0,
            "estimated_output_tokens": 0,
            "estimated_total_tokens": 0,
            "estimated_time_seconds": 0,
            "pages": {},  # 每页详细信息
        }

        # 项目运行时的实际数据
        self.runtime_stats = {
            "start_time": 0,
            "end_time": 0,
            "translation_start_time": 0,
            "translation_end_time": 0,
            "total_time_seconds": 0,
            "translation_time_seconds": 0,
            # LLM调用统计
            "llm_call_count": 0,
            "actual_input_tokens": 0,
            "actual_output_tokens": 0,
            "actual_total_tokens": 0,
            # 段落翻译统计
            "paragraph_total": 0,
            "paragraph_translated": 0,
            "paragraph_skipped_empty": 0,
            "paragraph_skipped_formula": 0,
            "paragraph_skipped_no_text": 0,
            # 表格翻译统计
            "table_cell_total": 0,
            "table_cell_translated": 0,
            "table_cell_skipped_empty": 0,
            "table_cell_skipped_no_text": 0,
        }

        # 是否为推理模式
        self.is_reasoning_mode = False
        # 文件路径信息
        self.input_files = []

    def set_pre_analysis_data(self, analysis_stats: Dict[str, Any]):
        """
        设置项目开始时的PDF分析数据

        Args:
            analysis_stats: 从analyze_pdf函数获得的分析统计数据
        """
        self.pre_stats.update({
            "page_count": analysis_stats.get("page_count", 0),
            "total_paragraph_count": analysis_stats.get("total_paragraph_count", 0),
            "total_paragraph_tokens": analysis_stats.get("total_paragraph_tokens", 0),
            "total_table_cell_count": analysis_stats.get("total_table_cell_count", 0),
            "total_table_tokens": analysis_stats.get("total_table_tokens", 0),
            "pages": analysis_stats.get("pages", {}),
        })

        # 计算估算的token使用量
        self._calculate_estimated_tokens()
        logger.debug(f"Pre-analysis data set: {self.pre_stats}")

    def _calculate_estimated_tokens(self):
        """计算估算的token使用量"""
        paragraph_count = self.pre_stats["total_paragraph_count"]
        table_cell_count = self.pre_stats["total_table_cell_count"]
        paragraph_tokens = self.pre_stats["total_paragraph_tokens"]
        table_tokens = self.pre_stats["total_table_tokens"]

        # 计算输入token（内容token + 模板token）
        input_tokens = (
            paragraph_tokens + paragraph_count * TEMPLATE_PROMPT_TOKEN_COUNT +
            table_tokens + table_cell_count * TEMPLATE_PROMPT_TOKEN_COUNT
        )

        # 计算输出token
        if self.is_reasoning_mode:
            # 推理模式：输出内容 + 思考内容
            output_tokens = (
                paragraph_tokens + paragraph_count * AVG_THINK_CONTENT +
                table_tokens + table_cell_count * AVG_THINK_CONTENT
            )
        else:
            # 默认模式：仅输出内容
            output_tokens = paragraph_tokens + table_tokens

        self.pre_stats.update({
            "estimated_input_tokens": input_tokens,
            "estimated_output_tokens": output_tokens,
            "estimated_total_tokens": input_tokens + output_tokens,
        })

    def estimate_translation_time(self, is_reasoning: bool = False):
        """
        估算翻译时间

        Args:
            is_reasoning: 是否为推理模式
        """
        self.is_reasoning_mode = is_reasoning

        # 重新计算token估算（如果推理模式发生变化）
        self._calculate_estimated_tokens()

        # 计算时间估算
        input_time = self.pre_stats["estimated_input_tokens"] / 100
        if is_reasoning:
            output_time = self.pre_stats["estimated_output_tokens"] / TPS / 2
        else:
            output_time = self.pre_stats["estimated_output_tokens"] / TPS / 3

        estimated_time = input_time + output_time
        self.pre_stats["estimated_time_seconds"] = estimated_time

        logger.debug(f"Time estimation: input={input_time:.2f}s, output={output_time:.2f}s, total={estimated_time:.2f}s")
        return estimated_time

    def start_runtime_tracking(self):
        """开始运行时统计跟踪"""
        self.runtime_stats["start_time"] = time.time()
        logger.debug("Runtime tracking started")

    def start_translation_tracking(self):
        """开始翻译阶段统计跟踪"""
        self.runtime_stats["translation_start_time"] = time.time()
        logger.debug("Translation tracking started")

    def end_translation_tracking(self):
        """结束翻译阶段统计跟踪"""
        self.runtime_stats["translation_end_time"] = time.time()
        self.runtime_stats["translation_time_seconds"] = (
            self.runtime_stats["translation_end_time"] - 
            self.runtime_stats["translation_start_time"]
        )
        logger.debug(f"Translation tracking ended: {self.runtime_stats['translation_time_seconds']:.2f}s")

    def end_runtime_tracking(self):
        """结束运行时统计跟踪"""
        self.runtime_stats["end_time"] = time.time()
        self.runtime_stats["total_time_seconds"] = (
            self.runtime_stats["end_time"] - self.runtime_stats["start_time"]
        )
        logger.debug(f"Runtime tracking ended: {self.runtime_stats['total_time_seconds']:.2f}s")

    def update_token_stats(self, token_stats: Dict[str, int]):
        """
        更新token统计信息

        Args:
            token_stats: 包含token使用统计的字典
        """
        self.runtime_stats.update({
            "llm_call_count": token_stats.get("translation_count", 0),
            "actual_input_tokens": token_stats.get("prompt_tokens", 0),
            "actual_output_tokens": token_stats.get("completion_tokens", 0),
            "actual_total_tokens": token_stats.get("total_tokens", 0),
        })
        logger.debug(f"Token stats updated: {token_stats}")

    def update_paragraph_stats(self, paragraph_stats: Dict[str, int]):
        """
        更新段落统计信息

        Args:
            paragraph_stats: 包含段落翻译统计的字典
        """
        self.runtime_stats.update({
            "paragraph_total": paragraph_stats.get("total_paragraphs", 0),
            "paragraph_translated": paragraph_stats.get("translated", 0),
            "paragraph_skipped_empty": paragraph_stats.get("skipped_empty", 0),
            "paragraph_skipped_formula": paragraph_stats.get("skipped_formula", 0),
            "paragraph_skipped_no_text": paragraph_stats.get("skipped_no_text", 0),
        })
        logger.debug(f"Paragraph stats updated: {paragraph_stats}")

    def update_table_stats(self, table_stats: Dict[str, int]):
        """
        更新表格统计信息

        Args:
            table_stats: 包含表格翻译统计的字典
        """
        self.runtime_stats.update({
            "table_cell_total": table_stats.get("total_cells", 0),
            "table_cell_translated": table_stats.get("translated", 0),
            "table_cell_skipped_empty": table_stats.get("skipped_empty", 0),
            "table_cell_skipped_no_text": table_stats.get("skipped_no_text", 0),
        })
        logger.debug(f"Table stats updated: {table_stats}")

    def set_input_files(self, files: list[str]):
        """设置输入文件列表"""
        self.input_files = files

    def get_estimation_summary(self) -> Dict[str, Any]:
        """获取估算数据摘要"""
        return {
            "content_stats": {
                "pages": self.pre_stats["page_count"],
                "paragraphs": self.pre_stats["total_paragraph_count"],
                "paragraph_tokens": self.pre_stats["total_paragraph_tokens"],
                "table_cells": self.pre_stats["total_table_cell_count"],
                "table_tokens": self.pre_stats["total_table_tokens"],
            },
            "token_estimation": {
                "input_tokens": self.pre_stats["estimated_input_tokens"],
                "output_tokens": self.pre_stats["estimated_output_tokens"],
                "total_tokens": self.pre_stats["estimated_total_tokens"],
            },
            "time_estimation": {
                "estimated_seconds": self.pre_stats["estimated_time_seconds"],
                "is_reasoning_mode": self.is_reasoning_mode,
            }
        }

    def get_runtime_summary(self) -> Dict[str, Any]:
        """获取运行时数据摘要"""
        return {
            "time_stats": {
                "total_time": self.runtime_stats["total_time_seconds"],
                "translation_time": self.runtime_stats["translation_time_seconds"],
            },
            "llm_stats": {
                "call_count": self.runtime_stats["llm_call_count"],
                "input_tokens": self.runtime_stats["actual_input_tokens"],
                "output_tokens": self.runtime_stats["actual_output_tokens"],
                "total_tokens": self.runtime_stats["actual_total_tokens"],
            },
            "translation_stats": {
                "paragraphs": {
                    "total": self.runtime_stats["paragraph_total"],
                    "translated": self.runtime_stats["paragraph_translated"],
                    "skipped_empty": self.runtime_stats["paragraph_skipped_empty"],
                    "skipped_formula": self.runtime_stats["paragraph_skipped_formula"],
                    "skipped_no_text": self.runtime_stats["paragraph_skipped_no_text"],
                },
                "table_cells": {
                    "total": self.runtime_stats["table_cell_total"],
                    "translated": self.runtime_stats["table_cell_translated"],
                    "skipped_empty": self.runtime_stats["table_cell_skipped_empty"],
                    "skipped_no_text": self.runtime_stats["table_cell_skipped_no_text"],
                }
            }
        }

    def generate_report_log(self, output_dir: str = ".") -> str:
        """
        生成统计报告日志文件

        Args:
            output_dir: 输出目录

        Returns:
            生成的日志文件路径
        """
        # 生成文件名
        if self.input_files:
            base_filename = os.path.splitext(os.path.basename(self.input_files[0]))[0]
        else:
            base_filename = "pdf_translation"

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_filename = f"{base_filename}_{timestamp}_statistics.log"
        log_filepath = os.path.join(output_dir, log_filename)

        # 写入报告
        with open(log_filepath, 'w', encoding='utf-8') as f:
            self._write_report_content(f)

        logger.info(f"统计报告已保存到: {log_filepath}")
        return log_filepath

    def _write_report_content(self, file_handle):
        """写入报告内容到文件句柄"""
        # 写入标题
        title = " PDF Translation Statistics Report "
        separator = "=" * ((80 - len(title)) // 2)
        header = f"{separator}{title}{separator}"
        file_handle.write(header + "\n\n")

        # 写入基本信息
        file_handle.write("1. 基本信息\n")
        file_handle.write("-" * 80 + "\n")
        if self.input_files:
            file_handle.write(f"输入文件: {', '.join(self.input_files)}\n")
        file_handle.write(f"生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        file_handle.write(f"推理模式: {'是' if self.is_reasoning_mode else '否'}\n\n")

        # 写入内容统计
        file_handle.write("2. 内容统计\n")
        file_handle.write("-" * 80 + "\n")
        file_handle.write(f"总页数: {self.pre_stats['page_count']}\n")
        file_handle.write(f"总段落数: {self.pre_stats['total_paragraph_count']}\n")
        file_handle.write(f"段落token数: {self.pre_stats['total_paragraph_tokens']}\n")
        file_handle.write(f"总表格单元格数: {self.pre_stats['total_table_cell_count']}\n")
        file_handle.write(f"表格token数: {self.pre_stats['total_table_tokens']}\n")
        file_handle.write(f"总内容token数: {self.pre_stats['total_paragraph_tokens'] + self.pre_stats['total_table_tokens']}\n\n")

        # 写入页面详细信息
        if self.pre_stats["pages"]:
            file_handle.write("3. 页面详细信息\n")
            file_handle.write("-" * 80 + "\n")
            for page_num, page_data in self.pre_stats["pages"].items():
                paragraph_count = page_data.get("paragraph_count", 0)
                table_count = page_data.get("table_count", 0)
                table_cell_count = page_data.get("table_cell_count", 0)
                paragraph_tokens = page_data.get("paragraph_token_count", 0)
                table_tokens = page_data.get("table_token_count", 0)
                file_handle.write(
                    f"第{page_num + 1}页: {paragraph_count}个段落({paragraph_tokens}token) | "
                    f"{table_count}个表格({table_cell_count}个单元格, {table_tokens}token)\n"
                )
            file_handle.write("\n")

        # 写入token估算
        file_handle.write("4. Token估算\n")
        file_handle.write("-" * 80 + "\n")
        file_handle.write(f"估算输入token: {self.pre_stats['estimated_input_tokens']}\n")
        file_handle.write(f"估算输出token: {self.pre_stats['estimated_output_tokens']}\n")
        file_handle.write(f"估算总token: {self.pre_stats['estimated_total_tokens']}\n\n")

        # 写入时间估算与实际时间对比
        file_handle.write("5. 时间分析\n")
        file_handle.write("-" * 80 + "\n")
        file_handle.write(f"估算翻译时间: {self.pre_stats['estimated_time_seconds']:.2f}秒\n")
        if self.runtime_stats["total_time_seconds"] > 0:
            file_handle.write(f"实际总时间: {self.runtime_stats['total_time_seconds']:.2f}秒\n")
            if self.runtime_stats["translation_time_seconds"] > 0:
                file_handle.write(f"实际翻译时间: {self.runtime_stats['translation_time_seconds']:.2f}秒\n")

            # 计算时间偏差
            if self.pre_stats['estimated_time_seconds'] > 0:
                time_diff = self.runtime_stats["total_time_seconds"] - self.pre_stats['estimated_time_seconds']
                diff_percentage = (time_diff / self.pre_stats['estimated_time_seconds']) * 100
                if time_diff > 0:
                    file_handle.write(f"时间偏差: +{time_diff:.2f}秒 ({diff_percentage:.1f}% 超出估算)\n")
                else:
                    file_handle.write(f"时间偏差: {time_diff:.2f}秒 ({abs(diff_percentage):.1f}% 低于估算)\n")
        file_handle.write("\n")

        # 写入LLM调用统计
        if self.runtime_stats["llm_call_count"] > 0:
            file_handle.write("6. LLM调用统计\n")
            file_handle.write("-" * 80 + "\n")
            file_handle.write(f"总调用次数: {self.runtime_stats['llm_call_count']}\n")
            file_handle.write(f"实际输入token: {self.runtime_stats['actual_input_tokens']}\n")
            file_handle.write(f"实际输出token: {self.runtime_stats['actual_output_tokens']}\n")
            file_handle.write(f"实际总token: {self.runtime_stats['actual_total_tokens']}\n")
            if self.runtime_stats['llm_call_count'] > 0:
                avg_tokens = self.runtime_stats['actual_total_tokens'] / self.runtime_stats['llm_call_count']
                file_handle.write(f"平均每次调用token数: {avg_tokens:.1f}\n")

            # Token估算vs实际对比
            if self.pre_stats['estimated_total_tokens'] > 0:
                token_diff = self.runtime_stats['actual_total_tokens'] - self.pre_stats['estimated_total_tokens']
                token_diff_percentage = (token_diff / self.pre_stats['estimated_total_tokens']) * 100
                if token_diff > 0:
                    file_handle.write(f"Token偏差: +{token_diff} ({token_diff_percentage:.1f}% 超出估算)\n")
                else:
                    file_handle.write(f"Token偏差: {token_diff} ({abs(token_diff_percentage):.1f}% 低于估算)\n")
            file_handle.write("\n")

        # 写入翻译统计
        if self.runtime_stats["paragraph_total"] > 0 or self.runtime_stats["table_cell_total"] > 0:
            file_handle.write("7. 翻译统计\n")
            file_handle.write("-" * 80 + "\n")

            # 段落统计
            if self.runtime_stats["paragraph_total"] > 0:
                file_handle.write("段落翻译:\n")
                file_handle.write(f"  总段落数: {self.runtime_stats['paragraph_total']}\n")
                file_handle.write(f"  已翻译: {self.runtime_stats['paragraph_translated']}\n")
                file_handle.write(f"  跳过空白: {self.runtime_stats['paragraph_skipped_empty']}\n")
                file_handle.write(f"  跳过公式: {self.runtime_stats['paragraph_skipped_formula']}\n")
                file_handle.write(f"  跳过无中英文: {self.runtime_stats['paragraph_skipped_no_text']}\n")

            # 表格统计
            if self.runtime_stats["table_cell_total"] > 0:
                file_handle.write("表格单元格翻译:\n")
                file_handle.write(f"  总单元格数: {self.runtime_stats['table_cell_total']}\n")
                file_handle.write(f"  已翻译: {self.runtime_stats['table_cell_translated']}\n")
                file_handle.write(f"  跳过空白: {self.runtime_stats['table_cell_skipped_empty']}\n")
                file_handle.write(f"  跳过无中英文: {self.runtime_stats['table_cell_skipped_no_text']}\n")
            file_handle.write("\n")

        # 写入结尾
        file_handle.write("=" * 80 + "\n")
        file_handle.write("报告结束\n")


def perform_pre_analysis(
    pdf_bytes: bytes,
    model,
    pages: Optional[list[int]] = None,
    is_reasoning: bool = False,
    cancellation_event=None
) -> PDFTranslationStatistics:
    """
    执行项目开始时的PDF分析估算

    Args:
        pdf_bytes: PDF文件字节数据
        model: ONNX布局分析模型
        pages: 要分析的页面列表，None表示所有页面
        is_reasoning: 是否为推理模式
        cancellation_event: 取消事件

    Returns:
        PDFTranslationStatistics: 初始化的统计对象
    """
    from pdf2zh.high_level import analyze_pdf

    logger.info("开始PDF内容分析...")

    # 分析PDF内容
    analysis_stats = analyze_pdf(
        pdf_bytes=pdf_bytes,
        model=model,
        pages=pages,
        cancellation_event=cancellation_event
    )

    # 创建统计对象并设置数据
    stats = PDFTranslationStatistics()
    stats.set_pre_analysis_data(analysis_stats)

    # 估算时间
    estimated_time = stats.estimate_translation_time(is_reasoning=is_reasoning)

    # 输出估算报告
    estimation_summary = stats.get_estimation_summary()
    logger.info("=" * 20 + " 预处理分析报告 " + "=" * 20)
    logger.info(f"总页数: {estimation_summary['content_stats']['pages']}")
    logger.info(f"总段落数: {estimation_summary['content_stats']['paragraphs']}")
    logger.info(f"段落token数: {estimation_summary['content_stats']['paragraph_tokens']}")
    logger.info(f"总表格单元格数: {estimation_summary['content_stats']['table_cells']}")
    logger.info(f"表格token数: {estimation_summary['content_stats']['table_tokens']}")
    logger.info(f"估算输入token: {estimation_summary['token_estimation']['input_tokens']}")
    logger.info(f"估算输出token: {estimation_summary['token_estimation']['output_tokens']}")
    logger.info(f"估算总token: {estimation_summary['token_estimation']['total_tokens']}")
    logger.info(f"估算翻译时间: {estimated_time:.2f}秒")
    logger.info(f"推理模式: {'是' if is_reasoning else '否'}")
    logger.info("=" * 59)

    return stats


def collect_runtime_stats(
    stats_obj: PDFTranslationStatistics,
    token_stats: Dict[str, int],
    paragraph_stats: Dict[str, int],
    table_stats: Dict[str, int]
) -> PDFTranslationStatistics:
    """
    收集项目运行时的统计数据

    Args:
        stats_obj: 统计对象
        token_stats: token使用统计
        paragraph_stats: 段落翻译统计
        table_stats: 表格翻译统计

    Returns:
        PDFTranslationStatistics: 更新后的统计对象
    """
    logger.info("收集运行时统计数据...")

    # 更新各项统计
    stats_obj.update_token_stats(token_stats)
    stats_obj.update_paragraph_stats(paragraph_stats)
    stats_obj.update_table_stats(table_stats)

    # 输出运行时报告
    runtime_summary = stats_obj.get_runtime_summary()
    
    logger.info("=" * 20 + " 运行时统计报告 " + "=" * 20)
    logger.info(f"总运行时间: {runtime_summary['time_stats']['total_time']:.2f}秒")
    if runtime_summary['time_stats']['translation_time'] > 0:
        logger.info(f"翻译时间: {runtime_summary['time_stats']['translation_time']:.2f}秒")
    
    # LLM统计
    if runtime_summary['llm_stats']['call_count'] > 0:
        logger.info(f"LLM调用次数: {runtime_summary['llm_stats']['call_count']}")
        logger.info(f"实际输入token: {runtime_summary['llm_stats']['input_tokens']}")
        logger.info(f"实际输出token: {runtime_summary['llm_stats']['output_tokens']}")
        logger.info(f"实际总token: {runtime_summary['llm_stats']['total_tokens']}")

    # 段落统计
    if runtime_summary['translation_stats']['paragraphs']['total'] > 0:
        paras = runtime_summary['translation_stats']['paragraphs']
        logger.info(f"段落翻译: {paras['translated']}/{paras['total']} " +
                   f"(跳过: 空白{paras['skipped_empty']}, 公式{paras['skipped_formula']}, 无中英文{paras['skipped_no_text']})")

    # 表格统计
    if runtime_summary['translation_stats']['table_cells']['total'] > 0:
        cells = runtime_summary['translation_stats']['table_cells']
        logger.info(f"表格单元格翻译: {cells['translated']}/{cells['total']} " +
                   f"(跳过: 空白{cells['skipped_empty']}, 无中英文{cells['skipped_no_text']})")

    logger.info("=" * 59)

    return stats_obj