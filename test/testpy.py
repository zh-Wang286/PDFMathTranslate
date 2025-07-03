import os
import sys
import logging
import time
import io
from datetime import datetime
from typing import Optional, Tuple, List, Dict, Any

# 添加项目根目录到 Python 路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pdf2zh.high_level import translate_stream_v2
from pdf2zh.pdf2zh import (
    perform_pdf_pre_analysis,
    initialize_statistics,
    read_inputs
)
from pdf2zh.statistics import PDFTranslationStatistics
from pdf2zh.doclayout import OnnxModel, ModelInstance
from rich.logging import RichHandler
from rich.console import Console
from rich.table import Table as RichTable

# --- Logging Setup ---
logging.basicConfig(
    level=logging.INFO,  # Keep root logger at INFO
    format="%(message)s",
    datefmt="[%X]",
    handlers=[RichHandler(rich_tracebacks=True, show_path=False)]
)

def initialize_model():
    """
    初始化ONNX模型
    """
    try:
        # 根据doclayout.py中的定义，正确初始化模型
        if ModelInstance.value is None:
            ModelInstance.value = OnnxModel.from_pretrained()
        return ModelInstance.value
    except Exception as e:
        logging.error(f"模型初始化失败: {str(e)}", exc_info=True)
        return None

def generate_statistics_report(
    output_dir: str,
    input_file: str,
    stats_obj: PDFTranslationStatistics,
):
    """
    在testpy中重新实现的统计报告生成函数
    """
    # 确保输出目录存在
    os.makedirs(output_dir, exist_ok=True)
    
    # 生成文件名
    base_filename = os.path.splitext(os.path.basename(input_file))[0]
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_filename = f"{base_filename}_{timestamp}_statistics.log"
    log_filepath = os.path.join(output_dir, log_filename)
    
    logging.info(f"生成详细统计报告到: {log_filepath}")

    try:
        with open(log_filepath, 'w', encoding='utf-8') as f:
            # 写入标题
            title = " PDF Translation Statistics Report "
            separator = "=" * ((80 - len(title)) // 2)
            header = f"{separator}{title}{separator}"
            f.write(header + "\n\n")

            # 写入基本信息
            f.write("1. 基本信息\n")
            f.write("-" * 80 + "\n")
            f.write(f"输入文件: {input_file}\n")
            f.write(f"生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            
            # 写入运行环境信息
            f.write("\n运行环境:\n")
            f.write(f"  翻译服务: {stats_obj.runtime_stats['service']}\n")
            f.write(f"  线程数量: {stats_obj.runtime_stats['thread_count']}\n\n")

            # 写入内容统计
            f.write("2. 内容统计 (预估)\n")
            f.write("-" * 80 + "\n")
            f.write(f"总页数: {stats_obj.pre_stats['page_count']}\n")
            f.write(f"总段落数: {stats_obj.pre_stats['total_paragraph_count']}\n")
            f.write(f"段落token数: {stats_obj.pre_stats['total_paragraph_tokens']}\n")
            f.write(f"总表格单元格数: {stats_obj.pre_stats['total_table_cell_count']}\n")
            f.write(f"表格token数: {stats_obj.pre_stats['total_table_tokens']}\n\n")

            # 写入时间分析
            f.write("3. 时间分析\n")
            f.write("-" * 80 + "\n")
            f.write(f"实际总时间: {stats_obj.get_total_time():.2f}秒\n\n")
            
            # 写入LLM调用统计
            f.write("4. LLM调用统计 (实际)\n")
            f.write("-" * 80 + "\n")
            f.write(f"总调用次数: {stats_obj.runtime_stats['llm_call_count']}\n")
            f.write(f"实际输入token: {stats_obj.runtime_stats['actual_input_tokens']}\n")
            f.write(f"实际输出token: {stats_obj.runtime_stats['actual_output_tokens']}\n")
            f.write(f"实际总token: {stats_obj.runtime_stats['actual_total_tokens']}\n\n")
            
            # 写入翻译统计
            f.write("5. 翻译统计 (实际)\n")
            f.write("-" * 80 + "\n")
            f.write("段落翻译:\n")
            f.write(f"  总段落数: {stats_obj.runtime_stats['paragraph_total']}\n")
            f.write(f"  已翻译: {stats_obj.runtime_stats['paragraph_translated']}\n")
            f.write("表格单元格翻译:\n")
            f.write(f"  总单元格数: {stats_obj.runtime_stats['table_cell_total']}\n")
            f.write(f"  已翻译: {stats_obj.runtime_stats['table_cell_translated']}\n\n")

            # 写入结尾
            f.write("=" * 80 + "\n")
            f.write("报告结束\n")
            
    except Exception as e:
        logging.error(f"生成统计报告时发生错误: {e}", exc_info=True)


def run_translation(
    input_pdf: str,
    output_directory: str,
    service: str = "azure-openai",
    thread: int = 100,
    lang_in: str = "en",
    lang_out: str = "zh",
    debug: bool = True,
    analysis_only: bool = False,
    analysis_report: bool = False,
    use_concurrent_table_translation: bool = True,
    ignore_cache: bool = True
) -> float:
    """
    运行翻译任务并返回耗时。

    Args:
        input_pdf (str): 输入PDF文件路径
        output_directory (str): 输出目录路径
        service (str): 翻译服务提供商，默认为"azure-openai"
        thread (int): 线程数，默认为100
        lang_in (str): 输入语言，默认为"en"
        lang_out (str): 输出语言，默认为"zh"
        debug (bool): 是否启用调试模式，默认为True
        analysis_only (bool): 是否仅执行分析而不翻译，默认为False
        analysis_report (bool): 是否生成分析报告文件并打印报告, 默认为False
        use_concurrent_table_translation (bool): 是否启用并发表格翻译，默认为True
        ignore_cache (bool): 是否忽略缓存，默认为True

    Returns:
        float: 执行时间（秒），失败返回-1
    """
    print("\n--- Starting Test: PDF Translation ---")
    
    # 设置日志级别
    log_level = logging.DEBUG if debug else logging.INFO
    logging.getLogger("pdf2zh").setLevel(log_level)
    
    stats_obj: Optional[PDFTranslationStatistics] = None
    
    try:
        # 初始化模型
        model = initialize_model()
        if model is None:
            raise Exception("模型初始化失败")

        # 读取PDF文件内容
        with open(input_pdf, 'rb') as file:
            pdf_stream = file.read()

        # 1. 如果需要报告，则进行预分析
        if analysis_report:
            stats_obj = initialize_statistics(True, [input_pdf], service, thread)
            # 在这里直接调用 statistics 模块的 perform_pre_analysis
            from pdf2zh.statistics import perform_pre_analysis as pre_analysis_func
            stats_obj = pre_analysis_func(
                pdf_bytes=pdf_stream,
                model=model,
            )
            stats_obj.start_runtime_tracking() # 重新开始计时

        start_time = time.time() # 实际计时开始
        
        # 如果只需要分析
        if analysis_only:
            if not stats_obj: # 如果之前没分析，这里补上
                from pdf2zh.statistics import perform_pre_analysis as pre_analysis_func
                stats_obj = pre_analysis_func(pdf_bytes=pdf_stream, model=model)
            
            print("\n--- PDF Analysis Report (Console) ---")
            summary = stats_obj.get_estimation_summary()
            print(summary)
            return time.time() - start_time

        # 2. 执行翻译
        if stats_obj:
            stats_obj.start_translation_tracking()

        translated_pdf, bilingual_pdf, token_stats, paragraph_stats, table_stats = translate_stream_v2(
            stream=pdf_stream,
            lang_in=lang_in,
            lang_out=lang_out,
            service=service,
            thread=thread,
            ignore_cache=ignore_cache,
            use_concurrent_table_translation=use_concurrent_table_translation,
            model=model  # 传入初始化好的模型
        )
        
        if stats_obj:
            stats_obj.end_translation_tracking()

        # 保存翻译结果
        base_filename = os.path.splitext(os.path.basename(input_pdf))[0]
        translated_path = os.path.join(output_directory, f"{base_filename}_translated.pdf")
        bilingual_path = os.path.join(output_directory, f"{base_filename}_bilingual.pdf")

        # 确保输出目录存在
        os.makedirs(output_directory, exist_ok=True)

        with open(translated_path, 'wb') as f:
            f.write(translated_pdf)
        with open(bilingual_path, 'wb') as f:
            f.write(bilingual_pdf)

        end_time = time.time()
        elapsed_time = end_time - start_time
        
        print("✓ Translation finished!")
        print(f"  - Translated file saved to: {translated_path}")
        print(f"  - Bilingual file saved to: {bilingual_path}")
        print(f"  - Total time: {elapsed_time:.2f} seconds")

        # 3. 收集运行时数据并生成报告
        if stats_obj:
            stats_obj.end_runtime_tracking()
            # 更新统计
            stats_obj.update_token_stats(token_stats)
            stats_obj.update_paragraph_stats(paragraph_stats)
            stats_obj.update_table_stats(table_stats)

            # 生成报告文件
            generate_statistics_report(
                output_dir=output_directory,
                input_file=input_pdf,
                stats_obj=stats_obj,
            )
            
            # 在控制台打印简要报告
            print("\n--- Translation Statistics (Console) ---")
            print(f"Token stats: {token_stats}")
            print(f"Paragraph stats: {paragraph_stats}")
            print(f"Table stats: {table_stats}")

        return elapsed_time

    except Exception as e:
        logging.error("❌ Translation failed.", exc_info=True)
        # 如果启用了统计，即使失败也要结束计时
        if stats_obj:
            stats_obj.end_runtime_tracking()
        return -1

# --- Main Execution ---
if __name__ == "__main__":
    # 示例用法
    translation_time = run_translation(
        input_pdf="/data01/PDFMathTranslate/files/2006-Blom-4.pdf",
        output_directory="/data01/PDFMathTranslate/translated_files",
        service="azure-openai",
        thread=100,
        lang_in="en",
        lang_out="zh",
        debug=True,
        analysis_only=False,
        analysis_report=True, # 设置为True以生成报告
        use_concurrent_table_translation=True,
        ignore_cache=True
    )

    if translation_time > 0:
        print("\n--- Final Report ---")
        print(f"  - Translation Time: {translation_time:.2f} seconds")
    else:
        print("\nCould not generate the final report because the test run failed.")
