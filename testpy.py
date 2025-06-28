import os
import logging
import time
from typing import Optional, Tuple, List
from pdf2zh import translate_file
from pdf2zh.pdf2zh import (
    perform_pdf_pre_analysis,
    initialize_statistics,
    read_inputs,
    PDFTranslationStatistics
)
from pdf2zh.doclayout import OnnxModel, ModelInstance
from rich.logging import RichHandler
from rich.console import Console
from rich.table import Table as RichTable

# --- Configuration ---
# Set to True to see detailed DEBUG logs from the pdf2zh library
DEBUG_MODE = False

# --- Logging Setup ---
log_level = logging.DEBUG if DEBUG_MODE else logging.INFO
logging.basicConfig(
    level=logging.INFO, # Keep root logger at INFO
    format="%(message)s",
    datefmt="[%X]",
    handlers=[RichHandler(rich_tracebacks=True, show_path=False)]
)
# Control pdf2zh's logger level from here
logging.getLogger("pdf2zh").setLevel(log_level)


class PDFTranslationTester:
    def __init__(
        self,
        input_pdf: str,
        output_directory: str,
        debug_mode: bool = False,
        pages: Optional[List[int]] = None
    ):
        self.input_pdf = input_pdf
        self.output_directory = output_directory
        self.debug_mode = debug_mode
        self.pages = pages
        self.base_filename = os.path.splitext(os.path.basename(input_pdf))[0]
        self.console = Console()
        
        # 设置日志
        self._setup_logging()
        
        # 确保输出目录存在
        os.makedirs(output_directory, exist_ok=True)
        
        # 初始化模型
        if not hasattr(ModelInstance, "value") or not ModelInstance.value:
            ModelInstance.value = OnnxModel.load_available()
    
    def _setup_logging(self):
        """设置日志配置"""
        log_level = logging.DEBUG if self.debug_mode else logging.INFO
        logging.basicConfig(
            level=logging.INFO,
            format="%(message)s",
            datefmt="[%X]",
            handlers=[RichHandler(rich_tracebacks=True, show_path=False)]
        )
        logging.getLogger("pdf2zh").setLevel(log_level)
    
    def run_translation_test(self, concurrent_mode: bool) -> float:
        """运行翻译测试并返回耗时"""
        mode_str = "并发" if concurrent_mode else "串行"
        self.console.print(f"\n[bold cyan]--- Starting Test: Table Translation Mode [{mode_str}] ---[/]")
        
        output_filename = f"{self.base_filename}_{mode_str}.pdf"
        output_path = os.path.join(self.output_directory, output_filename)
        
        start_time = time.time()
        
        try:
            translate_file(
                input_file=self.input_pdf,
                output_dir=output_path,
                service="xinference:qwen3",
                pages=self.pages,
                thread=100,
                debug=self.debug_mode,
                ignore_cache=True,
                generate_analysis_report=True,
                use_concurrent_table_translation=concurrent_mode,
                analysis_only=True,
            )
            end_time = time.time()
            elapsed_time = end_time - start_time
            
            self.console.print(f"[green]✓ {mode_str} mode translation finished![/]")
            self.console.print(f"  - File saved to: {output_path}")
            self.console.print(f"  - Total time: {elapsed_time:.2f} seconds")
            return elapsed_time
        except Exception as e:
            self.console.print(f"[red]❌ {mode_str} mode translation failed.[/]", style="bold red")
            logging.error("Error details:", exc_info=True)
            return -1
    
    def run_performance_comparison(self):
        """运行性能对比测试"""
        # 运行并发模式测试
        concurrent_time = self.run_translation_test(concurrent_mode=True)
        
        # 运行串行模式测试
        serial_time = self.run_translation_test(concurrent_mode=False)
        
        # 生成报告
        if concurrent_time > 0 and serial_time > 0:
            self.console.print("\n[bold cyan]--- Final Performance Report ---[/]")
            self.console.print(f"  - Concurrent Mode Time: {concurrent_time:.2f} seconds")
            self.console.print(f"  - Serial Mode Time: {serial_time:.2f} seconds")
            
            if concurrent_time < serial_time:
                improvement = ((serial_time - concurrent_time) / serial_time) * 100
                self.console.print(f"\n[green]🚀 Conclusion: Concurrent mode improved performance by {improvement:.2f}%[/]")
            else:
                degradation = ((concurrent_time - serial_time) / serial_time) * 100
                self.console.print(f"\n[yellow]🤔 Conclusion: Concurrent mode was {degradation:.2f}% slower.[/]")
        else:
            self.console.print("\n[red]Could not generate a comparison report because one or both test runs failed.[/]")

    def run_direct_analysis(self) -> Optional[PDFTranslationStatistics]:
        """直接运行PDF预分析，不进行翻译"""
        self.console.print("\n[bold cyan]--- Running Direct PDF Analysis ---[/]")
        
        try:
            # 读取PDF文件
            pdf_bytes_list = read_inputs([self.input_pdf])
            if not pdf_bytes_list:
                raise ValueError("Could not read input file.")
            
            # 初始化统计对象
            stats_obj = initialize_statistics(
                needs_statistics=True,
                files=[self.input_pdf],
                service="analysis_only"
            )
            
            # 执行预分析
            stats_obj = perform_pdf_pre_analysis(
                stats_obj=stats_obj,
                files=[self.input_pdf],
                model=ModelInstance.value,
                pages=self.pages,
                service="analysis_only",
                reasoning=False
            )
            
            # 打印分析结果
            self._print_analysis_results(stats_obj)
            
            return stats_obj
            
        except Exception as e:
            self.console.print("[red]❌ PDF analysis failed.[/]", style="bold red")
            logging.error("Error details:", exc_info=True)
            return None
    
    def _print_analysis_results(self, stats: PDFTranslationStatistics):
        """打印分析结果的原始数据"""
        if not stats:
            return
            
        # 获取预分析数据
        pre_analysis_data = {
            # 基础统计
            "basic_stats": {
                "page_count": stats.pre_stats["page_count"],
                "total_paragraph_count": stats.pre_stats["total_paragraph_count"],
                "total_table_cell_count": stats.pre_stats["total_table_cell_count"],
            },
            # Token统计
            "token_stats": {
                "paragraph_tokens": stats.pre_stats["total_paragraph_tokens"],
                "table_tokens": stats.pre_stats["total_table_tokens"],
                "estimated_input_tokens": stats.pre_stats["estimated_input_tokens"],
                "estimated_output_tokens": stats.pre_stats["estimated_output_tokens"],
                "estimated_total_tokens": stats.pre_stats["estimated_total_tokens"],
            },
            # 时间估算
            "time_estimation": {
                "estimated_seconds": stats.pre_stats["estimated_time_seconds"],
                "is_reasoning_mode": stats.is_reasoning_mode,
            },
            # 页面详细信息
            "pages": stats.pre_stats["pages"]
        }
        
        # 使用rich的打印功能，但直接显示数据结构
        self.console.print("\n[bold cyan]Raw Analysis Data:[/]")
        self.console.print(pre_analysis_data)


def main():
    # 测试配置
    input_pdf = "/data1/PDFMathTranslate/files/2006-Blom-4.pdf"
    output_directory = "/data1/PDFMathTranslate/translated_files"
    debug_mode = False
    # pages_to_translate = [1, 2, 3]  # 可选的页面范围
    
    # 创建测试器实例
    tester = PDFTranslationTester(
        input_pdf=input_pdf,
        output_directory=output_directory,
        debug_mode=debug_mode,
        # pages=pages_to_translate
    )
    
    # 运行直接分析测试
    stats = tester.run_direct_analysis()
    
    # # 运行性能对比测试
    # tester.run_performance_comparison()


if __name__ == "__main__":
    main()

