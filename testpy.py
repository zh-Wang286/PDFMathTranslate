import os
import logging
from pdf2zh import translate_file
from rich.logging import RichHandler

# --- 配置日志记录 ---
# `debug=True` 会将 pdf2zh 的日志级别设为 DEBUG，但我们仍需配置一个 Handler 来显示它们。
# 我们将根日志级别设为 INFO，然后只把 pdf2zh 的级别调低，这样可以避免其他库产生过多的日志。
logging.basicConfig(
    level=logging.INFO,
    format="%(message)s",
    datefmt="[%X]",
    handlers=[RichHandler(rich_tracebacks=True, show_path=False)]
)
# 为了在控制台看到 `debug=True` 带来的详细日志
logging.getLogger("pdf2zh").setLevel(logging.DEBUG)


# --- 配置您的翻译任务 ---
input_pdf = "/data1/PDFMathTranslate/files/2006-Blom-4.pdf"
output_directory = "/data1/PDFMathTranslate/translated_files"
# pages_to_translate = [1, 2, 3] # 翻译第 2-4 页

# --- 执行翻译并生成报告 ---
print(f"开始翻译文件: {os.path.basename(input_pdf)}")
translated_path, stats = translate_file(
    input_file=input_pdf,
    output_dir=output_directory,
    service="azure-openai",
    # pages=pages_to_translate,
    thread=500,
    generate_analysis_report=True, # 启用统计报告生成,
    debug=True,
    ignore_cache=True, # 添加此项以禁用缓存，获得与命令行相似的行为
)

# --- 查看结果 ---
if translated_path:
    print(f"✓ 翻译成功！文件保存在: {translated_path}")
if stats:
    print(f"✓ 已生成统计信息。")
    print(f"  - 总耗时: {stats.get_total_time():.2f} 秒")
    # 您可以访问 stats 对象的其他属性获取更多信息

# # --- 或者，只执行分析 ---
# print("\n开始仅分析模式...")
# _, analysis_stats = translate_file(
#     input_file=input_pdf,
#     analysis_only=True,
#     generate_analysis_report=True
# )

# if analysis_stats:
#     print(f"✓ 分析完成。预估翻译 token 数: {analysis_stats.get_estimated_total_tokens()}")
