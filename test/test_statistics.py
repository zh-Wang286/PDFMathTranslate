#!/usr/bin/env python3
"""
PDF翻译统计系统测试脚本

测试新的统计管理系统是否能正常工作
"""

import sys
import os
import tempfile
from pathlib import Path

# 添加项目根目录到Python路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from pdf2zh.statistics import PDFTranslationStatistics, perform_pre_analysis, collect_runtime_stats
from pdf2zh.doclayout import OnnxModel


def test_statistics_class():
    """测试PDFTranslationStatistics类的基本功能"""
    print("=" * 50)
    print("测试 PDFTranslationStatistics 类")
    print("=" * 50)
    
    # 创建统计对象
    stats = PDFTranslationStatistics()
    
    # 测试设置输入文件
    stats.set_input_files(["test.pdf"])
    assert stats.input_files == ["test.pdf"]
    print("✓ 设置输入文件测试通过")
    
    # 测试设置预分析数据
    mock_analysis_data = {
        "page_count": 5,
        "total_paragraph_count": 50,
        "total_paragraph_tokens": 1000,
        "total_table_cell_count": 20,
        "total_table_tokens": 200,
        "pages": {
            0: {"paragraph_count": 10, "paragraph_token_count": 200, "table_count": 1, "table_cell_count": 4, "table_token_count": 40},
            1: {"paragraph_count": 12, "paragraph_token_count": 240, "table_count": 0, "table_cell_count": 0, "table_token_count": 0},
        }
    }
    stats.set_pre_analysis_data(mock_analysis_data)
    assert stats.pre_stats["page_count"] == 5
    assert stats.pre_stats["total_paragraph_count"] == 50
    print("✓ 设置预分析数据测试通过")
    
    # 测试时间估算
    estimated_time = stats.estimate_translation_time(is_reasoning=False)
    assert estimated_time > 0
    print(f"✓ 默认模式时间估算: {estimated_time:.2f}秒")
    
    reasoning_time = stats.estimate_translation_time(is_reasoning=True)
    assert reasoning_time > estimated_time  # 推理模式应该需要更多时间
    print(f"✓ 推理模式时间估算: {reasoning_time:.2f}秒")
    
    # 测试运行时统计跟踪
    stats.start_runtime_tracking()
    stats.start_translation_tracking()
    
    # 模拟一些延迟
    import time
    time.sleep(0.1)
    
    stats.end_translation_tracking()
    stats.end_runtime_tracking()
    
    assert stats.runtime_stats["total_time_seconds"] > 0
    assert stats.runtime_stats["translation_time_seconds"] > 0
    print("✓ 运行时统计跟踪测试通过")
    
    # 测试更新统计信息
    mock_token_stats = {
        "translation_count": 10,
        "prompt_tokens": 500,
        "completion_tokens": 300,
        "total_tokens": 800,
    }
    stats.update_token_stats(mock_token_stats)
    assert stats.runtime_stats["llm_call_count"] == 10
    print("✓ Token统计更新测试通过")
    
    mock_paragraph_stats = {
        "total_paragraphs": 50,
        "translated": 40,
        "skipped_empty": 5,
        "skipped_formula": 3,
        "skipped_no_text": 2,
    }
    stats.update_paragraph_stats(mock_paragraph_stats)
    assert stats.runtime_stats["paragraph_total"] == 50
    assert stats.runtime_stats["paragraph_translated"] == 40
    print("✓ 段落统计更新测试通过")
    
    mock_table_stats = {
        "total_cells": 20,
        "translated": 15,
        "skipped_empty": 3,
        "skipped_no_text": 2,
    }
    stats.update_table_stats(mock_table_stats)
    assert stats.runtime_stats["table_cell_total"] == 20
    assert stats.runtime_stats["table_cell_translated"] == 15
    print("✓ 表格统计更新测试通过")
    
    # 测试生成报告
    with tempfile.TemporaryDirectory() as temp_dir:
        log_file = stats.generate_report_log(temp_dir)
        assert os.path.exists(log_file)
        print(f"✓ 报告生成测试通过: {log_file}")
        
        # 检查报告内容
        with open(log_file, 'r', encoding='utf-8') as f:
            content = f.read()
            assert "PDF Translation Statistics Report" in content
            assert "总页数: 5" in content
            assert "总段落数: 50" in content
            print("✓ 报告内容验证通过")
    
    print("\n✅ PDFTranslationStatistics 类测试全部通过！\n")


def test_estimation_summary():
    """测试估算数据摘要功能"""
    print("=" * 50)
    print("测试估算数据摘要功能")
    print("=" * 50)
    
    stats = PDFTranslationStatistics()
    
    # 设置测试数据
    mock_analysis_data = {
        "page_count": 3,
        "total_paragraph_count": 30,
        "total_paragraph_tokens": 600,
        "total_table_cell_count": 10,
        "total_table_tokens": 100,
        "pages": {}
    }
    stats.set_pre_analysis_data(mock_analysis_data)
    stats.estimate_translation_time(is_reasoning=False)
    
    summary = stats.get_estimation_summary()
    
    assert summary["content_stats"]["pages"] == 3
    assert summary["content_stats"]["paragraphs"] == 30
    assert summary["token_estimation"]["input_tokens"] > 0
    assert summary["token_estimation"]["output_tokens"] > 0
    assert summary["time_estimation"]["estimated_seconds"] > 0
    
    print("✓ 估算数据摘要测试通过")
    
    # 测试推理模式
    stats.estimate_translation_time(is_reasoning=True)
    reasoning_summary = stats.get_estimation_summary()
    
    assert reasoning_summary["time_estimation"]["is_reasoning_mode"] == True
    assert reasoning_summary["token_estimation"]["output_tokens"] > summary["token_estimation"]["output_tokens"]
    
    print("✓ 推理模式估算数据摘要测试通过")
    
    print("\n✅ 估算数据摘要功能测试通过！\n")


def test_runtime_summary():
    """测试运行时数据摘要功能"""
    print("=" * 50)
    print("测试运行时数据摘要功能")
    print("=" * 50)
    
    stats = PDFTranslationStatistics()
    
    # 模拟运行时数据
    stats.start_runtime_tracking()
    import time
    time.sleep(0.05)
    stats.end_runtime_tracking()
    
    # 更新各种统计信息
    stats.update_token_stats({
        "translation_count": 5,
        "prompt_tokens": 250,
        "completion_tokens": 150,
        "total_tokens": 400,
    })
    
    stats.update_paragraph_stats({
        "total_paragraphs": 25,
        "translated": 20,
        "skipped_empty": 3,
        "skipped_formula": 1,
        "skipped_no_text": 1,
    })
    
    stats.update_table_stats({
        "total_cells": 8,
        "translated": 6,
        "skipped_empty": 1,
        "skipped_no_text": 1,
    })
    
    runtime_summary = stats.get_runtime_summary()
    
    assert runtime_summary["time_stats"]["total_time"] > 0
    assert runtime_summary["llm_stats"]["call_count"] == 5
    assert runtime_summary["llm_stats"]["total_tokens"] == 400
    assert runtime_summary["translation_stats"]["paragraphs"]["total"] == 25
    assert runtime_summary["translation_stats"]["paragraphs"]["translated"] == 20
    assert runtime_summary["translation_stats"]["table_cells"]["total"] == 8
    assert runtime_summary["translation_stats"]["table_cells"]["translated"] == 6
    
    print("✓ 运行时数据摘要测试通过")
    print("\n✅ 运行时数据摘要功能测试通过！\n")


def main():
    """运行所有测试"""
    print("开始测试PDF翻译统计系统...")
    print("=" * 80)
    
    try:
        test_statistics_class()
        test_estimation_summary()
        test_runtime_summary()
        
        print("=" * 80)
        print("🎉 所有测试通过！新的统计系统工作正常。")
        print("=" * 80)
        
        return 0
        
    except Exception as e:
        print(f"❌ 测试失败: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main()) 