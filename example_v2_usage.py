#!/usr/bin/env python3
"""
PDF2ZH V2 API 使用示例

展示新的三阶段、状态化翻译管线的完整工作流程：
1. 阶段一：创建会话与预估分析
2. 阶段二：执行异步翻译
3. 阶段三：获取结果与报告

作者：AI助手
日期：2024年
"""

import json
import time
import requests
from typing import Dict, Any, Optional


class PDF2ZHV2Client:
    """PDF2ZH V2 API客户端"""
    
    def __init__(self, base_url: str = "http://localhost:9997"):
        self.base_url = base_url
        self.session = requests.Session()
    
    def create_session(self, pdf_path: str, analysis_params: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        阶段一：创建会话与预估分析
        
        Args:
            pdf_path: PDF文件路径
            analysis_params: 分析参数
            
        Returns:
            包含session_id和预估数据的响应
        """
        print("🔍 阶段一：开始创建会话与预估分析...")
        
        url = f"{self.base_url}/v2/session/create"
        
        # 准备文件和参数
        files = {'file': open(pdf_path, 'rb')}
        data = {}
        if analysis_params:
            data['params'] = json.dumps(analysis_params)
        
        try:
            response = self.session.post(url, files=files, data=data)
            response.raise_for_status()
            
            result = response.json()
            
            print(f"✅ 会话创建成功！")
            print(f"   会话ID: {result['session_id']}")
            print(f"   文件大小: {result['file_info']['size_bytes']} 字节")
            print(f"   预估段落: {result['estimation']['estimated_paragraphs']}")
            print(f"   预估表格单元格: {result['estimation']['estimated_table_cells']}")
            print(f"   预估Token: {result['estimation']['total_estimated_tokens']}")
            print(f"   预估翻译时间: {result['estimation']['estimated_translation_time_seconds']} 秒")
            print(f"   分析耗时: {result['analysis_duration_seconds']:.2f} 秒")
            
            return result
            
        except requests.exceptions.RequestException as e:
            print(f"❌ 创建会话失败: {e}")
            raise
        finally:
            files['file'].close()
    
    def start_translation(self, session_id: str, translation_params: Dict[str, Any]) -> Dict[str, Any]:
        """
        阶段二：启动异步翻译任务
        
        Args:
            session_id: 会话ID
            translation_params: 翻译参数
            
        Returns:
            包含task_id的响应
        """
        print("🚀 阶段二：开始启动异步翻译任务...")
        
        url = f"{self.base_url}/v2/session/{session_id}/translate"
        
        try:
            response = self.session.post(url, json=translation_params)
            response.raise_for_status()
            
            result = response.json()
            
            print(f"✅ 翻译任务启动成功！")
            print(f"   任务ID: {result['task_id']}")
            print(f"   状态: {result['status']}")
            
            return result
            
        except requests.exceptions.RequestException as e:
            print(f"❌ 启动翻译任务失败: {e}")
            raise
    
    def wait_for_translation(self, task_id: str, check_interval: float = 5.0) -> bool:
        """
        等待翻译完成
        
        Args:
            task_id: 任务ID
            check_interval: 检查间隔（秒）
            
        Returns:
            是否成功完成
        """
        print("⏳ 等待翻译完成...")
        
        url = f"{self.base_url}/v2/task/{task_id}/status"
        
        while True:
            try:
                response = self.session.get(url)
                response.raise_for_status()
                
                result = response.json()
                state = result['state']
                
                if state == 'PROGRESS':
                    info = result.get('info', {})
                    current = info.get('current', 0)
                    total = info.get('total', 0)
                    status = info.get('status', '翻译中')
                    print(f"   进度: {current}/{total} - {status}")
                    
                elif state == 'SUCCESS':
                    print("✅ 翻译完成！")
                    return True
                    
                elif state == 'FAILURE':
                    error = result.get('error', '未知错误')
                    print(f"❌ 翻译失败: {error}")
                    return False
                    
                else:
                    print(f"   状态: {state}")
                
                time.sleep(check_interval)
                
            except requests.exceptions.RequestException as e:
                print(f"❌ 检查任务状态失败: {e}")
                return False
    
    def get_translation_report(self, task_id: str) -> Dict[str, Any]:
        """
        阶段三：获取翻译统计报告
        
        Args:
            task_id: 任务ID
            
        Returns:
            最终统计报告
        """
        print("📊 阶段三：获取翻译统计报告...")
        
        url = f"{self.base_url}/v2/task/{task_id}/result/report_data"
        
        try:
            response = self.session.get(url)
            response.raise_for_status()
            
            report = response.json()
            
            print("✅ 统计报告获取成功！")
            self._print_report_summary(report)
            
            return report
            
        except requests.exceptions.RequestException as e:
            print(f"❌ 获取统计报告失败: {e}")
            raise
    
    def download_pdf(self, task_id: str, pdf_type: str = "dual", output_path: Optional[str] = None) -> str:
        """
        下载翻译后的PDF文件
        
        Args:
            task_id: 任务ID
            pdf_type: PDF类型 ("mono" 或 "dual")
            output_path: 输出路径
            
        Returns:
            保存的文件路径
        """
        print(f"💾 下载{pdf_type}PDF文件...")
        
        url = f"{self.base_url}/v2/task/{task_id}/result/{pdf_type}_pdf"
        
        try:
            response = self.session.get(url)
            response.raise_for_status()
            
            # 从响应头获取文件名
            if output_path is None:
                content_disposition = response.headers.get('content-disposition', '')
                if 'filename=' in content_disposition:
                    output_path = content_disposition.split('filename=')[1].strip('"')
                else:
                    output_path = f"translated_{pdf_type}_{task_id[:8]}.pdf"
            
            with open(output_path, 'wb') as f:
                f.write(response.content)
            
            print(f"✅ PDF文件已保存: {output_path}")
            return output_path
            
        except requests.exceptions.RequestException as e:
            print(f"❌ 下载PDF文件失败: {e}")
            raise
    
    def _print_report_summary(self, report: Dict[str, Any]) -> None:
        """打印报告摘要"""
        session_info = report.get('session_info', {})
        estimation = report.get('estimation_summary', {})
        runtime = report.get('runtime_summary', {})
        comparison = report.get('comparison', {})
        performance = report.get('performance_metrics', {})
        
        print("\n" + "="*60)
        print("📋 翻译统计报告摘要")
        print("="*60)
        
        print(f"会话信息:")
        print(f"  会话ID: {session_info.get('session_id', 'N/A')}")
        print(f"  文件大小: {session_info.get('file_size_bytes', 0)} 字节")
        print(f"  处理页数: {session_info.get('pages_processed', 0)}")
        
        print(f"\n预估 vs 实际:")
        print(f"  Token预估: {estimation.get('estimated_tokens', 0)}")
        print(f"  Token实际: {runtime.get('actual_tokens', {}).get('total', 0)}")
        print(f"  准确度: {comparison.get('token_accuracy', {}).get('accuracy_level', 'N/A')}")
        
        print(f"\n翻译成果:")
        print(f"  段落总数: {runtime.get('paragraphs', {}).get('total', 0)}")
        print(f"  已翻译段落: {runtime.get('paragraphs', {}).get('translated', 0)}")
        print(f"  表格单元格总数: {runtime.get('table_cells', {}).get('total', 0)}")
        print(f"  已翻译单元格: {runtime.get('table_cells', {}).get('translated', 0)}")
        
        print(f"\n性能指标:")
        print(f"  分析耗时: {performance.get('analysis_duration_seconds', 0):.2f} 秒")
        print(f"  翻译耗时: {performance.get('translation_duration_seconds', 0):.2f} 秒")
        print(f"  翻译速度: {performance.get('tokens_per_second', 0):.2f} tokens/秒")
        
        print("="*60)


def main():
    """主函数：完整的使用示例"""
    print("🎯 PDF2ZH V2 API 使用示例")
    print("="*60)
    
    # 初始化客户端
    client = PDF2ZHV2Client()
    
    # 配置参数
    pdf_path = "files/2006-Blom-4.pdf"  # 替换为实际的PDF文件路径
    analysis_params = {
        "service": "google",
        "reasoning": False,
        "pages": None  # None表示处理所有页面
    }
    
    translation_params = {
        "lang_in": "en",
        "lang_out": "zh",
        "service": "azure-openai",
        "thread": 100,
        "use_concurrent_table_translation": True,
        "ignore_cache": True,
    }
    
    try:
        # 阶段一：创建会话与预估分析
        session_result = client.create_session(pdf_path, analysis_params)
        session_id = session_result['session_id']
        
        print(f"\n{'='*60}")
        
        # 阶段二：启动异步翻译任务
        task_result = client.start_translation(session_id, translation_params)
        task_id = task_result['task_id']
        
        print(f"\n{'='*60}")
        
        # 等待翻译完成
        if client.wait_for_translation(task_id):
            print(f"\n{'='*60}")
            
            # 阶段三：获取结果与报告
            report = client.get_translation_report(task_id)
            
            # 下载翻译文件
            mono_pdf = client.download_pdf(task_id, "mono")
            dual_pdf = client.download_pdf(task_id, "dual")
            
            print(f"\n🎉 翻译流程完成！")
            print(f"   单语PDF: {mono_pdf}")
            print(f"   双语PDF: {dual_pdf}")
            print(f"   详细报告已显示在上方")
            
        else:
            print("❌ 翻译过程中出现错误")
            
    except Exception as e:
        print(f"❌ 发生异常: {e}")


if __name__ == "__main__":
    main() 