import json
import io
import pickle
import logging
from datetime import datetime, timedelta
from typing import Dict, Any, Optional
import os
import uuid

from flask import Flask, request, send_file, jsonify
from celery import Celery, Task
from celery.result import AsyncResult
import redis

from pdf2zh import translate_stream
from pdf2zh.high_level import (
    start_analysis_task, 
    execute_translation_only, 
    finalize_statistics_data,
    PDFTranslationStatistics
)
import tqdm
from pdf2zh.doclayout import ModelInstance, OnnxModel
from pdf2zh.config import ConfigManager
from pdf2zh.statistics import collect_runtime_stats, perform_pre_analysis

# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# 初始化布局分析模型
def ensure_model_initialized():
    """确保布局分析模型已初始化"""
    if not hasattr(ModelInstance, "value") or not ModelInstance.value:
        logger.info("正在初始化布局分析模型...")
        try:
            ModelInstance.value = OnnxModel.load_available()
            logger.info("布局分析模型初始化成功")
        except Exception as e:
            logger.error(f"布局分析模型初始化失败: {e}")
            raise

# 在启动时初始化模型
ensure_model_initialized()

# Flask应用配置
flask_app = Flask("pdf2zh")

# 创建 Celery 实例
celery_app = Celery(
    "pdf2zh",
    broker="redis://127.0.0.1:6379/0",
    backend="redis://127.0.0.1:6379/0",
)

# 更新 Celery 配置
celery_app.conf.update(
    task_serializer='json',
    accept_content=['json'],
    result_serializer='json',
    timezone='UTC',
    enable_utc=True,
    task_track_started=True,  # 启用任务开始追踪
    task_ignore_result=False,  # 不忽略任务结果
)

# 定义 Flask-Celery 任务基类
class FlaskTask(Task):
    def __call__(self, *args, **kwargs):
        with flask_app.app_context():
            return self.run(*args, **kwargs)

celery_app.Task = FlaskTask
celery_app.set_default()

# 确保任务被注册
celery_app.autodiscover_tasks(['pdf2zh.backend'])

# 将 celery_app 添加到 flask_app 的扩展中
flask_app.extensions["celery"] = celery_app

# Redis连接配置
REDIS_URL = ConfigManager.get("REDIS_URL", "redis://127.0.0.1:6379/1")
redis_client = redis.from_url(REDIS_URL, decode_responses=False)

# 会话配置
SESSION_EXPIRY_HOURS = 24  # 会话过期时间（小时）
TASK_SESSION_MAP_EXPIRY_HOURS = 48  # 任务-会话映射过期时间（小时）


class SessionManager:
    """会话管理器
    
    负责管理PDF翻译会话的生命周期，包括：
    - 创建和存储会话数据
    - 管理PDF字节流和统计对象
    - 处理会话过期
    """
    
    @staticmethod
    def store_session_data(session_id: str, pdf_bytes: bytes, stats_obj: PDFTranslationStatistics) -> None:
        """存储会话数据到Redis"""
        try:
            # 存储PDF字节流
            pdf_key = f"pdf:{session_id}"
            redis_client.setex(pdf_key, timedelta(hours=SESSION_EXPIRY_HOURS), pdf_bytes)
            
            # 存储统计对象（序列化）
            stats_key = f"stats:{session_id}"
            stats_data = pickle.dumps(stats_obj)
            redis_client.setex(stats_key, timedelta(hours=SESSION_EXPIRY_HOURS), stats_data)
            
            logger.info(f"会话数据已存储: {session_id}")
        except Exception as e:
            logger.error(f"存储会话数据失败 {session_id}: {e}")
            raise
    
    @staticmethod
    def get_pdf_data(session_id: str) -> Optional[bytes]:
        """从Redis获取PDF数据"""
        try:
            pdf_key = f"pdf:{session_id}"
            pdf_data = redis_client.get(pdf_key)
            return pdf_data
        except Exception as e:
            logger.error(f"获取PDF数据失败 {session_id}: {e}")
            return None
    
    @staticmethod
    def get_stats_object(session_id: str) -> Optional[PDFTranslationStatistics]:
        """从Redis获取统计对象"""
        try:
            stats_key = f"stats:{session_id}"
            stats_data = redis_client.get(stats_key)
            if stats_data:
                return pickle.loads(stats_data)
            return None
        except Exception as e:
            logger.error(f"获取统计对象失败 {session_id}: {e}")
            return None
    
    @staticmethod
    def link_task_to_session(task_id: str, session_id: str) -> None:
        """建立任务ID到会话ID的映射"""
        try:
            mapping_key = f"task_to_session:{task_id}"
            redis_client.setex(mapping_key, timedelta(hours=TASK_SESSION_MAP_EXPIRY_HOURS), session_id)
            logger.debug(f"任务-会话映射已建立: {task_id} -> {session_id}")
        except Exception as e:
            logger.error(f"建立任务-会话映射失败 {task_id} -> {session_id}: {e}")
            raise
    
    @staticmethod
    def get_session_from_task(task_id: str) -> Optional[str]:
        """通过任务ID获取会话ID"""
        try:
            mapping_key = f"task_to_session:{task_id}"
            session_id = redis_client.get(mapping_key)
            return session_id.decode('utf-8') if session_id else None
        except Exception as e:
            logger.error(f"获取任务对应会话失败 {task_id}: {e}")
            return None
    
    @staticmethod
    def cleanup_session(session_id: str) -> None:
        """清理会话数据"""
        try:
            pdf_key = f"pdf:{session_id}"
            stats_key = f"stats:{session_id}"
            redis_client.delete(pdf_key, stats_key)
            logger.info(f"会话数据已清理: {session_id}")
        except Exception as e:
            logger.error(f"清理会话数据失败 {session_id}: {e}")


# =============================================================================
# V2 API: 三阶段翻译管线
# =============================================================================

@flask_app.route("/v2/session/create", methods=["POST"])
def create_session_v2():
    """
    阶段一：创建会话并执行预估分析
    """
    if 'file' not in request.files:
        return jsonify({"error": "没有找到上传的文件"}), 400

    file = request.files['file']
    pdf_bytes = file.read()
    session_id = str(uuid.uuid4())
    
    try:
        # 1. 创建统计对象实例
        stats_obj = PDFTranslationStatistics()
        stats_obj.set_input_files([file.filename]) # 记录文件名
        stats_obj._session_file_size = len(pdf_bytes)  # 记录文件大小
        stats_obj.start_runtime_tracking() # 开始总计时
        
        # 2. 执行PDF内容分析
        is_reasoning = request.form.get('reasoning', 'false').lower() == 'true'
        pages_str = request.form.get('pages')
        pages = [int(p) for p in pages_str.split(',')] if pages_str else None

        # 从 pdf2zh.statistics 导入并使用 perform_pre_analysis
        # 注意：perform_pre_analysis 会返回一个已经填充好预估数据的 stats 对象
        # 我们需要将这些数据合并到我们已创建的 stats_obj 中
        
        pre_analysis_stats_obj = perform_pre_analysis(
            pdf_bytes=pdf_bytes,
            model=ModelInstance.value,
            pages=pages,
            is_reasoning=is_reasoning,
        )
        
        # 3. 合并预估数据到主统计对象
        stats_obj.pre_stats = pre_analysis_stats_obj.pre_stats
        stats_obj.is_reasoning_mode = pre_analysis_stats_obj.is_reasoning_mode

        stats_obj.end_runtime_tracking() # 结束计时(仅分析阶段)
        
        # 4. 存储会话数据
        SessionManager.store_session_data(session_id, pdf_bytes, stats_obj)

        # 5. 返回预估结果
        estimation_summary = stats_obj.get_estimation_summary()
        # 将嵌套的统计摘要"拍平"以适应V2 API的响应格式
        flattened_estimation = {
            "estimated_pages": estimation_summary["content_stats"]["pages"],
            "estimated_paragraphs": estimation_summary["content_stats"]["paragraphs"],
            "estimated_paragraph_tokens": estimation_summary["content_stats"]["paragraph_tokens"],
            "estimated_table_cells": estimation_summary["content_stats"]["table_cells"],
            "estimated_table_tokens": estimation_summary["content_stats"]["table_tokens"],
            "total_estimated_tokens": estimation_summary["token_estimation"]["total_tokens"],
            "estimated_translation_time_seconds": estimation_summary["time_estimation"]["estimated_seconds"],
        }
        
        response_data = {
            "session_id": session_id,
            "file_info": {
                "filename": file.filename,
                "size_bytes": len(pdf_bytes),
            },
            "estimation": flattened_estimation,
            "analysis_duration_seconds": stats_obj.get_total_time(),
        }
        return jsonify(response_data), 200

    except Exception as e:
        logger.error(f"创建会话失败: {e}", exc_info=True)
        return jsonify({"error": f"创建会话失败: {str(e)}"}), 500


@flask_app.route("/v2/session/<session_id>/translate", methods=["POST"])
def start_translation_task(session_id: str):
    """
    阶段二：启动异步翻译任务
    
    根据会话ID启动翻译任务，返回任务ID。
    """
    try:
        # 检查会话是否存在
        pdf_data = SessionManager.get_pdf_data(session_id)
        if pdf_data is None:
            return jsonify({"error": "会话不存在或已过期"}), 404
        
        # 获取翻译参数
        translation_params = request.get_json() or {}
        
        # 验证必要参数
        if not translation_params.get('lang_out'):
            return jsonify({"error": "缺少目标语言参数"}), 400
        if not translation_params.get('service'):
            return jsonify({"error": "缺少翻译服务参数"}), 400
        
        # 启动Celery异步翻译任务
        task = translate_v2_task.delay(session_id, translation_params)
        
        # 建立任务ID到会话ID的映射
        SessionManager.link_task_to_session(task.id, session_id)
        
        logger.info(f"翻译任务已启动: {task.id} for session {session_id}")
        
        return jsonify({
            "task_id": task.id,
            "session_id": session_id,
            "status": "started"
        }), 200
        
    except Exception as e:
        logger.error(f"启动翻译任务失败 {session_id}: {e}")
        return jsonify({"error": f"启动翻译任务失败: {str(e)}"}), 500


@flask_app.route("/v2/task/<task_id>/status", methods=["GET"])
def get_task_status(task_id: str):
    """获取异步任务状态"""
    try:
        result: AsyncResult = celery_app.AsyncResult(task_id)
        
        response_data = {
            "task_id": task_id,
            "state": str(result.state),
        }
        
        if str(result.state) == "PROGRESS":
            response_data["info"] = result.info
        elif str(result.state) == "FAILURE":
            response_data["error"] = str(result.info)
        
        return jsonify(response_data), 200
        
    except Exception as e:
        logger.error(f"获取任务状态失败 {task_id}: {e}")
        return jsonify({"error": f"获取任务状态失败: {str(e)}"}), 500


@flask_app.route("/v2/task/<task_id>/result/<artifact>", methods=["GET"])
def get_task_result(task_id: str, artifact: str):
    """
    阶段三：获取任务结果
    
    支持的artifact类型：
    - report_data: JSON格式的最终统计报告
    - mono_pdf: 单语PDF文件
    - dual_pdf: 双语PDF文件
    """
    try:
        # 检查任务状态
        result: AsyncResult = celery_app.AsyncResult(task_id)
        
        if not result.ready():
            return jsonify({"error": "任务尚未完成"}), 400
        
        if not result.successful():
            error_info = str(result.info) if result.info else "任务执行失败"
            return jsonify({"error": error_info}), 400
        
        # 获取任务结果
        task_result = result.get()
        logger.info(f"获取任务结果 {task_id}/{artifact}，结果键: {list(task_result.keys()) if isinstance(task_result, dict) else 'Not a dict'}")
        
        if artifact == "report_data":
            # 获取会话数据生成最终报告
            session_id = SessionManager.get_session_from_task(task_id)
            if not session_id:
                return jsonify({"error": "无法找到对应的会话数据"}), 404
            
            stats_obj = SessionManager.get_stats_object(session_id)
            if not stats_obj:
                return jsonify({"error": "会话统计数据不存在或已过期"}), 404
            
            # 生成最终统计报告
            raw_runtime_stats = task_result.get("raw_runtime_stats", {})
            final_report = finalize_statistics_data(stats_obj, raw_runtime_stats)
            
            # 更新session_id到报告中
            final_report["session_info"]["session_id"] = session_id
            
            return jsonify(final_report), 200
            
        elif artifact == "mono_pdf":
            # 返回单语PDF文件
            mono_pdf = task_result.get("mono_pdf")
            logger.info(f"mono_pdf数据: 存在={mono_pdf is not None}, 大小={len(mono_pdf) if mono_pdf else 0}")
            if not mono_pdf:
                logger.error(f"mono_pdf数据不存在，任务结果: {task_result}")
                return jsonify({"error": "单语PDF数据不存在"}), 404
            
            return send_file(
                io.BytesIO(mono_pdf), 
                mimetype="application/pdf",
                as_attachment=True,
                download_name=f"translated_mono_{task_id[:8]}.pdf"
            )
            
        elif artifact == "dual_pdf":
            # 返回双语PDF文件
            dual_pdf = task_result.get("dual_pdf")
            logger.info(f"dual_pdf数据: 存在={dual_pdf is not None}, 大小={len(dual_pdf) if dual_pdf else 0}")
            if not dual_pdf:
                logger.error(f"dual_pdf数据不存在，任务结果: {task_result}")
                return jsonify({"error": "双语PDF数据不存在"}), 404
            
            return send_file(
                io.BytesIO(dual_pdf), 
                mimetype="application/pdf",
                as_attachment=True,
                download_name=f"translated_dual_{task_id[:8]}.pdf"
            )
        else:
            return jsonify({"error": f"不支持的artifact类型: {artifact}"}), 400
            
    except Exception as e:
        logger.error(f"获取任务结果失败 {task_id}/{artifact}: {e}")
        return jsonify({"error": f"获取任务结果失败: {str(e)}"}), 500


@flask_app.route("/v2/task/<task_id>", methods=["DELETE"])
def cancel_v2_task(task_id: str):
    """取消V2任务"""
    try:
        result: AsyncResult = celery_app.AsyncResult(task_id)
        result.revoke(terminate=True)
        
        return jsonify({
            "task_id": task_id,
            "state": "REVOKED",
            "message": "任务已取消"
        }), 200
        
    except Exception as e:
        logger.error(f"取消任务失败 {task_id}: {e}")
        return jsonify({"error": f"取消任务失败: {str(e)}"}), 500


@celery_app.task(bind=True, name='pdf2zh.translate_v2_task')
def translate_v2_task(self: Task, session_id: str, translation_params: Dict[str, Any]):
    """
    V2异步翻译任务
    
    从会话中获取PDF数据，执行纯翻译，返回结果和统计数据。
    """
    try:
        # 确保模型已初始化（重要：在Worker进程中也需要初始化）
        ensure_model_initialized()
        
        def progress_callback(t: tqdm.tqdm):
            """进度回调函数"""
            self.update_state(
                state="PROGRESS", 
                meta={
                    "current": t.n, 
                    "total": t.total,
                    "status": f"翻译进度 {t.n}/{t.total} 页"
                }
            )
            logger.info(f"翻译进度: {t.n}/{t.total} 页")
        
        # 从Redis获取PDF数据
        pdf_data = SessionManager.get_pdf_data(session_id)
        stats_obj = SessionManager.get_stats_object(session_id)
        
        if not pdf_data or not stats_obj:
            raise ValueError("会话数据不完整或已过期")
        
        # 获取并更新统计对象的翻译开始时间
        stats_obj.start_translation_tracking()
        
        logger.info(f"开始执行翻译任务: {session_id}")
        
        # 执行阶段二：纯翻译
        translation_result = execute_translation_only(
            stream=pdf_data,
            callback=progress_callback,
            model=ModelInstance.value,
            **translation_params,
        )
        
        # 收集所有统计数据（从translation_result的顶级字典中提取）
        token_stats = translation_result.get("token_stats", {})
        paragraph_stats = translation_result.get("paragraph_stats", {})
        table_stats = translation_result.get("table_stats", {})
        
        logger.info("详细统计信息:")
        logger.info(f"Token统计: {token_stats}")
        logger.info(f"段落统计: {paragraph_stats}")
        logger.info(f"表格统计: {table_stats}")
        
        # 结束翻译计时
        stats_obj.end_translation_tracking()

        # 使用 collect_runtime_stats 函数统一处理统计数据
        stats_obj = collect_runtime_stats(stats_obj, token_stats, paragraph_stats, table_stats)
        
        # 结束总计时
        stats_obj.end_runtime_tracking()
        
        # 将更新后的统计对象存回会话
        SessionManager.store_session_data(session_id, pdf_data, stats_obj)
        
        # 准备返回给Celery的结果
        raw_runtime_stats = stats_obj.get_runtime_summary()
        
        # 生成并保存最终的统计日志文件
        log_dir = os.path.join(os.getcwd(), "outputs")
        stats_obj.generate_report_log(output_dir=log_dir)

        # 准备最终返回给客户端的数据
        final_result = {
            "mono_pdf": translation_result.get("mono_pdf"),
            "dual_pdf": translation_result.get("dual_pdf"),
            "raw_runtime_stats": raw_runtime_stats
        }

        # Celery会自动设置SUCCESS状态，不需要手动更新
        return final_result

    except Exception as e:
        logger.error(f"翻译任务失败 {session_id}: {e}", exc_info=True)
        # 确保在异常情况下也能获取部分统计信息
        if stats_obj:
            stats_obj.end_runtime_tracking()
            SessionManager.store_session_data(session_id, pdf_data, stats_obj)
        self.update_state(state='FAILURE', meta={'exc_type': type(e).__name__, 'exc_message': str(e)})
        raise e


# =============================================================================
# V1 API: 兼容性保持
# =============================================================================

@celery_app.task(bind=True, name='pdf2zh.translate_task')
def translate_task(self: Task, stream: bytes, args: dict):
    """V1兼容性任务"""
    def progress_bar(t: tqdm.tqdm):
        self.update_state(state="PROGRESS", meta={"n": t.n, "total": t.total})
        print(f"Translating {t.n} / {t.total} pages")

    doc_mono, doc_dual = translate_stream(
        stream,
        callback=progress_bar,
        model=ModelInstance.value,
        **args,
    )
    return doc_mono, doc_dual


@flask_app.route("/v1/translate", methods=["POST"])
def create_translate_tasks():
    """V1兼容性接口"""
    file = request.files["file"]
    stream = file.stream.read()
    print(request.form.get("data"))
    args = json.loads(request.form.get("data"))
    task = translate_task.delay(stream, args)
    return {"id": task.id}


@flask_app.route("/v1/translate/<id>", methods=["GET"])
def get_translate_task(id: str):
    """V1兼容性接口"""
    result: AsyncResult = celery_app.AsyncResult(id)
    if str(result.state) == "PROGRESS":
        return {"state": str(result.state), "info": result.info}
    else:
        return {"state": str(result.state)}


@flask_app.route("/v1/translate/<id>", methods=["DELETE"])
def delete_translate_task(id: str):
    """V1兼容性接口"""
    result: AsyncResult = celery_app.AsyncResult(id)
    result.revoke(terminate=True)
    return {"state": str(result.state)}


@flask_app.route("/v1/translate/<id>/<format>")
def get_translate_result(id: str, format: str):
    """V1兼容性接口"""
    result = celery_app.AsyncResult(id)
    if not result.ready():
        return {"error": "task not finished"}, 400
    if not result.successful():
        return {"error": "task failed"}, 400
    doc_mono, doc_dual = result.get()
    to_send = doc_mono if format == "mono" else doc_dual
    return send_file(io.BytesIO(to_send), "application/pdf")


# =============================================================================
# 健康检查和管理接口
# =============================================================================

@flask_app.route("/health", methods=["GET"])
def health_check():
    """健康检查接口"""
    try:
        # 检查Redis连接
        redis_client.ping()
        redis_status = "connected"
    except Exception as e:
        redis_status = f"error: {str(e)}"
    
    # 检查Celery连接
    try:
        celery_inspect = celery_app.control.inspect()
        active_tasks = celery_inspect.active()
        celery_status = "connected" if active_tasks is not None else "error"
    except Exception as e:
        celery_status = f"error: {str(e)}"
    
    return jsonify({
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "services": {
            "redis": redis_status,
            "celery": celery_status
        }
    }), 200


@flask_app.route("/v2/sessions/cleanup", methods=["POST"])
def cleanup_expired_sessions():
    """清理过期会话（管理接口）"""
    try:
        # 这里可以实现更复杂的清理逻辑
        # 目前Redis的TTL机制会自动处理过期
        return jsonify({
            "message": "会话清理完成",
            "timestamp": datetime.now().isoformat()
        }), 200
    except Exception as e:
        return jsonify({"error": f"清理失败: {str(e)}"}), 500


if __name__ == "__main__":
    flask_app.run(debug=True, port=9997)
