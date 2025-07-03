from __future__ import annotations

import io
import json
import logging
import os
import re
from typing import Any, Optional

import tqdm
from celery import Celery, Task
from celery.result import AsyncResult
from flask import Flask, jsonify, request, send_file, current_app

from pdf2zh.config import ConfigManager
from pdf2zh.doclayout import ModelInstance, OnnxModel
from pdf2zh.high_level import analyze_pdf, translate_stream_v2
from pdf2zh.statistics import PDFTranslationStatistics

# ==============================================================================
# 模块级配置和初始化
# ==============================================================================

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# --- Flask & Celery App Initialization ---
# 1. 初始化 Flask 应用
app = Flask("pdf2zh-v2")

# 2. 配置: 从用户需求中获取绝对路径
TRANSLATED_FILES_DIR = "/data01/PDFMathTranslate/translated_files"
app.config["TRANSLATED_FILES_DIR"] = TRANSLATED_FILES_DIR
os.makedirs(TRANSLATED_FILES_DIR, exist_ok=True)

# 3. 直接配置和初始化 Celery，解决 result_backend 'disabled' 问题
celery_app = Celery(
    app.name,
    broker="redis://127.0.0.1:6379/0",
    backend="redis://127.0.0.1:6379/0",
    include=["pdf2zh.backend-v2"]
)

# 4. 更新 Celery 配置
celery_app.conf.update(
    task_track_started=True,
    result_expires=3600,  # 任务结果一小时后过期
)


# 5. 创建上下文任务，确保Celery任务能访问Flask应用上下文
class ContextTask(celery_app.Task):

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        with app.app_context():
            return self.run(*args, **kwargs)


celery_app.Task = ContextTask


# ==============================================================================
# ONNX 模型和配置加载
# ==============================================================================
with app.app_context():
    if not hasattr(ModelInstance, "value") or not ModelInstance.value:
        logger.info("Loading ONNX model...")
        ModelInstance.value = OnnxModel.from_pretrained()
        logger.info("ONNX model loaded successfully.")
    onnx_model = ModelInstance.value


# ==============================================================================
# Celery 异步任务定义
# ==============================================================================
@celery_app.task(bind=True)
def estimate_task_v2(self: Task, file_content: bytes) -> dict[str, Any]:
    """Celery task for asynchronous PDF estimation."""
    try:
        # Since analysis is a single step, we don't report granular progress,
        # but we could update state to 'STARTED' if needed.
        analysis_results = analyze_pdf(file_content, model=onnx_model)
        stats = PDFTranslationStatistics()
        stats.set_pre_analysis_data(analysis_results)
        stats.estimate_translation_time()
        estimation_summary = stats.get_estimation_summary()
        return {
            "estimated_time_seconds": estimation_summary["time_estimation"]["estimated_seconds"],
            "total_tokens": estimation_summary["token_estimation"]["total_tokens"],
            "total_pages": estimation_summary["content_stats"]["pages"],
        }
    except Exception as e:
        logger.error(f"Estimation Task {self.request.id} failed: {e}", exc_info=True)
        raise


@celery_app.task(bind=True)
def translate_task_v2(self: Task, file_content: bytes, translation_args: dict[str, Any]) -> dict[str, Any]:
    """
    Celery task for asynchronous PDF translation.
    """
    task_id = self.request.id
    try:
        self.update_state(state="PROGRESS", meta={"message": "Translation started..."})

        def progress_callback(t: tqdm.tqdm):
            message = f"Translating page {t.n}/{t.total}"
            self.update_state(state="PROGRESS", meta={"message": message})

        # 调用核心翻译函数，并正确处理返回的5元组
        result_tuple = translate_stream_v2(
            stream=file_content,
            model=onnx_model,
            callback=progress_callback,
            **translation_args,
        )
        translated_mono_stream, translated_dual_stream = result_tuple[0], result_tuple[1]


        # 保存结果文件
        translated_dir = current_app.config["TRANSLATED_FILES_DIR"]
        mono_path = os.path.join(translated_dir, f"{task_id}-mono.pdf")
        dual_path = os.path.join(translated_dir, f"{task_id}-dual.pdf")

        with open(mono_path, "wb") as f:
            f.write(translated_mono_stream)
        with open(dual_path, "wb") as f:
            f.write(translated_dual_stream)

        return {"status": "SUCCESS", "mono_path": mono_path, "dual_path": dual_path}
    except Exception as e:
        logger.error(f"Task {task_id} failed: {e}", exc_info=True)
        # Re-raise the exception to allow Celery to handle the failure state.
        # This will correctly store the traceback and prevent the downstream 'KeyError: exc_type'.
        raise


# ==============================================================================
# API Endpoints (Flask Routes)
# ==============================================================================
@app.route("/api/v1/translate/estimate", methods=["POST"])
def create_estimate_task():
    """Creates an asynchronous estimation task."""
    if "file" not in request.files:
        return jsonify({"error": "No file part"}), 400
    file = request.files["file"]
    if file.filename == "":
        return jsonify({"error": "No selected file"}), 400
    if file:
        file_content = file.read()
        try:
            task = estimate_task_v2.apply_async(args=[file_content])
            return jsonify({"task_id": task.id}), 202
        except Exception as e:
            logger.error(f"Estimation task creation failed: {e}", exc_info=True)
            return jsonify({"error": f"Failed to create estimation task: {e}"}), 500
    return jsonify({"error": "Invalid file"}), 400


@app.route("/api/v1/translate/estimate/<string:task_id>", methods=["GET"])
def get_estimate_status(task_id: str):
    """Fetches the status and result of an estimation task."""
    task = AsyncResult(task_id, app=celery_app)
    response = {"task_id": task_id, "status": task.state}
    if task.state == "PROGRESS":
        response["progress"] = task.info
    elif task.state == "SUCCESS":
        response["result"] = task.result
    elif task.state == "FAILURE":
        response["error"] = str(task.info)
    return jsonify(response)


@app.route("/api/v1/translate/task", methods=["POST"])
def create_translation_task():
    """创建异步翻译任务接口。"""
    if "file" not in request.files:
        return jsonify({"error": "No file part"}), 400
    file = request.files["file"]
    if file.filename == "":
        return jsonify({"error": "No selected file"}), 400

    try:
        file_content = file.read()
        form_data = request.form.get("data", "{}")
        translation_args = json.loads(form_data)

        task = translate_task_v2.apply_async(args=[file_content, translation_args])
        return jsonify({"task_id": task.id}), 202
    except Exception as e:
        logger.error(f"Task creation failed: {e}", exc_info=True)
        return jsonify({"error": f"Failed to create task: {e}"}), 500


@app.route("/api/v1/translate/task/<string:task_id>", methods=["GET"])
def get_task_status(task_id: str):
    """查询任务状态接口。"""
    task = AsyncResult(task_id, app=celery_app)
    response = {"task_id": task_id, "status": task.state}
    if task.state == "PROGRESS":
        response["progress"] = task.info
    elif task.state == "SUCCESS":
        response["result"] = task.result
    elif task.state == "FAILURE":
        response["error"] = str(task.info)  # task.info contains the exception
    return jsonify(response)


@app.route("/api/v1/translate/task/<string:task_id>", methods=["DELETE"])
def cancel_task(task_id: str):
    """取消任务接口。"""
    celery_app.control.revoke(task_id, terminate=True)
    return jsonify({"status": "revoked", "task_id": task_id})


@app.route("/api/v1/translate/task/<string:task_id>/result/<string:format>", methods=["GET"])
def download_result(task_id: str, format: str):
    """下载结果文件接口。"""
    if format not in ["mono", "dual"]:
        return jsonify({"error": "Invalid format specified"}), 400

    task = AsyncResult(task_id, app=celery_app)
    if not task.successful():
        return jsonify({"error": "Task is not completed successfully"}), 404

    result_path = task.result.get(f"{format}_path")
    if not result_path or not os.path.exists(result_path):
        return jsonify({"error": "Result file not found"}), 404

    return send_file(result_path, as_attachment=True)

@app.route("/api/v1/translate/tasks", methods=["GET"])
def list_tasks():
    """
    一个简化的任务列表功能 (仅用于演示目的).
    注意: 在生产环境中，这应该由一个持久化的任务数据库支持。
    """
    i = celery_app.control.inspect()
    return jsonify({
        "active": i.active(),
        "scheduled": i.scheduled(),
        "reserved": i.reserved(),
    })


# ==============================================================================
# Main Execution
# ==============================================================================
if __name__ == "__main__":
    # For development:
    app.run(host="0.0.0.0", port=5000, debug=True)

    # For production (example with waitress):
    # from waitress import serve
    # serve(app, host="0.0.0.0", port=5000) 