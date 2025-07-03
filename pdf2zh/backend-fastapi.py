"""
FastAPI-based backend for the PDF translation service.

This module provides a RESTful API for estimating translation time,
submitting translation tasks, checking task status, and downloading results.
It uses Celery for asynchronous task management and ONNX for document layout analysis.
"""
from __future__ import annotations

import json
import logging
import os
from typing import Any

import tqdm
from celery import Celery, Task
from celery.result import AsyncResult
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from starlette.responses import FileResponse, JSONResponse

from pdf2zh.doclayout import OnnxModel
from pdf2zh.high_level import analyze_pdf, translate_stream_v2
from pdf2zh.statistics import PDFTranslationStatistics

# ==============================================================================
# 模块级配置和初始化
# ==============================================================================

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# --- Configuration ---
# 1. 定义文件存储目录
TRANSLATED_FILES_DIR = "/data01/PDFMathTranslate/translated_files"
os.makedirs(TRANSLATED_FILES_DIR, exist_ok=True)

# 2. 初始化 FastAPI 应用
app = FastAPI(
    title="PDF Translation Service - v2 (FastAPI)",
    description="An asynchronous API for translating PDF files.",
    version="2.0.0",
)

# 3. 配置和初始化 Celery
celery_app = Celery(
    "pdf2zh.backend-fastapi",  # 使用模块名
    broker="redis://127.0.0.1:6379/0",
    backend="redis://127.0.0.1:6379/0",
    include=["pdf2zh.backend-fastapi"]  # 包含任务的模块
)

# 4. 更新 Celery 配置
celery_app.conf.update(
    task_track_started=True,
    result_expires=3600,  # 任务结果一小时后过期
)
# 注意: FastAPI 中不再需要 Flask 的 ContextTask

# ==============================================================================
# ONNX 模型加载
# ==============================================================================
# 在应用启动时直接加载模型
logger.info("Loading ONNX model...")
onnx_model = OnnxModel.from_pretrained()
logger.info("ONNX model loaded successfully.")


# ==============================================================================
# Celery 异步任务定义
# ==============================================================================
@celery_app.task(bind=True)
def estimate_task_v2(self: Task, file_content: bytes) -> dict[str, Any]:
    """Celery task for asynchronous PDF estimation."""
    try:
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
        self.update_state(state='FAILURE', meta={'exc_type': type(e).__name__, 'exc_message': str(e)})
        raise


@celery_app.task(bind=True)
def translate_task_v2(self: Task, file_content: bytes, translation_args: dict[str, Any]) -> dict[str, Any]:
    """Celery task for asynchronous PDF translation."""
    task_id = self.request.id
    try:
        self.update_state(state="PROGRESS", meta={"message": "Translation started..."})

        def progress_callback(t: tqdm.tqdm):
            message = f"Translating page {t.n}/{t.total}"
            self.update_state(state="PROGRESS", meta={"message": message})

        result_tuple = translate_stream_v2(
            stream=file_content,
            model=onnx_model,
            callback=progress_callback,
            **translation_args,
        )
        translated_mono_stream, translated_dual_stream = result_tuple[0], result_tuple[1]

        # 保存结果文件
        mono_path = os.path.join(TRANSLATED_FILES_DIR, f"{task_id}-mono.pdf")
        dual_path = os.path.join(TRANSLATED_FILES_DIR, f"{task_id}-dual.pdf")

        with open(mono_path, "wb") as f:
            f.write(translated_mono_stream)
        with open(dual_path, "wb") as f:
            f.write(translated_dual_stream)

        return {"status": "SUCCESS", "mono_path": mono_path, "dual_path": dual_path}
    except Exception as e:
        logger.error(f"Task {task_id} failed: {e}", exc_info=True)
        self.update_state(state='FAILURE', meta={'exc_type': type(e).__name__, 'exc_message': str(e)})
        raise


# ==============================================================================
# API Endpoints (FastAPI Path Operations)
# ==============================================================================
@app.post("/api/v1/translate/estimate", status_code=202)
async def create_estimate_task(file: UploadFile = File(...)):
    """Creates an asynchronous estimation task."""
    if not file:
        raise HTTPException(status_code=400, detail="No file part")
    if file.filename == "":
        raise HTTPException(status_code=400, detail="No selected file")

    try:
        file_content = await file.read()
        task = estimate_task_v2.apply_async(args=[file_content])
        return JSONResponse(content={"task_id": task.id}, status_code=202)
    except Exception as e:
        logger.error(f"Estimation task creation failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Failed to create estimation task: {e}")


@app.get("/api/v1/translate/estimate/{task_id}")
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
    return response


@app.post("/api/v1/translate/task", status_code=202)
async def create_translation_task(file: UploadFile = File(...), data: str = Form("{}")):
    """Creates an asynchronous translation task."""
    if not file or file.filename == "":
        raise HTTPException(status_code=400, detail="A valid file is required.")

    try:
        file_content = await file.read()
        translation_args = json.loads(data)
        task = translate_task_v2.apply_async(args=[file_content, translation_args])
        return JSONResponse(content={"task_id": task.id}, status_code=202)
    except json.JSONDecodeError:
        raise HTTPException(status_code=400, detail="Invalid JSON in 'data' field.")
    except Exception as e:
        logger.error(f"Task creation failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Failed to create task: {e}")


@app.get("/api/v1/translate/task/{task_id}")
def get_task_status(task_id: str):
    """Fetches the status and result of a translation task."""
    task = AsyncResult(task_id, app=celery_app)
    response = {"task_id": task_id, "status": task.state}
    if task.state == "PROGRESS":
        response["progress"] = task.info
    elif task.state == "SUCCESS":
        response["result"] = task.result
    elif task.state == "FAILURE":
        response["error"] = str(task.info)
    return response


@app.delete("/api/v1/translate/task/{task_id}")
def cancel_task(task_id: str):
    """Cancels a running task."""
    celery_app.control.revoke(task_id, terminate=True)
    return {"status": "revoked", "task_id": task_id}


@app.get("/api/v1/translate/task/{task_id}/result/{file_format}")
def download_result(task_id: str, file_format: str):
    """Downloads the resulting translated file."""
    if file_format not in ["mono", "dual"]:
        raise HTTPException(status_code=400, detail="Invalid format specified. Use 'mono' or 'dual'.")

    task = AsyncResult(task_id, app=celery_app)
    if not task.successful():
        raise HTTPException(status_code=404, detail="Task has not completed successfully.")

    result_path_key = f"{file_format}_path"
    result_path = task.result.get(result_path_key)

    if not result_path or not os.path.exists(result_path):
        raise HTTPException(status_code=404, detail="Result file not found.")

    return FileResponse(result_path, media_type="application/pdf", filename=os.path.basename(result_path))


@app.get("/api/v1/translate/tasks")
def list_tasks():
    """
    A simplified task list feature.
    NOTE: In a production environment, this should be backed by a persistent task database.
    """
    inspector = celery_app.control.inspect()
    return {
        "active": inspector.active(),
        "scheduled": inspector.scheduled(),
        "reserved": inspector.reserved(),
    }


# ==============================================================================
# Main Execution
# ==============================================================================
if __name__ == "__main__":
    import uvicorn
    # For development, run with uvicorn
    uvicorn.run(app, host="0.0.0.0", port=5000)
