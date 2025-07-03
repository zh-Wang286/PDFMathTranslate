"""
FastAPI后端服务 - PDF翻译系统

使用FastAPI重写的PDF翻译服务，支持：
- V2 API: 三阶段翻译管线（会话创建、异步翻译、结果获取）
- V1 API: 兼容性保持
- 异步任务处理（Celery）
- 会话管理（Redis）
- 健康检查和管理接口
"""

import json
import io
import pickle
import logging
import os
import uuid
from datetime import datetime, timedelta
from typing import Dict, Any, Optional, List
from contextlib import asynccontextmanager

from fastapi import FastAPI, File, UploadFile, Form, HTTPException, BackgroundTasks, Depends
from fastapi.responses import StreamingResponse, JSONResponse
from pydantic import BaseModel, Field
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


# =============================================================================
# Pydantic模型定义
# =============================================================================

class TranslationParams(BaseModel):
    """翻译参数模型"""
    lang_out: str = Field(..., description="目标语言")
    service: str = Field(..., description="翻译服务")
    lang_in: Optional[str] = Field(None, description="源语言")
    model: Optional[str] = Field(None, description="模型名称")
    reasoning: Optional[bool] = Field(False, description="是否启用推理模式")
    pages: Optional[str] = Field(None, description="页面范围，逗号分隔")


class SessionCreateResponse(BaseModel):
    """会话创建响应模型"""
    session_id: str
    file_info: Dict[str, Any]
    estimation: Dict[str, Any]
    analysis_duration_seconds: float


class TranslationTaskResponse(BaseModel):
    """翻译任务响应模型"""
    task_id: str
    session_id: str
    status: str


class TaskStatusResponse(BaseModel):
    """任务状态响应模型"""
    task_id: str
    state: str
    info: Optional[Dict[str, Any]] = None
    error: Optional[str] = None


class HealthCheckResponse(BaseModel):
    """健康检查响应模型"""
    status: str
    timestamp: str
    services: Dict[str, str]


# =============================================================================
# 模型初始化
# =============================================================================

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


# =============================================================================
# 应用生命周期管理
# =============================================================================

@asynccontextmanager
async def lifespan(app: FastAPI):
    """应用生命周期管理"""
    # 启动时的初始化
    logger.info("FastAPI PDF翻译服务启动中...")
    ensure_model_initialized()
    logger.info("FastAPI PDF翻译服务启动完成")
    
    yield
    
    # 关闭时的清理
    logger.info("FastAPI PDF翻译服务正在关闭...")
    # 这里可以添加清理逻辑
    logger.info("FastAPI PDF翻译服务已关闭")


# =============================================================================
# FastAPI应用配置
# =============================================================================

app = FastAPI(
    title="PDF2ZH Translation Service",
    description="PDF翻译服务 - 支持多阶段翻译管线和异步处理",
    version="2.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
    lifespan=lifespan
)


# =============================================================================
# Celery配置
# =============================================================================

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

# 定义 FastAPI-Celery 任务基类
class FastAPITask(Task):
    def __call__(self, *args, **kwargs):
        # FastAPI不需要应用上下文，直接运行
        return self.run(*args, **kwargs)


celery_app.Task = FastAPITask
celery_app.set_default()

# 确保任务被注册
celery_app.autodiscover_tasks(['pdf2zh.backend'])


# =============================================================================
# Redis配置
# =============================================================================

REDIS_URL = ConfigManager.get("REDIS_URL", "redis://127.0.0.1:6379/1")
redis_client = redis.from_url(REDIS_URL, decode_responses=False)

# 会话配置
SESSION_EXPIRY_HOURS = 24  # 会话过期时间（小时）
TASK_SESSION_MAP_EXPIRY_HOURS = 48  # 任务-会话映射过期时间（小时）


# =============================================================================
# 会话管理器
# =============================================================================

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

@app.post("/v2/session/create", response_model=SessionCreateResponse)
async def create_session_v2(
    file: UploadFile = File(..., description="PDF文件"),
    reasoning: str = Form("false", description="是否启用推理模式"),
    pages: Optional[str] = Form(None, description="页面范围，逗号分隔")
):
    """
    阶段一：创建会话并执行预估分析
    """
    if not file.filename:
        raise HTTPException(status_code=400, detail="没有找到上传的文件")
    
    # 读取文件内容
    pdf_bytes = await file.read()
    session_id = str(uuid.uuid4())
    
    try:
        # 1. 创建统计对象实例
        stats_obj = PDFTranslationStatistics()
        stats_obj.set_input_files([file.filename])  # 记录文件名
        stats_obj._session_file_size = len(pdf_bytes)  # 记录文件大小
        stats_obj.start_runtime_tracking()  # 开始总计时
        
        # 2. 执行PDF内容分析
        is_reasoning = reasoning.lower() == 'true'
        pages_list = [int(p) for p in pages.split(',')] if pages else None

        # 从 pdf2zh.statistics 导入并使用 perform_pre_analysis
        pre_analysis_stats_obj = perform_pre_analysis(
            pdf_bytes=pdf_bytes,
            model=ModelInstance.value,
            pages=pages_list,
            is_reasoning=is_reasoning,
        )
        
        # 3. 合并预估数据到主统计对象
        stats_obj.pre_stats = pre_analysis_stats_obj.pre_stats
        stats_obj.is_reasoning_mode = pre_analysis_stats_obj.is_reasoning_mode

        stats_obj.end_runtime_tracking()  # 结束计时(仅分析阶段)
        
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
        
        return SessionCreateResponse(
            session_id=session_id,
            file_info={
                "filename": file.filename,
                "size_bytes": len(pdf_bytes),
            },
            estimation=flattened_estimation,
            analysis_duration_seconds=stats_obj.get_total_time()
        )

    except Exception as e:
        logger.error(f"创建会话失败: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"创建会话失败: {str(e)}")


@app.post("/v2/session/{session_id}/translate", response_model=TranslationTaskResponse)
async def start_translation_task(session_id: str, translation_params: TranslationParams):
    """
    阶段二：启动异步翻译任务
    
    根据会话ID启动翻译任务，返回任务ID。
    """
    try:
        # 检查会话是否存在
        pdf_data = SessionManager.get_pdf_data(session_id)
        if pdf_data is None:
            raise HTTPException(status_code=404, detail="会话不存在或已过期")
        
        # 转换参数为字典
        params_dict = translation_params.dict()
        
        # 启动Celery异步翻译任务
        task = translate_v2_task.delay(session_id, params_dict)
        
        # 建立任务ID到会话ID的映射
        SessionManager.link_task_to_session(task.id, session_id)
        
        logger.info(f"翻译任务已启动: {task.id} for session {session_id}")
        
        return TranslationTaskResponse(
            task_id=task.id,
            session_id=session_id,
            status="started"
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"启动翻译任务失败 {session_id}: {e}")
        raise HTTPException(status_code=500, detail=f"启动翻译任务失败: {str(e)}")


@app.get("/v2/task/{task_id}/status", response_model=TaskStatusResponse)
async def get_task_status(task_id: str):
    """获取异步任务状态"""
    try:
        result: AsyncResult = celery_app.AsyncResult(task_id)
        
        response_data = TaskStatusResponse(
            task_id=task_id,
            state=str(result.state)
        )
        
        if str(result.state) == "PROGRESS":
            response_data.info = result.info
        elif str(result.state) == "FAILURE":
            response_data.error = str(result.info)
        
        return response_data
        
    except Exception as e:
        logger.error(f"获取任务状态失败 {task_id}: {e}")
        raise HTTPException(status_code=500, detail=f"获取任务状态失败: {str(e)}")


@app.get("/v2/task/{task_id}/result/{artifact}")
async def get_task_result(task_id: str, artifact: str):
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
            raise HTTPException(status_code=400, detail="任务尚未完成")
        
        if not result.successful():
            error_info = str(result.info) if result.info else "任务执行失败"
            raise HTTPException(status_code=400, detail=error_info)
        
        # 获取任务结果
        task_result = result.get()
        logger.info(f"获取任务结果 {task_id}/{artifact}，结果键: {list(task_result.keys()) if isinstance(task_result, dict) else 'Not a dict'}")
        
        if artifact == "report_data":
            # 获取会话数据生成最终报告
            session_id = SessionManager.get_session_from_task(task_id)
            if not session_id:
                raise HTTPException(status_code=404, detail="无法找到对应的会话数据")
            
            stats_obj = SessionManager.get_stats_object(session_id)
            if not stats_obj:
                raise HTTPException(status_code=404, detail="会话统计数据不存在或已过期")
            
            # 生成最终统计报告
            raw_runtime_stats = task_result.get("raw_runtime_stats", {})
            final_report = finalize_statistics_data(stats_obj, raw_runtime_stats)
            
            # 更新session_id到报告中
            final_report["session_info"]["session_id"] = session_id
            
            return JSONResponse(content=final_report)
            
        elif artifact == "mono_pdf":
            # 返回单语PDF文件
            mono_pdf = task_result.get("mono_pdf")
            logger.info(f"mono_pdf数据: 存在={mono_pdf is not None}, 大小={len(mono_pdf) if mono_pdf else 0}")
            if not mono_pdf:
                logger.error(f"mono_pdf数据不存在，任务结果: {task_result}")
                raise HTTPException(status_code=404, detail="单语PDF数据不存在")
            
            return StreamingResponse(
                io.BytesIO(mono_pdf),
                media_type="application/pdf",
                headers={"Content-Disposition": f"attachment; filename=translated_mono_{task_id[:8]}.pdf"}
            )
            
        elif artifact == "dual_pdf":
            # 返回双语PDF文件
            dual_pdf = task_result.get("dual_pdf")
            logger.info(f"dual_pdf数据: 存在={dual_pdf is not None}, 大小={len(dual_pdf) if dual_pdf else 0}")
            if not dual_pdf:
                logger.error(f"dual_pdf数据不存在，任务结果: {task_result}")
                raise HTTPException(status_code=404, detail="双语PDF数据不存在")
            
            return StreamingResponse(
                io.BytesIO(dual_pdf),
                media_type="application/pdf",
                headers={"Content-Disposition": f"attachment; filename=translated_dual_{task_id[:8]}.pdf"}
            )
        else:
            raise HTTPException(status_code=400, detail=f"不支持的artifact类型: {artifact}")
            
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"获取任务结果失败 {task_id}/{artifact}: {e}")
        raise HTTPException(status_code=500, detail=f"获取任务结果失败: {str(e)}")


@app.delete("/v2/task/{task_id}")
async def cancel_v2_task(task_id: str):
    """取消V2任务"""
    try:
        result: AsyncResult = celery_app.AsyncResult(task_id)
        result.revoke(terminate=True)
        
        return {
            "task_id": task_id,
            "state": "REVOKED",
            "message": "任务已取消"
        }
        
    except Exception as e:
        logger.error(f"取消任务失败 {task_id}: {e}")
        raise HTTPException(status_code=500, detail=f"取消任务失败: {str(e)}")


# =============================================================================
# Celery任务定义
# =============================================================================

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
        
        # 移除可能与布局模型参数冲突的 'model' 键
        # 这是因为 execute_translation_only 已经接收了 model=ModelInstance.value
        translation_params.pop('model', None)
        
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
        if 'stats_obj' in locals() and stats_obj:
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


@app.post("/v1/translate")
async def create_translate_tasks(
    file: UploadFile = File(...),
    data: str = Form(...)
):
    """V1兼容性接口"""
    stream = await file.read()
    args = json.loads(data)
    task = translate_task.delay(stream, args)
    return {"id": task.id}


@app.get("/v1/translate/{id}")
async def get_translate_task(id: str):
    """V1兼容性接口"""
    result: AsyncResult = celery_app.AsyncResult(id)
    if str(result.state) == "PROGRESS":
        return {"state": str(result.state), "info": result.info}
    else:
        return {"state": str(result.state)}


@app.delete("/v1/translate/{id}")
async def delete_translate_task(id: str):
    """V1兼容性接口"""
    result: AsyncResult = celery_app.AsyncResult(id)
    result.revoke(terminate=True)
    return {"state": str(result.state)}


@app.get("/v1/translate/{id}/{format}")
async def get_translate_result(id: str, format: str):
    """V1兼容性接口"""
    result = celery_app.AsyncResult(id)
    if not result.ready():
        raise HTTPException(status_code=400, detail="task not finished")
    if not result.successful():
        raise HTTPException(status_code=400, detail="task failed")
    doc_mono, doc_dual = result.get()
    to_send = doc_mono if format == "mono" else doc_dual
    return StreamingResponse(io.BytesIO(to_send), media_type="application/pdf")


# =============================================================================
# 健康检查和管理接口
# =============================================================================

@app.get("/health", response_model=HealthCheckResponse)
async def health_check():
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
    
    return HealthCheckResponse(
        status="healthy",
        timestamp=datetime.now().isoformat(),
        services={
            "redis": redis_status,
            "celery": celery_status
        }
    )


@app.post("/v2/sessions/cleanup")
async def cleanup_expired_sessions():
    """清理过期会话（管理接口）"""
    try:
        # 这里可以实现更复杂的清理逻辑
        # 目前Redis的TTL机制会自动处理过期
        return {
            "message": "会话清理完成",
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"清理失败: {str(e)}")


# =============================================================================
# 主程序入口
# =============================================================================

if __name__ == "__main__":
    import uvicorn
    
    uvicorn.run(
        "pdf2zh.backend_fastapi:app",
        host="0.0.0.0",
        port=9997,
        reload=True,
        log_level="info"
    ) 