# PDF Translation Service (FastAPI Version)

基于 FastAPI 和 Celery 的异步 PDF 翻译服务。本服务提供了高性能的 PDF 文档翻译功能，支持异步任务处理和实时进度跟踪。

## 功能特点

- 基于 FastAPI 的现代化异步 Web API
- 使用 Celery 进行异步任务处理
- 支持 PDF 翻译时间预估
- 实时翻译进度跟踪
- 支持双语和单语翻译结果
- 任务状态管理和取消功能
- 自动文档（由 FastAPI 提供）

## 系统要求

- Python 3.8+
- Redis 服务器（用于 Celery）
- 足够的磁盘空间用于存储翻译结果

## 安装说明

1. 安装依赖包：

```bash
pip install "fastapi[all]" celery redis python-multipart
```

2. 确保 Redis 服务器正在运行：

```bash
# Ubuntu/Debian
docker run -d --name pdf-translate-redis redis:8.0
```

3. 配置存储目录：

默认的翻译文件存储目录为：
```
/data01/PDFMathTranslate/translated_files
```

请确保该目录存在并具有适当的写入权限。

## 启动服务

1. 启动 FastAPI 服务器：

```bash
# 开发模式
uvicorn pdf2zh.backend-fastapi:app --host 0.0.0.0 --port 5000 --reload

# 生产模式
uvicorn pdf2zh.backend-fastapi:app --host 0.0.0.0 --port 5000 --workers 4
```

2. 启动 Celery Worker：

```bash
# 开发模式（单进程）
celery -A pdf2zh.backend-fastapi.celery_app worker --loglevel=info -P solo

# 生产模式（多进程）
celery -A pdf2zh.backend-fastapi.celery_app worker --loglevel=info
```

## API 接口说明

### 1. 翻译时间预估

**请求**:
```http
POST /api/v1/translate/estimate
Content-Type: multipart/form-data

file: <PDF文件>
```

**响应**:
```json
{
    "task_id": "任务ID"
}
```

### 2. 获取预估结果

**请求**:
```http
GET /api/v1/translate/estimate/{task_id}
```

**响应**:
```json
{
    "task_id": "任务ID",
    "status": "SUCCESS",
    "result": {
        "estimated_time_seconds": 120,
        "total_tokens": 5000,
        "total_pages": 10
    }
}
```

### 3. 创建翻译任务

**请求**:
```http
POST /api/v1/translate/task
Content-Type: multipart/form-data

file: <PDF文件>
data: {
    "lang_in": "en",
    "lang_out": "zh",
    "service": "azure-openai",
    "thread": 100,
    "use_concurrent_table_translation": true
}
```

**响应**:
```json
{
    "task_id": "任务ID"
}
```

### 4. 获取任务状态

**请求**:
```http
GET /api/v1/translate/task/{task_id}
```

**响应**:
```json
{
    "task_id": "任务ID",
    "status": "PROGRESS",
    "progress": {
        "message": "Translating page 5/10"
    }
}
```

### 5. 取消任务

**请求**:
```http
DELETE /api/v1/translate/task/{task_id}
```

**响应**:
```json
{
    "status": "revoked",
    "task_id": "任务ID"
}
```

### 6. 下载翻译结果

**请求**:
```http
GET /api/v1/translate/task/{task_id}/result/{format}
```
其中 `format` 可以是 `mono`（单语言）或 `dual`（双语言）。

**响应**:
- 成功：返回 PDF 文件流
- 失败：返回错误信息

### 7. 查看任务列表

**请求**:
```http
GET /api/v1/translate/tasks
```

**响应**:
```json
{
    "active": [...],
    "scheduled": [...],
    "reserved": [...]
}
```

## 自动化测试

项目提供了完整的端到端测试脚本 `test-fastapi.py`，用于验证所有 API 功能：

```bash
python test-fastapi.py
```

测试脚本会执行以下步骤：
1. 上传 PDF 获取翻译时间预估
2. 创建翻译任务
3. 轮询任务状态直到完成
4. 下载翻译结果（双语和单语版本）

## API 文档

FastAPI 自动生成的交互式 API 文档可以在以下地址访问：

- Swagger UI: `http://localhost:5000/docs`
- ReDoc: `http://localhost:5000/redoc`

## 错误处理

服务使用标准的 HTTP 状态码进行错误报告：

- 400: 请求参数错误
- 404: 资源未找到
- 500: 服务器内部错误

详细的错误信息会在响应的 JSON 中提供。

## 生产环境部署建议

1. 使用 nginx 作为反向代理
2. 配置 SSL/TLS
3. 实现适当的认证机制
4. 使用环境变量管理配置
5. 设置监控和日志
6. 配置任务队列的持久化存储

## 注意事项

1. 文件上传大小限制默认为 100MB
2. 任务结果保存时间为 1 小时
3. 确保足够的磁盘空间用于存储翻译文件
4. 建议定期清理已完成的任务文件

## 开发者说明

- 代码遵循 Google Python 风格指南
- 使用 yapf 进行代码格式化
- 使用 flake8 进行代码检查
- 所有代码都包含完整的类型注解
