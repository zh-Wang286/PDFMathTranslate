#!/usr/bin/env python3
"""
Celery配置测试脚本
用于验证Celery的broker和result backend配置
"""

import os
from celery import Celery
from pdf2zh.config import ConfigManager

def test_celery_config():
    print("=== Celery配置测试 ===")
    
    # 获取配置
    broker_url = ConfigManager.get("CELERY_BROKER", "redis://127.0.0.1:6379/0")
    result_backend = ConfigManager.get("CELERY_RESULT", "redis://127.0.0.1:6379/0")
    
    print(f"Broker URL: {broker_url}")
    print(f"Result Backend: {result_backend}")
    
    # 创建Celery实例
    app = Celery('test_app')
    
    # 直接设置配置
    app.conf.update(
        broker_url=broker_url,
        result_backend=result_backend,
        task_serializer='json',
        accept_content=['json'],
        result_serializer='json',
        timezone='UTC',
        enable_utc=True,
        task_track_started=True,
        task_time_limit=18000,  # 5 hours
        result_expires=86400,  # 24 hours
        worker_prefetch_multiplier=1,
        worker_max_tasks_per_child=100
    )
    
    # 测试连接
    try:
        # 测试broker连接
        conn = app.connection()
        conn.ensure_connection(max_retries=3)
        print("✅ Broker连接成功")
        conn.close()
        
        # 测试result backend
        backend = app.backend
        print(f"Result backend type: {type(backend)}")
        print(f"Result backend URL: {backend.url if hasattr(backend, 'url') else None}")
        
        if hasattr(backend, 'ping'):
            try:
                backend.ping()
                print("✅ Result backend连接成功")
            except Exception as e:
                print(f"❌ Result backend连接失败: {e}")
        
    except Exception as e:
        print(f"❌ 连接测试失败: {e}")
    
    # 打印当前环境变量
    print("\n=== 相关环境变量 ===")
    env_vars = ['CELERY_BROKER', 'CELERY_RESULT', 'REDIS_URL']
    for var in env_vars:
        value = os.environ.get(var, '未设置')
        print(f"{var}: {value}")

if __name__ == "__main__":
    test_celery_config() 