"""测试 Celery 的 results 功能

这个脚本创建一个简单的 Celery 任务并跟踪其执行结果，用于验证 Celery 的 results backend 是否正常工作。
"""

import time
import logging
from celery import Celery
from celery.result import AsyncResult

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# 创建 Celery 实例
app = Celery('test_celery_results',  # 修改应用名称与模块名一致
             include=['test_celery_results'])  # 显式包含任务模块
app.conf.update(
    broker_url='redis://127.0.0.1:6379/0',
    result_backend='redis://127.0.0.1:6379/0',
    task_serializer='json',
    accept_content=['json'],
    result_serializer='json',
    task_track_started=True,  # 启用任务开始追踪
    task_ignore_result=False,  # 不忽略任务结果
)

@app.task(bind=True, name='test_celery_results.test_task')  # 显式指定任务名称
def test_task(self, sleep_time=5):
    """测试任务，模拟一个耗时操作并返回结果"""
    logger.info(f"任务开始执行，将睡眠 {sleep_time} 秒")
    
    # 更新任务状态为进行中
    self.update_state(
        state='PROGRESS',
        meta={'current': 0, 'total': sleep_time}
    )
    
    # 模拟任务进度
    for i in range(sleep_time):
        time.sleep(1)
        self.update_state(
            state='PROGRESS',
            meta={'current': i + 1, 'total': sleep_time}
        )
        logger.info(f"任务进度: {i + 1}/{sleep_time}")
    
    result = {
        'status': 'completed',
        'execution_time': sleep_time,
        'result': f'任务执行完成，总共睡眠了 {sleep_time} 秒'
    }
    logger.info("任务执行完成")
    return result

def monitor_task(task_id):
    """监控任务执行状态"""
    result = AsyncResult(task_id, app=app)
    logger.info(f"开始监控任务: {task_id}")
    
    while not result.ready():
        if result.state == 'PROGRESS':
            progress = result.info
            logger.info(f"任务进度: {progress['current']}/{progress['total']}")
        else:
            logger.info(f"任务状态: {result.state}")
        time.sleep(1)
    
    if result.successful():
        logger.info(f"任务成功完成！结果: {result.get()}")
    else:
        logger.error(f"任务执行失败: {result.get(propagate=False)}")

if __name__ == '__main__':
    # 启动测试任务
    logger.info("启动测试任务...")
    task = test_task.delay(sleep_time=8)
    
    # 监控任务执行
    monitor_task(task.id) 