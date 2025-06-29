from pdf2zh.config import ConfigManager

# Broker settings
broker_url = ConfigManager.get("CELERY_BROKER", "redis://127.0.0.1:6379/0")

# Result backend settings
result_backend = ConfigManager.get("CELERY_RESULT", "redis://127.0.0.1:6379/0")

# Task settings
task_serializer = 'json'
accept_content = ['json']
result_serializer = 'json'
timezone = 'UTC'
enable_utc = True

# Task result settings
task_track_started = True
task_time_limit = 18000  # 5 hours
result_expires = 86400  # 24 hours

# Worker settings
worker_prefetch_multiplier = 1
worker_max_tasks_per_child = 100 