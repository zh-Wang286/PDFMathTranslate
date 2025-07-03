import json
import os
import time

import requests

# --- 配置 ---
BASE_URL = "http://127.0.0.1:5000/api/v1/translate"
PDF_PATH = "files/2006-Blom-4.pdf"
RESULT_DIR = "test_results"
os.makedirs(RESULT_DIR, exist_ok=True)


def estimate_translation():
    """调用预估接口。"""
    print("--- 步骤 1: 开始时间预估 ---")
    if not os.path.exists(PDF_PATH):
        print(f"错误: 文件未找到 {PDF_PATH}")
        return None

    with open(PDF_PATH, "rb") as f:
        files = {"file": (os.path.basename(PDF_PATH), f, "application/pdf")}
        try:
            response = requests.post(f"{BASE_URL}/estimate", files=files, timeout=60)
            response.raise_for_status()
            data = response.json()
            print("预估成功:")
            print(json.dumps(data, indent=2, ensure_ascii=False))
            return data
        except requests.exceptions.RequestException as e:
            print(f"预估失败: {e}")
            if e.response:
                print("服务器返回:", e.response.text)
            return None


def start_translation_task():
    """开始一个异步翻译任务。"""
    print("\n--- 步骤 2: 创建异步翻译任务 ---")
    if not os.path.exists(PDF_PATH):
        print(f"错误: 文件未找到 {PDF_PATH}")
        return None

    # 翻译参数
    translation_args = {
        "lang_in": "en",
        "lang_out": "zh",
        "service": "azure-openai",
        "thread": 100,
        "use_concurrent_table_translation": True,
    }

    with open(PDF_PATH, "rb") as f:
        files = {"file": (os.path.basename(PDF_PATH), f, "application/pdf")}
        data = {"data": json.dumps(translation_args)}
        try:
            response = requests.post(f"{BASE_URL}/task", files=files, data=data, timeout=10)
            response.raise_for_status()
            task_info = response.json()
            print("任务创建成功:")
            print(json.dumps(task_info, indent=2))
            return task_info.get("task_id")
        except requests.exceptions.RequestException as e:
            print(f"任务创建失败: {e}")
            if e.response:
                print("服务器返回:", e.response.text)
            return None


def poll_task_status(task_id: str):
    """轮询任务状态直到完成。"""
    print(f"\n--- 步骤 3: 轮询任务 {task_id} 状态 ---")
    while True:
        try:
            response = requests.get(f"{BASE_URL}/task/{task_id}", timeout=5)
            response.raise_for_status()
            status_info = response.json()
            status = status_info.get("status")
            print(f"当前状态: {status}")

            if status == "PROGRESS":
                progress = status_info.get("progress", {})
                print(f"  -> {progress.get('message', '处理中...')}")
            elif status == "SUCCESS":
                print("任务成功完成!")
                return status_info
            elif status in ["FAILURE", "REVOKED"]:
                print(f"任务终止，状态: {status}")
                print("错误信息:", status_info.get("error", "无"))
                return None

            time.sleep(5)  # 每5秒查询一次
        except requests.exceptions.RequestException as e:
            print(f"查询状态失败: {e}")
            return None


def download_result_file(task_id: str, format: str = "dual"):
    """下载翻译结果。"""
    print(f"\n--- 步骤 4: 下载结果文件 (格式: {format}) ---")
    try:
        response = requests.get(f"{BASE_URL}/task/{task_id}/result/{format}", stream=True, timeout=30)
        response.raise_for_status()

        result_filename = os.path.join(RESULT_DIR, f"{task_id}-{format}.pdf")
        with open(result_filename, "wb") as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
        print(f"文件成功下载到: {result_filename}")
        return result_filename
    except requests.exceptions.RequestException as e:
        print(f"下载失败: {e}")
        if e.response:
            print("服务器返回:", e.response.text)
        return None


def cancel_task(task_id: str):
    """(可选) 取消任务。"""
    print(f"\n--- (可选操作): 取消任务 {task_id} ---")
    try:
        response = requests.delete(f"{BASE_URL}/task/{task_id}", timeout=5)
        response.raise_for_status()
        print("取消请求发送成功:")
        print(json.dumps(response.json(), indent=2))
    except requests.exceptions.RequestException as e:
        print(f"取消失败: {e}")


def main():
    """主测试流程。"""
    # 1. 预估
    estimate_translation()

    # 2. 创建任务
    task_id = start_translation_task()
    if not task_id:
        return

    # --- 取消任务测试 (取消下面一行的注释来测试) ---
    # time.sleep(2) # 等待任务开始处理
    # cancel_task(task_id)

    # 3. 轮询状态
    final_status = poll_task_status(task_id)

    # 4. 下载
    if final_status and final_status.get("status") == "SUCCESS":
        download_result_file(task_id, "dual")
        download_result_file(task_id, "mono")
    else:
        print("\n任务未成功，不下载文件。")


if __name__ == "__main__":
    main()
