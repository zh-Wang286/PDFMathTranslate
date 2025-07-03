"""
End-to-end test script for the FastAPI-based PDF translation service.

This script tests the full workflow:
1.  Upload a PDF to get a time/cost estimation.
2.  Submit the PDF for translation.
3.  Poll the task status until completion.
4.  Download the translated PDF files (dual-language and mono-language).
"""

import json
import os
import time

import requests

# --- 配置 ---
BASE_URL = "http://127.0.0.1:5000/api/v1/translate"
PDF_PATH = "files/2006-Blom-4.pdf"
RESULT_DIR = "test_results_fastapi"  # 使用新目录以区分
os.makedirs(RESULT_DIR, exist_ok=True)


def poll_status(task_id: str, endpoint_url: str):
    """
    Polls the status of a given task ID from a specified endpoint.

    Args:
        task_id: The ID of the task to poll.
        endpoint_url: The base URL of the task status endpoint.

    Returns:
        The final status information dictionary if the task succeeds, otherwise None.
    """
    print(f"\n--- Polling Task {task_id} ---")
    while True:
        try:
            response = requests.get(f"{endpoint_url}/{task_id}", timeout=10)
            response.raise_for_status()
            status_info = response.json()
            status = status_info.get("status")
            print(f"Current status: {status}")

            if status == "PROGRESS":
                progress = status_info.get("progress", {})
                print(f"  -> {progress.get('message', 'Processing...')}")
            elif status == "SUCCESS":
                print("Task completed successfully!")
                return status_info
            elif status in ["FAILURE", "REVOKED"]:
                print(f"Task terminated with status: {status}")
                print("Error info:", status_info.get("error", "N/A"))
                return None

            time.sleep(3)  # Poll every 3 seconds
        except requests.exceptions.RequestException as e:
            print(f"Polling failed: {e}")
            return None


def estimate_translation_async():
    """Tests the asynchronous estimation endpoint."""
    print("--- Step 1: Starting Asynchronous Time Estimation ---")
    if not os.path.exists(PDF_PATH):
        print(f"Error: File not found at {PDF_PATH}")
        return None

    with open(PDF_PATH, "rb") as f:
        files = {"file": (os.path.basename(PDF_PATH), f, "application/pdf")}
        try:
            # Create estimation task
            response = requests.post(f"{BASE_URL}/estimate", files=files, timeout=10)
            response.raise_for_status()
            task_info = response.json()
            task_id = task_info.get("task_id")
            print("Estimation task created:")
            print(json.dumps(task_info, indent=2))

            # Poll for the estimation result
            estimation_info = poll_status(task_id, f"{BASE_URL}/estimate")
            if estimation_info:
                result = estimation_info.get("result", {})
                print("\nEstimation successful:")
                print(json.dumps(result, indent=2, ensure_ascii=False))
                return result
            else:
                print("\nEstimation failed or was terminated.")
                return None

        except requests.exceptions.RequestException as e:
            print(f"Failed to create estimation task: {e}")
            if e.response:
                print("Server response:", e.response.text)
            return None


def start_translation_task():
    """Tests the asynchronous translation task creation."""
    print("\n--- Step 2: Creating Asynchronous Translation Task ---")
    if not os.path.exists(PDF_PATH):
        print(f"Error: File not found at {PDF_PATH}")
        return None

    # Translation parameters
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
            print("Task creation successful:")
            print(json.dumps(task_info, indent=2))
            return task_info.get("task_id")
        except requests.exceptions.RequestException as e:
            print(f"Task creation failed: {e}")
            if e.response:
                print("Server response:", e.response.text)
            return None


def download_result_file(task_id: str, file_format: str = "dual"):
    """Tests the result download functionality."""
    print(f"\n--- Step 4: Downloading Result File (Format: {file_format}) ---")
    try:
        url = f"{BASE_URL}/task/{task_id}/result/{file_format}"
        response = requests.get(url, stream=True, timeout=30)
        response.raise_for_status()

        result_filename = os.path.join(RESULT_DIR, f"{task_id}-{file_format}.pdf")
        with open(result_filename, "wb") as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
        print(f"File successfully downloaded to: {result_filename}")
        return result_filename
    except requests.exceptions.RequestException as e:
        print(f"Download failed: {e}")
        if e.response:
            print("Server response:", e.response.text)
        return None


def cancel_task(task_id: str):
    """(Optional) Tests task cancellation."""
    print(f"\n--- (Optional Action): Cancelling Task {task_id} ---")
    try:
        response = requests.delete(f"{BASE_URL}/task/{task_id}", timeout=5)
        response.raise_for_status()
        print("Cancellation request sent successfully:")
        print(json.dumps(response.json(), indent=2))
    except requests.exceptions.RequestException as e:
        print(f"Cancellation failed: {e}")


def main():
    """Main test workflow."""
    # 1. Get estimation
    estimation_data = estimate_translation_async()
    if not estimation_data:
        print("\nProcess stopped due to estimation failure.")
        return

    # 2. Create translation task
    task_id = start_translation_task()
    if not task_id:
        return

    # --- To test task cancellation, uncomment the following lines ---
    # print("\n--- Testing cancellation in 2 seconds ---")
    # time.sleep(2)
    # cancel_task(task_id)

    # 3. Poll for translation status
    final_status_info = poll_status(task_id, f"{BASE_URL}/task")

    # 4. Download results if successful
    if final_status_info and final_status_info.get("status") == "SUCCESS":
        download_result_file(task_id, "dual")
        download_result_file(task_id, "mono")
    else:
        print("\nTask did not succeed, skipping download.")


if __name__ == "__main__":
    main() 