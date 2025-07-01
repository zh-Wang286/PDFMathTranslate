import os
import logging
import time
from pdf2zh import translate_file
from rich.logging import RichHandler

# --- Configuration ---
DEBUG_MODE = False

# --- Logging Setup ---
log_level = logging.DEBUG if DEBUG_MODE else logging.INFO
logging.basicConfig(
    level=logging.INFO,  # Keep root logger at INFO
    format="%(message)s",
    datefmt="[%X]",
    handlers=[RichHandler(rich_tracebacks=True, show_path=False)]
)
logging.getLogger("pdf2zh").setLevel(log_level)

# --- Translation Task Config ---
input_pdf = "/data1/PDFMathTranslate/files/2006-Blom-4.pdf"
output_directory = "/data1/PDFMathTranslate/translated_files"
base_filename = os.path.splitext(os.path.basename(input_pdf))[0]

# --- Simplified Test Runner ---
def run_translation():
    """Runs the translation task and returns the elapsed time."""
    print("\n--- Starting Test: Table Translation ---")
    
    output_filename = f"{base_filename}.pdf"
    output_path = os.path.join(output_directory, output_filename)
    
    start_time = time.time()
    
    try:
        translate_file(
            input_file=input_pdf,
            output_dir=output_path,
            service="xinference:qwen3",
            thread=4,
            debug=DEBUG_MODE,
            ignore_cache=True,
        )
        end_time = time.time()
        elapsed_time = end_time - start_time
        
        print("✓ Translation finished!")
        print(f"  - File saved to: {output_path}")
        print(f"  - Total time: {elapsed_time:.2f} seconds")
        return elapsed_time
    except Exception as e:
        logging.error("❌ Translation failed.", exc_info=True)
        return -1

# --- Main Execution ---
if not os.path.exists(output_directory):
    os.makedirs(output_directory)

translation_time = run_translation()

if translation_time > 0:
    print("\n--- Final Report ---")
    print(f"  - Translation Time: {translation_time:.2f} seconds")
else:
    print("\nCould not generate the final report because the test run failed.")
