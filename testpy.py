import os
import logging
import time
from pdf2zh import translate_file
from rich.logging import RichHandler

# --- Configuration ---
# Set to True to see detailed DEBUG logs from the pdf2zh library
DEBUG_MODE = False

# --- Logging Setup ---
log_level = logging.DEBUG if DEBUG_MODE else logging.INFO
logging.basicConfig(
    level=logging.INFO, # Keep root logger at INFO
    format="%(message)s",
    datefmt="[%X]",
    handlers=[RichHandler(rich_tracebacks=True, show_path=False)]
)
# Control pdf2zh's logger level from here
logging.getLogger("pdf2zh").setLevel(log_level)


# --- Translation Task Config ---
input_pdf = "/data1/PDFMathTranslate/files/2006-Blom-4.pdf" # Using the file from the user's log
output_directory = "/data1/PDFMathTranslate/translated_files"
# pages_to_translate = [1, 2, 3] # Uncomment to translate specific pages
base_filename = os.path.splitext(os.path.basename(input_pdf))[0]


# --- Test Runner ---
def run_translation_test(concurrent_mode: bool):
    """Runs a single translation test and returns the elapsed time."""
    mode_str = "并发" if concurrent_mode else "串行"
    print(f"\n--- Starting Test: Table Translation Mode [{mode_str}] ---")
    
    output_filename = f"{base_filename}_{mode_str}.pdf"
    output_path = os.path.join(output_directory, output_filename)
    
    start_time = time.time()
    
    try:
        # Corrected function call with proper parameter names
        translate_file(
            input_file=input_pdf,  # Using correct parameter name
            output_dir=output_path,  # Using correct parameter name
            service="xinference:qwen3",
            # pages=pages_to_translate, # This was commented out in user's snippet
            thread=4,
            debug=DEBUG_MODE, # Use the flag from the top
            ignore_cache=True,
            use_concurrent_table_translation=concurrent_mode,
        )
        end_time = time.time()
        elapsed_time = end_time - start_time
        
        print(f"✓ {mode_str} mode translation finished!")
        print(f"  - File saved to: {output_path}")
        print(f"  - Total time: {elapsed_time:.2f} seconds")
        return elapsed_time
    except Exception as e:
        # Using rich handler's traceback printing
        logging.error(f"❌ {mode_str} mode translation failed.", exc_info=True)
        return -1


# --- Main Execution ---
if not os.path.exists(output_directory):
    os.makedirs(output_directory)

# Run test for concurrent mode
concurrent_time = run_translation_test(concurrent_mode=False)

# Run test for serial mode
serial_time = run_translation_test(concurrent_mode=True)

# --- Final Report ---
if concurrent_time > 0 and serial_time > 0:
    print("\n\n--- Final Performance Report ---")
    print(f"  - Concurrent Mode Time: {concurrent_time:.2f} seconds")
    print(f"  - Serial Mode Time: {serial_time:.2f} seconds")
    
    if concurrent_time < serial_time:
        improvement = ((serial_time - concurrent_time) / serial_time) * 100
        print(f"\n🚀 Conclusion: Concurrent mode improved performance by {improvement:.2f}%")
    else:
        degradation = ((concurrent_time - serial_time) / serial_time) * 100
        print(f"\n🤔 Conclusion: Concurrent mode was {degradation:.2f}% slower. This can happen if the overhead of threading outweighs the translation time for very small tables.")
else:
    print("\nCould not generate a comparison report because one or both test runs failed.")

