from datetime import datetime
import os

def gen_logger(type_log: str = "INFO", message: str='', log_file: str = "logs/generate.log", init: bool = False):
    # Ensure log directory exists
    os.makedirs(os.path.dirname(log_file), exist_ok=True)
    
    # Initialize new run header if requested
    if init:
        with open(log_file, 'a') as f:
            f.write("\n" + "="*40 + f"\nNew Run at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n" + "="*40 + "\n")

    # Format the log entry with timestamp and log type
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    log_entry = f"{timestamp} [{type_log}] {message}\n"
    
    # Append the log entry to the file
    try:
        with open(log_file, "a") as f:
            f.write(log_entry)
    except IOError as e:
        print(f"Logging Error: {e}")