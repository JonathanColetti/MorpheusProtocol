import logging
import sys
import os


import logging
import sys
import os

class NoisyLibFilter(logging.Filter):
    """
    Filters out logs from specific libraries unless they are WARNING or higher.
    """
    def __init__(self, noisy_libs):
        self.noisy_libs = noisy_libs
    
    def filter(self, record):
        # Check if the log is coming from one of the noisy libraries
        is_noisy = any(record.name.startswith(lib) for lib in self.noisy_libs)
        
        if is_noisy:
            # If it's a noisy lib, ONLY show it if it's a WARNING or ERROR
            return record.levelno >= logging.WARNING
        
        # Otherwise, allow the log
        return True

def setup_logging():
    os.makedirs("logs", exist_ok=True)
    
    # Libraries to hide from console (but keep in debug.log)
    noisy_libs = [
        "httpcore", 
        "httpx", 
        "urllib3", 
        "google.auth", 
        "google_genai", 
        "google.adk"
    ]
    
    debug_formatter = logging.Formatter(
        '%(asctime)s | %(name)s | %(levelname)s | %(filename)s:%(lineno)d | %(message)s'
    )
    standard_formatter = logging.Formatter(
        '%(asctime)s - %(levelname)s - %(message)s'
    )
    
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.DEBUG)
    
    if root_logger.hasHandlers():
        root_logger.handlers.clear()
    
    # 1. Debug File Handler (Captures EVERYTHING, including noisy libs)
    debug_file_handler = logging.FileHandler('logs/debug.log', mode='w')
    debug_file_handler.setLevel(logging.DEBUG)
    debug_file_handler.setFormatter(debug_formatter)
    root_logger.addHandler(debug_file_handler)
    
    # 2. Main App Log (Clean logs)
    main_file_handler = logging.FileHandler('logs/app.log', mode='a')
    main_file_handler.setLevel(logging.INFO)
    main_file_handler.setFormatter(standard_formatter)
    root_logger.addHandler(main_file_handler)
    
    # 3. Console Handler (Visual Output)
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(standard_formatter)
    console_handler.addFilter(NoisyLibFilter(noisy_libs))
    root_logger.addHandler(console_handler)
    
    # 4. Dialogue Log Handler (Agent dialogue and tool execution)
    dialogue_logger = logging.getLogger('dialogue')
    dialogue_logger.setLevel(logging.INFO)
    dialogue_logger.propagate = False  # Don't propagate to root logger
    
    dialogue_handler = logging.FileHandler('logs/dialogue.log', mode='a')
    dialogue_handler.setLevel(logging.INFO)
    dialogue_handler.setFormatter(logging.Formatter('%(message)s'))  # Raw JSON output
    dialogue_logger.addHandler(dialogue_handler)
    
    # 5. Error Log Handler (Errors only)
    error_logger = logging.getLogger('error')
    error_logger.setLevel(logging.ERROR)
    error_logger.propagate = False  # Don't propagate to root logger
    
    error_handler = logging.FileHandler('logs/error.log', mode='a')
    error_handler.setLevel(logging.ERROR)
    error_handler.setFormatter(debug_formatter)
    error_logger.addHandler(error_handler)
    
    # 6. Metrics Log Handler (Metrics data only)
    metrics_logger = logging.getLogger('metrics')
    metrics_logger.setLevel(logging.INFO)
    metrics_logger.propagate = False  # Don't propagate to root logger
    
    metrics_handler = logging.FileHandler('logs/metrics.log', mode='a')
    metrics_handler.setLevel(logging.INFO)
    metrics_handler.setFormatter(logging.Formatter('%(message)s'))  # Raw JSON output
    metrics_logger.addHandler(metrics_handler)
    
    # Configure the noisy libraries to emit logs at DEBUG level
    for lib_name in noisy_libs:
        lib_logger = logging.getLogger(lib_name)
        lib_logger.setLevel(logging.DEBUG)
        lib_logger.propagate = True