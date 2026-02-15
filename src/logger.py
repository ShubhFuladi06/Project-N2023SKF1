## to trace the execution of the code, we can use logging instead of print statements
import logging
import os   
from datetime import datetime


LOG_FILE = f"log_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}.log"  ## to create a log file with a timestamp
LOG_path = os.path.join(os.getcwd(), "logs", LOG_FILE)  ## to specify the log file path
os.makedirs(LOG_path, exist_ok=True)  ## to create a directory for logs if it does not exist

LOG_FILE_PATH = os.path.join(LOG_path, LOG_FILE)  ## to specify the log file path


logging.basicConfig(
    filename=LOG_FILE_PATH,
    format='%(asctime)s - %(levelname)s - %(message)s',
    level=logging.INFO,
)

