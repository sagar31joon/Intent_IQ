#logger.py

import os
import datetime
from core import config

class Logger:

    def __init__(self):
        os.makedirs(config.LOG_DIR, exist_ok=True)
        self.log_file = os.path.join(config.LOG_DIR, "intent_iq.log")

    def _write(self, level, msg):
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        line = f"{timestamp} [{level}] {msg}"

        print(line)

        with open(self.log_file, "a") as f:
            f.write(line + "\n")

    def info(self, msg):
        self._write("INFO", msg)

    def warn(self, msg):
        self._write("WARN", msg)

    def error(self, msg):
        self._write("ERROR", msg)

# global logger instance
log = Logger()
