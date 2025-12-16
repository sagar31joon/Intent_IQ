# skills/get_time.py

import datetime

def run(text):
    now = datetime.datetime.now().strftime("%H:%M:%S")
    print(f"[Task] Current time is {now}")
