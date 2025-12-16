# ui/dashboard.py

import streamlit as st
import subprocess
import threading
import queue
import time
import sys
import os

# Queue to collect logs from subprocess thread
log_queue = queue.Queue()
process = None
running = False


def stream_process_output(proc):
    """Reads stdout from subprocess and pushes lines into log_queue."""
    for line in iter(proc.stdout.readline, b""):
        log_queue.put(line.decode("utf-8"))
    proc.stdout.close()


def start_intentiq():
    global process, running
    if running:
        return

    st.session_state["logs"] = ""
    running = True

    # Launch IntentIQ main engine as subprocess
    process = subprocess.Popen(
        [sys.executable, "main.py"],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        bufsize=1
    )

    # Thread to stream output
    threading.Thread(target=stream_process_output, args=(process,), daemon=True).start()


def stop_intentiq():
    global process, running
    if process and running:
        process.terminate()
        running = False


########################################
# STREAMLIT UI
########################################

st.title("🎤 IntentIQ Dashboard")
st.subheader("Real-time Intent Engine Controller")

col1, col2 = st.columns(2)

with col1:
    if st.button("▶ Start IntentIQ"):
        start_intentiq()

with col2:
    if st.button("⏹ Stop IntentIQ"):
        stop_intentiq()

st.divider()
st.write("### 📜 Engine Output:")

# Initialize logs in session state
if "logs" not in st.session_state:
    st.session_state["logs"] = ""

log_box = st.empty()

########################################
# Update logs dynamically
########################################

while True:
    try:
        line = log_queue.get_nowait()
        st.session_state["logs"] += line
        log_box.text(st.session_state["logs"])
    except queue.Empty:
        break

time.sleep(0.1)
