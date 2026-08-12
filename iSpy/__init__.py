import os

# Suppress OpenCV's own log spam (e.g. MSMF "can't grab frame" warnings
# flooding stderr at ~20/s while a camera stream is unavailable). Must be
# set before the cv2 module is first imported anywhere in the process.
os.environ.setdefault("OPENCV_LOG_LEVEL", "ERROR")