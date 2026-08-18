import os

# kill opencv log spam before cv2 is imported
os.environ.setdefault("OPENCV_LOG_LEVEL", "ERROR")