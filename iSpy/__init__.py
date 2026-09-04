import os

# kill opencv log spam before cv2 is imported
os.environ["OPENCV_LOG_LEVEL"] = "ERROR"
# videoio probes every /dev/video* node on each VideoCapture() open and spams
# "ioctl(VIDIOC_QUERYCAP): Inappropriate ioctl for device" + obsensor index
# errors to stderr for harmless availability checks - silence that subsystem
os.environ["OPENCV_VIDEOIO_LOG_LEVEL"] = "SILENT"

__version__ = "0.4.0"