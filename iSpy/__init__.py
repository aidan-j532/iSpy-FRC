import os

# kill opencv log spam before cv2 is imported. The v4l2 backend only reports
# availability failures (like QUERYCAP on a non-camera /dev/video* node, or the
# obsensor enumeration spam) through its logger - and the level it uses varies
# by OpenCV build, so 'ERROR' is NOT enough (ioctl(VIDIOC_QUERYCAP) messages
# leak through it). Errors still surface via isOpened()/return codes at the
# callers, so hard-disable the whole OpenCV logger on embedded boards.
os.environ["OPENCV_LOG_LEVEL"] = "SILENT"
os.environ["OPENCV_VIDEOIO_LOG_LEVEL"] = "SILENT"

__version__ = "0.4.0"