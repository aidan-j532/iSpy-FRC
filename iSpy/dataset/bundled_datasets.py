# The bundled calibration image ZIPs (from the project's GitHub release assets)
# have been removed. Use your own images for int8/uint8 quantization; the
# pipeline falls back to synthetic calibration images otherwise.
BUNDLED_DATASETS: list[dict] = []
