# Pre-bundled quantization datasets offered on the "Bundled Datasets" page.
#
# Each entry becomes a card the user can download. On install, the release zip
# is pulled from `url` and its images are extracted flat into
# QuantizeDataset/<name>/ (img1.png, img2.jpg, ...) with a matching
# dataset.txt so it is immediately ready for quantization.
#
# To offer more datasets, add an entry here and attach the zip to a GitHub
# release - no other code changes needed.
BUNDLED_DATASETS: list[dict] = [
    {
        "name": "robotics_calibration",
        "url": "https://github.com/aidan-j532/iSpy-FRC/releases/download/RKNN_Quantization/200.Robotics.Images.zip",
        "description": "200 generic robotics calibration images. Good starting "
                       "point for int8/uint8 quantization, but for best results "
                       "use your own images of your actual game pieces and field.",
    },
    {
        "name": "validation_images",
        "url": "https://github.com/aidan-j532/iSpy-FRC/releases/download/Test_Images/valid.zip",
        "description": "Held-out validation images used by the optimized-model "
                       "accuracy comparison after conversion.",
    },
]
