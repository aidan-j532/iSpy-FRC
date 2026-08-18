import cv2


def clamp(value, minimum=0.0, maximum=100.0):
    return max(minimum, min(maximum, value))


def validate_camera_frame(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    sharpness = cv2.Laplacian(gray, cv2.CV_64F).var()

    sharpness_score = clamp(
        (sharpness - 75) / (500 - 75) * 100
    )
    
    dark = (gray < 20).mean()
    bright = (gray > 235).mean()

    exposure_score = 100 - ((dark + bright) * 100)

    exposure_score = clamp(exposure_score)

    contrast = gray.std()

    contrast_score = clamp(
        (contrast - 10) / (60 - 10) * 100
    )

    overall_score = (
        sharpness_score * 0.45 +
        exposure_score * 0.30 +
        contrast_score * 0.25
    )

    overall_score = round(clamp(overall_score), 1)

    return {
        "valid": overall_score >= 70,
        "score": overall_score,

        "sharpness": round(sharpness_score, 1),
        "exposure": round(exposure_score, 1),
        "contrast": round(contrast_score, 1),

        # Raw values are useful for debugging/tuning
        "raw": {
            "sharpness": sharpness,
            "dark_pixels": dark,
            "bright_pixels": bright,
            "contrast": contrast,
        }
    }