import cv2
import numpy as np

def setup_camera_exposure(cap):
    """Enables auto-exposure and resets baseline brightness/contrast on hardware."""
    # CAP_PROP_AUTO_EXPOSURE value definitions vary by OS/driver:
    # 3 (or 0.75) typically forces Auto Mode on V4L2 / DirectShow
    cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 3) 
    cap.set(cv2.CAP_PROP_AUTO_WB, 1)        # Auto White Balance
    
    # Optional: Reset hardware brightness/contrast to neutral defaults if previously overridden
    cap.set(cv2.CAP_PROP_BRIGHTNESS, 128)
    cap.set(cv2.CAP_PROP_CONTRAST, 128)

def optimize_frame_for_detection(frame):
    """
    Converts frame to grayscale and applies Contrast Limited Adaptive Histogram 
    Equalization (CLAHE) to boost AprilTag border readability dynamically.
    """
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    return clahe.apply(gray)

def create_transform_matrix(rvec, tvec):
    rotation_matrix, _ = cv2.Rodrigues(rvec)
    transform_matrix = np.eye(4)
    transform_matrix[:3, :3] = rotation_matrix
    transform_matrix[:3, 3] = tvec.squeeze()
    return transform_matrix

def extract_6dof(T_relative):
    x, y, z = T_relative[:3, 3]
    R = T_relative[:3, :3]
    angles, _, _, _, _, _ = cv2.RQDecomp3x3(R)
    pitch, yaw, roll = angles
    return x, y, z, roll, pitch, yaw

def calculate_relative_6dof(rvec1, tvec1, rvec2, tvec2):
    T_cam1_to_tag = create_transform_matrix(rvec1, tvec1)
    T_cam2_to_tag = create_transform_matrix(rvec2, tvec2)
    T_tag_to_cam2 = np.linalg.inv(T_cam2_to_tag)
    T_cam2_to_cam1 = np.dot(T_cam1_to_tag, T_tag_to_cam2)
    return extract_6dof(T_cam2_to_cam1)

def main():
    cap1 = cv2.VideoCapture(0)
    cap2 = cv2.VideoCapture(2)

    # Enable hardware-level auto exposure
    setup_camera_exposure(cap1)
    setup_camera_exposure(cap2)

    # Camera Calibration Intrinsics
    cam_matrix = np.array([[800, 0, 320], [0, 800, 240], [0, 0, 1]], dtype=float)
    dist_coeffs = np.zeros((4, 1))

    # AprilTag Detector Configuration
    dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_APRILTAG_36h11)
    parameters = cv2.aruco.DetectorParameters()
    
    # Optimize detection parameters for varying brightness/contrast
    parameters.adaptiveThreshWinSizeMin = 3
    parameters.adaptiveThreshWinSizeMax = 23
    parameters.adaptiveThreshWinSizeStep = 10
    
    detector = cv2.aruco.ArucoDetector(dictionary, parameters)

    tag_size = 0.165 
    half_size = tag_size / 2.0
    obj_points = np.array([
        [-half_size,  half_size, 0],
        [ half_size,  half_size, 0],
        [ half_size, -half_size, 0],
        [-half_size, -half_size, 0]
    ], dtype=np.float32)

    print("Live tracking with optimized exposure active. Press 'q' to stop.\n")

    while True:
        cap1.grab()
        cap2.grab()
        
        _, frame1 = cap1.retrieve()
        _, frame2 = cap2.retrieve()

        if frame1 is None or frame2 is None:
            break

        # Pre-process frames using CLAHE for dynamic local brightness adjustment
        proc1 = optimize_frame_for_detection(frame1)
        proc2 = optimize_frame_for_detection(frame2)

        # Run detection on enhanced grayscale frames
        corners1, ids1, _ = detector.detectMarkers(proc1)
        corners2, ids2, _ = detector.detectMarkers(proc2)

        if ids1 is not None and ids2 is not None:
            common_ids = np.intersect1d(ids1, ids2)
            
            if len(common_ids) > 0:
                target_id = common_ids[0]
                idx1 = np.where(ids1 == target_id)[0][0]
                idx2 = np.where(ids2 == target_id)[0][0]

                _, rvec1, tvec1 = cv2.solvePnP(obj_points, corners1[idx1], cam_matrix, dist_coeffs)
                _, rvec2, tvec2 = cv2.solvePnP(obj_points, corners2[idx2], cam_matrix, dist_coeffs)

                x, y, z, roll, pitch, yaw = calculate_relative_6dof(rvec1, tvec1, rvec2, tvec2)
                
                print(
                    f"\r[Cam 2 Rel to Cam 1] "
                    f"X: {x:+6.3f}m | Y: {y:+6.3f}m | Z: {z:+6.3f}m | "
                    f"Roll: {roll:+6.1f}° | Pitch: {pitch:+6.1f}° | Yaw: {yaw:+6.1f}°",
                    end="",
                    flush=True
                )

        # Draw detected marker outlines onto color preview frames for visual confirmation
        if ids1 is not None:
            cv2.aruco.drawDetectedMarkers(frame1, corners1, ids1)
        if ids2 is not None:
            cv2.aruco.drawDetectedMarkers(frame2, corners2, ids2)

        cv2.imshow('Camera 1 (Color Feed)', frame1)
        cv2.imshow('Camera 2 (Color Feed)', frame2)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap1.release()
    cap2.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()