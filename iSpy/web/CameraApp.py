import os
from flask import Flask, Response, render_template_string, jsonify, request
import cv2
import threading
import logging
import json

PAGE = """
<!DOCTYPE html>
<html>
<head><title>Camera Feed</title></head>
<body style="margin:0;background:#111">
<img id="feed" src="/video_feed" style="display:block"/>
<script>
  fetch('/dimensions')
    .then(r => r.json())
    .then(d => {
      const img = document.getElementById('feed');
      img.style.width  = d.width  + 'px';
      img.style.height = d.height + 'px';
    });
</script>
</body>
</html>
"""

class CameraApp:
    def __init__(self, cameras=None, config=None):
        self.app    = Flask(__name__)
        self.frame  = None
        self.lock   = threading.Lock()
        self.width  = 640 # Updates once first frame arrives
        self.height = 480 # Updates once first frame arrives
        self.logger = logging.getLogger(__name__)

        # Store camera references
        self.cameras = cameras or []
        self.config = config
        self.camera_frames = {}  # Store frames per camera

        self.app.add_url_rule('/',                           'index',                self._index)
        self.app.add_url_rule('/video_feed',                 'video_feed',           self._video_feed)
        self.app.add_url_rule('/dimensions',                 'dimensions',           self._dimensions)
        self.app.add_url_rule('/api/cameras',                'api_cameras',          self._api_cameras)
        self.app.add_url_rule('/api/camera/<camera_name>/settings', 'api_get_settings', self._api_get_settings, methods=['GET'])
        self.app.add_url_rule('/api/camera/<camera_name>/settings', 'api_update_settings', self._api_update_settings, methods=['POST'])
        self.app.add_url_rule('/api/camera/<camera_name>/feed', 'api_camera_feed', self._api_camera_feed)

    def set_frame(self, frame, camera_name=None):
        if frame is None:
            return
        with self.lock:
            if camera_name:
                self.camera_frames[camera_name] = frame
            else:
                self.frame  = frame
                self.height, self.width = frame.shape[:2]

    def run(self, host='0.0.0.0', port=5000):
        try:
            import werkzeug.serving
            werkzeug.serving.show_server_banner = lambda *a, **kw: None
        except Exception:
            pass
        self.app.run(host=host, port=port, threaded=True)

    def _index(self):
        return render_template_string(PAGE)

    def _dimensions(self):
        from flask import jsonify
        return jsonify(width=self.width, height=self.height)

    def _video_feed(self):
        return Response(
            self._generate(),
            mimetype='multipart/x-mixed-replace; boundary=frame'
        )

    def _api_cameras(self):
        camera_list = []
        for i, camera in enumerate(self.cameras):
            try:
                name = camera.config.get("name", f"Camera {i+1}") if hasattr(camera, 'config') else f"Camera {i+1}"
                camera_list.append({
                    "name": name,
                    "id": i,
                    "source": camera.config.get("source", "unknown") if hasattr(camera, 'config') else "unknown"
                })
            except Exception as e:
                self.logger.warning(f"Error getting camera info: {e}")
                camera_list.append({"name": f"Camera {i+1}", "id": i, "source": "unknown"})
        return jsonify(cameras=camera_list)

    def _api_get_settings(self, camera_name):
        try:
            for i, camera in enumerate(self.cameras):
                cam_name = camera.config.get("name", f"Camera {i+1}") if hasattr(camera, 'config') else f"Camera {i+1}"
                if cam_name == camera_name or str(i) == camera_name:
                    if hasattr(camera, 'config') and hasattr(camera.config, 'data'):
                        return jsonify(settings=camera.config.data)
                    else:
                        return jsonify(error="Camera config not available"), 400
            return jsonify(error="Camera not found"), 404
        except Exception as e:
            self.logger.error(f"Error getting camera settings: {e}")
            return jsonify(error=str(e)), 500

    def _api_update_settings(self, camera_name):
        try:
            data = request.get_json()
            if not data:
                return jsonify(error="No data provided"), 400

            for i, camera in enumerate(self.cameras):
                cam_name = camera.config.get("name", f"Camera {i+1}") if hasattr(camera, 'config') else f"Camera {i+1}"
                if cam_name == camera_name or str(i) == camera_name:
                    if hasattr(camera, 'config') and hasattr(camera.config, 'data'):
                        # Update the config
                        for key, value in data.items():
                            if key in camera.config.data:
                                if isinstance(camera.config.data[key], dict) and isinstance(value, dict):
                                    camera.config.data[key].update(value)
                                else:
                                    camera.config.data[key] = value
                        return jsonify(success=True, settings=camera.config.data)
                    else:
                        return jsonify(error="Camera config not available"), 400
            return jsonify(error="Camera not found"), 404
        except Exception as e:
            self.logger.error(f"Error updating camera settings: {e}")
            return jsonify(error=str(e)), 500

    def _api_camera_feed(self, camera_name):
        try:
            camera_index = None
            for i, camera in enumerate(self.cameras):
                cam_name = camera.config.get("name", f"Camera {i+1}") if hasattr(camera, 'config') else f"Camera {i+1}"
                if cam_name == camera_name or str(i) == camera_name:
                    camera_index = i
                    break

            if camera_index is None:
                return "Camera not found", 404

            return Response(
                self._generate_camera_feed(camera_name),
                mimetype='multipart/x-mixed-replace; boundary=frame'
            )
        except Exception as e:
            self.logger.error(f"Error streaming camera feed: {e}")
            return str(e), 500

    def _generate_camera_feed(self, camera_name):
        import time
        while True:
            with self.lock:
                frame = self.camera_frames.get(camera_name)

            if frame is None:
                time.sleep(0.05)
                continue

            ok, buf = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
            if not ok:
                continue

            yield (
                b'--frame\r\n'
                b'Content-Type: image/jpeg\r\n\r\n'
                + buf.tobytes()
                + b'\r\n'
            )

    def _generate(self):
        import time
        while True:
            with self.lock:
                frame = self.frame

            if frame is None:
                time.sleep(0.05)
                continue

            ok, buf = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
            if not ok:
                continue

            yield (
                b'--frame\r\n'
                b'Content-Type: image/jpeg\r\n\r\n'
                + buf.tobytes()
                + b'\r\n'
            )
