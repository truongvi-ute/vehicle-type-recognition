from __future__ import annotations

import os
import sys
from pathlib import Path

from flask import Flask, jsonify
from flask_cors import CORS


BACKEND_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = BACKEND_DIR.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

from routes.predict import predict_bp  # noqa: E402
from routes.metrics import metrics_bp  # noqa: E402


def create_app() -> Flask:
    app = Flask(__name__)
    app.config["MAX_CONTENT_LENGTH"] = 10 * 1024 * 1024
    app.config["MODELS_DIR"] = str(PROJECT_ROOT / "models")

    CORS(app, resources={r"/api/*": {"origins": "*"}})
    app.register_blueprint(predict_bp, url_prefix="/api")
    app.register_blueprint(metrics_bp, url_prefix="/api")

    @app.get("/api/health")
    def health() -> tuple[object, int]:
        return jsonify({"status": "ok"}), 200

    @app.errorhandler(413)
    def file_too_large(_error: object) -> tuple[object, int]:
        return jsonify({"error": "Uploaded image is too large. Limit is 10 MB."}), 413

    return app


app = create_app()


if __name__ == "__main__":
    host = os.environ.get("FLASK_HOST", "0.0.0.0")
    port = int(os.environ.get("FLASK_PORT", "5000"))
    debug = os.environ.get("FLASK_DEBUG", "0") == "1"
    app.run(host=host, port=port, debug=debug)
