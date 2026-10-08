import sys
import os
import traceback
from flask import Flask

# Ensure parent directory is in sys.path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    from app import app
except Exception as e:
    err_str = traceback.format_exc()
    app = Flask(__name__)

    @app.route("/", defaults={"path": ""})
    @app.route("/<path:path>")
    def catch_all(path):
        return (
            f"<h2>Vercel Serverless Initialization Error</h2>"
            f"<p>An error occurred during startup on Vercel:</p>"
            f"<pre style='background:#f4f4f4;padding:15px;border-radius:5px;overflow:auto;'>{err_str}</pre>",
            500,
        )
