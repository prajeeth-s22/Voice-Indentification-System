import sys
import os
import base64
from pathlib import Path

# Add project root directory to sys.path
root_dir = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(root_dir))

from app import app


def handler(event, context):
    """
    Netlify Serverless Function handler for Flask app.
    Converts Netlify event/context into Flask test client response.
    """
    path = event.get("path", "/")
    http_method = event.get("httpMethod", "GET")
    headers = event.get("headers", {})
    query_params = event.get("queryStringParameters") or {}

    body = event.get("body", "")
    if event.get("isBase64Encoded", False) and body:
        try:
            body = base64.b64decode(body)
        except Exception:
            pass

    try:
        with app.test_client() as client:
            if http_method == "GET":
                response = client.get(path, query_string=query_params, headers=headers)
            elif http_method == "POST":
                response = client.post(
                    path, data=body, query_string=query_params, headers=headers
                )
            else:
                response = client.open(
                    path, method=http_method, data=body, query_string=query_params, headers=headers
                )

            # Format headers
            res_headers = {}
            for k, v in response.headers:
                res_headers[k] = v

            return {
                "statusCode": response.status_code,
                "headers": res_headers,
                "body": response.get_data(as_text=True),
            }
    except Exception as e:
        import traceback
        return {
            "statusCode": 500,
            "headers": {"Content-Type": "text/html"},
            "body": f"<h2>Netlify Function Invocation Error</h2><pre>{traceback.format_exc()}</pre>",
        }
