import argparse
import importlib.util
import json
import logging
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlsplit

from .tasks import GROUPS, INPUT_TYPES, InputError, MAX_ROWS, TASKS, catalog, generate, model_for, predict_request


STATIC = Path(__file__).resolve().with_name("web")
ASSETS = {
    "/": ("index.html", "text/html; charset=utf-8"),
    "/app.css": ("app.css", "text/css; charset=utf-8"),
    "/app.js": ("app.js", "text/javascript; charset=utf-8"),
}
MAX_BYTES = 2 * 1024 * 1024


class Handler(BaseHTTPRequestHandler):
    def send_content(self, status, mimetype, data):
        self.send_response(status)
        self.send_header("Content-Type", mimetype)
        self.send_header("Content-Length", str(len(data)))
        self.send_header("Cache-Control", "no-store")
        self.send_header("X-Content-Type-Options", "nosniff")
        self.end_headers()
        self.wfile.write(data)

    def json(self, data, status=HTTPStatus.OK):
        encoded = json.dumps(data, ensure_ascii=False, allow_nan=False).encode("utf-8")
        self.send_content(status, "application/json; charset=utf-8", encoded)

    def do_GET(self):
        path = urlsplit(self.path).path
        if path in ASSETS:
            filename, mimetype = ASSETS[path]
            self.send_content(HTTPStatus.OK, mimetype, (STATIC / filename).read_bytes())
        elif path == "/api/v1/tasks":
            missing = [name for name in ("torch", "numpy", "matplotlib", "PIL") if importlib.util.find_spec(name) is None]
            self.json({"tasks": catalog(), "groups": GROUPS, "input_types": INPUT_TYPES,
                       "max_rows": MAX_ROWS, "max_bytes": MAX_BYTES, "missing_dependencies": missing})
        else:
            self.json({"error": "页面不存在。"}, HTTPStatus.NOT_FOUND)

    def do_POST(self):
        parts = urlsplit(self.path).path.split("/")
        if len(parts) != 6 or parts[:4] != ["", "api", "v1", "tasks"] or parts[4] not in TASKS:
            self.json({"error": "任务不存在。"}, HTTPStatus.NOT_FOUND)
            return
        task = TASKS[parts[4]]
        action = parts[5]
        if action not in ("predict", "generate"):
            self.json({"error": "操作不存在。"}, HTTPStatus.NOT_FOUND)
            return
        if self.headers.get_content_type() != "application/json":
            self.json({"error": "请求需要使用 JSON 格式。"}, HTTPStatus.UNSUPPORTED_MEDIA_TYPE)
            return
        try:
            size = int(self.headers.get("Content-Length", "0"))
        except ValueError:
            self.json({"error": "请求长度格式有误。"}, HTTPStatus.BAD_REQUEST)
            return
        try:
            if size > MAX_BYTES:
                self.json({"error": "请求数据超过 2 MiB。"}, HTTPStatus.REQUEST_ENTITY_TOO_LARGE)
                return
            if size < 1:
                raise InputError("请求需要包含数据。")
            body = json.loads(self.rfile.read(size))
            if not isinstance(body, dict):
                raise InputError("请求需要包含 JSON 对象。")
            model = model_for(task, body.get("model"))
            if action == "generate":
                result = generate(task, model, body.get("parameters", {}))
            else:
                result = predict_request(task, model, body)
            self.json(result)
        except InputError as error:
            self.json({"error": str(error)}, HTTPStatus.BAD_REQUEST)
        except (json.JSONDecodeError, UnicodeError):
            self.json({"error": "请求数据格式有误。"}, HTTPStatus.BAD_REQUEST)
        except ModuleNotFoundError as error:
            logging.exception("Missing dependency")
            self.json({"error": f"缺少 Python 依赖 {error.name}。请先安装工作台依赖。"}, HTTPStatus.SERVICE_UNAVAILABLE)
        except FileNotFoundError:
            logging.exception("Missing model file")
            self.json({"error": "找不到本地模型权重。请确认项目目录完整。"}, HTTPStatus.SERVICE_UNAVAILABLE)
        except Exception:
            logging.exception("Task failed: %s/%s", task.id, action)
            self.json({"error": "运行失败。详细原因已记录在启动工作台的终端。"}, HTTPStatus.INTERNAL_SERVER_ERROR)


def main():
    parser = argparse.ArgumentParser(description="Start the local model workbench")
    parser.add_argument("--port", type=int, default=8765)
    args = parser.parse_args()
    with ThreadingHTTPServer(("127.0.0.1", args.port), Handler) as server:
        print(f"工作台运行于 http://127.0.0.1:{server.server_port}", flush=True)
        try:
            server.serve_forever()
        except KeyboardInterrupt:
            pass


if __name__ == "__main__":
    main()
