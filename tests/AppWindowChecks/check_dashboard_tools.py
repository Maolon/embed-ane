#!/usr/bin/env python3
"""Compile App-only helpers; exercise the client and copied examples on loopback.

No model, installed app, shared configuration, or third-party Python packages.
Run from anywhere: python3 tests/AppWindowChecks/check_dashboard_tools.py
"""
import json
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
import subprocess
import sys
import tempfile
import threading

ROOT = Path(__file__).resolve().parents[2]
received = []
redirects = []


class Fixture(BaseHTTPRequestHandler):
    def log_message(self, *_):
        pass

    def do_POST(self):
        body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        received.append((self.path, body))
        model = body["model"]
        if self.path == "/redirect":
            redirects.append(body)
        if model == "fixture-redirect":
            self.send_response(307)
            self.send_header("Location", f"http://127.0.0.1:{self.server.server_port}/redirect")
            self.end_headers()
            return
        if model == "fixture-loading":
            self.send_response(503)
            payload = {"error": {"code": "loading", "message": "raw-secret-error"}}
        else:
            self.send_response(200)
            payload = {"object": "list", "data": [{"object": "embedding", "index": 0,
                       "embedding": [0, 0] if model == "fixture-malformed" else [0.6, 0.8, 0, 0]}]}
        data = json.dumps(payload).encode()
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)


with tempfile.TemporaryDirectory(prefix="embed-ane-window-checks-") as temporary:
    binary = Path(temporary) / "dashboard-tools-checks"
    subprocess.run(["swiftc", "-swift-version", "6", "-target", "arm64-apple-macos15.0", "-parse-as-library",
                    str(ROOT / "App/Sources/DashboardTools.swift"),
                    str(ROOT / "tests/AppWindowChecks/DashboardToolsChecks.swift"),
                    "-o", str(binary)], check=True)
    server = HTTPServer(("127.0.0.1", 0), Fixture)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        result = subprocess.run([str(binary), str(server.server_port)], check=True, text=True,
                                stdout=subprocess.PIPE, timeout=30)
        examples = json.loads(result.stdout)
        assert len(received) == 4, received
        assert not redirects, "The client must refuse redirects"
        for example in examples:
            for command in (["/bin/bash", "-c", example["curl"]], [sys.executable, "-c", example["python"]]):
                before = len(received)
                output = subprocess.run(command, check=True, text=True, stdout=subprocess.PIPE,
                                        stderr=subprocess.PIPE, timeout=10)
                assert json.loads(output.stdout)["object"] == "list", output.stdout
                assert len(received) == before + 1, "Exactly one request per copied example"
                assert received[-1] == ("/v1/embeddings", {"model": "fixture-valid",
                    "input": example["text"], "encoding_format": "float"}), received[-1]
        assert not redirects
        print(f"PASS: request/preview/error checks, 4 loopback client cases, {len(examples) * 2} executed curl/Python round trips; redirects refused.")
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)
