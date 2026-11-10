#!/usr/bin/env python3
"""Headless Chromium smoke checks for the trellis2.cpp demo.

Uses only the Python standard library and Chromium's DevTools protocol so it can
run on the headless demo host without Playwright/Selenium. It intentionally
loads a persisted mesh and the showcase, exercising fetch, binary parsing,
WebGL2 buffer upload, and rendering rather than merely checking the HTML.
"""

import argparse
import base64
import json
import os
import secrets
import socket
import struct
import subprocess
import tempfile
import time
import urllib.request
from pathlib import Path
from urllib.parse import urlparse


class CDP:
    def __init__(self, url):
        parsed = urlparse(url)
        self.sock = socket.create_connection((parsed.hostname, parsed.port), timeout=15)
        key = base64.b64encode(secrets.token_bytes(16)).decode()
        request = (
            f"GET {parsed.path} HTTP/1.1\r\n"
            f"Host: {parsed.hostname}:{parsed.port}\r\n"
            "Upgrade: websocket\r\nConnection: Upgrade\r\n"
            f"Sec-WebSocket-Key: {key}\r\nSec-WebSocket-Version: 13\r\n\r\n"
        )
        self.sock.sendall(request.encode())
        response = b""
        while b"\r\n\r\n" not in response:
            response += self.sock.recv(4096)
        if b" 101 " not in response.split(b"\r\n", 1)[0]:
            raise RuntimeError(f"WebSocket upgrade failed: {response[:200]!r}")
        self.next_id = 1
        self.events = []

    def _send_frame(self, payload):
        data = payload.encode()
        mask = secrets.token_bytes(4)
        n = len(data)
        header = bytearray([0x81])
        if n < 126:
            header.append(0x80 | n)
        elif n < 65536:
            header.append(0x80 | 126)
            header.extend(struct.pack("!H", n))
        else:
            header.append(0x80 | 127)
            header.extend(struct.pack("!Q", n))
        header.extend(mask)
        header.extend(bytes(b ^ mask[i % 4] for i, b in enumerate(data)))
        self.sock.sendall(header)

    def _read_exact(self, n):
        chunks = []
        while n:
            chunk = self.sock.recv(n)
            if not chunk:
                raise RuntimeError("DevTools WebSocket closed")
            chunks.append(chunk)
            n -= len(chunk)
        return b"".join(chunks)

    def _recv_frame(self):
        b0, b1 = self._read_exact(2)
        opcode, n = b0 & 0x0F, b1 & 0x7F
        if n == 126:
            n = struct.unpack("!H", self._read_exact(2))[0]
        elif n == 127:
            n = struct.unpack("!Q", self._read_exact(8))[0]
        masked = bool(b1 & 0x80)
        mask = self._read_exact(4) if masked else None
        data = self._read_exact(n)
        if mask:
            data = bytes(b ^ mask[i % 4] for i, b in enumerate(data))
        if opcode == 0x9:  # ping
            self._send_frame(data.decode(errors="ignore"))
            return self._recv_frame()
        if opcode == 0x8:
            raise RuntimeError("DevTools WebSocket closed")
        return json.loads(data.decode())

    def call(self, method, params=None, timeout=90):
        ident = self.next_id
        self.next_id += 1
        self._send_frame(json.dumps({"id": ident, "method": method, "params": params or {}}))
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            self.sock.settimeout(max(0.1, deadline - time.monotonic()))
            message = self._recv_frame()
            if message.get("id") == ident:
                if "error" in message:
                    raise RuntimeError(f"{method}: {message['error']}")
                return message.get("result", {})
            self.events.append(message)
        raise TimeoutError(method)

    def evaluate(self, expression, timeout=90):
        result = self.call("Runtime.evaluate", {
            "expression": expression,
            "awaitPromise": True,
            "returnByValue": True,
            "userGesture": True,
        }, timeout)
        value = result.get("result", {})
        if "exceptionDetails" in result:
            raise RuntimeError(result["exceptionDetails"])
        return value.get("value")


def wait_json(url, timeout=15):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        try:
            with urllib.request.urlopen(url, timeout=1) as response:
                return json.load(response)
        except Exception:
            time.sleep(0.1)
    raise TimeoutError(url)


def screenshot(cdp, path):
    encoded = cdp.call("Page.captureScreenshot", {
        "format": "png", "captureBeyondViewport": False,
    }, timeout=30)["data"]
    path.write_bytes(base64.b64decode(encoded))


def navigate(cdp, url, timeout=30):
    cdp.call("Page.navigate", {"url": url}, timeout=timeout)
    expected = urlparse(url).path or "/"
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        try:
            state = cdp.evaluate("({path: location.pathname, ready: document.readyState})", timeout=2)
            if state and state["path"] == expected and state["ready"] == "complete":
                return
        except Exception:
            pass
        time.sleep(0.1)
    raise TimeoutError(f"navigation to {url}")


def browser_errors(events):
    errors = []
    for event in events:
        method, params = event.get("method"), event.get("params", {})
        if method == "Runtime.exceptionThrown":
            detail = params.get("exceptionDetails", {})
            errors.append(detail.get("text", "JavaScript exception"))
        elif method == "Log.entryAdded":
            entry = params.get("entry", {})
            if entry.get("level") in ("error", "warning"):
                location = f" ({entry['url']})" if entry.get("url") else ""
                errors.append(f"{entry.get('level')}: {entry.get('text', '')}{location}")
        elif method == "Runtime.consoleAPICalled" and params.get("type") == "error":
            args = params.get("args", [])
            errors.append("console: " + " ".join(str(a.get("value", a.get("description", ""))) for a in args))
    return errors


MAIN_CHECK = r"""
(async () => {
  const sleep = ms => new Promise(resolve => setTimeout(resolve, ms));
  for (let i = 0; i < 100 && (!history || !history.length); i++) await sleep(100);
  if (!history.length) throw new Error('no persisted generations in history');
  // Prefer the oldest retained asset: it is stable across new generations and
  // avoids coupling this reusable smoke test to a particular server/job ID.
  const preferred = history[history.length - 1];
  await viewGeneration(preferred.id);
  await sleep(1200);
  await new Promise(resolve => requestAnimationFrame(() => requestAnimationFrame(resolve)));
  const mesh = originalExportMesh;
  const pbr = mesh && mesh.pbr;
  const range = pbr ? {min: Array(6).fill(Infinity), max: Array(6).fill(-Infinity)} : null;
  if (pbr) {
    const stride = Math.max(6, Math.floor((pbr.length / 6) / 50000) * 6);
    for (let i = 0; i < pbr.length; i += stride) {
      for (let c = 0; c < 6; c++) {
        range.min[c] = Math.min(range.min[c], pbr[i + c]);
        range.max[c] = Math.max(range.max[c], pbr[i + c]);
      }
    }
  }
  const canvas = document.getElementById('gl');
  const gl = canvas.getContext('webgl2');
  const pixel = new Uint8Array(4);
  gl.readPixels(canvas.width >> 1, canvas.height >> 1, 1, 1, gl.RGBA, gl.UNSIGNED_BYTE, pixel);
  return {
    path: location.pathname, title: document.title,
    historyCount: history.length, selected: preferred.id,
    status: document.getElementById('status').textContent,
    meshInfo: document.getElementById('meshinfo').textContent,
    mesh: mesh ? {vertices: mesh.nv, triangles: mesh.nt, textured: mesh.textured} : null,
    pbrRange: range, webgl2: !!gl, glError: gl.getError(), centerPixel: [...pixel],
    canvas: {width: canvas.width, height: canvas.height,
             cssWidth: canvas.clientWidth, cssHeight: canvas.clientHeight},
    liveSteps: {text: document.getElementById('tsteps').textContent,
                pressed: document.getElementById('tsteps').getAttribute('aria-pressed'),
                keyframesDisabled: document.getElementById('tkf').disabled},
    exportEnabled: !document.getElementById('dglb').disabled,
    regenerateEnabled: !document.getElementById('regen').disabled,
  };
})()
"""


SHOWCASE_CHECK = r"""
(async () => {
  const sleep = ms => new Promise(resolve => setTimeout(resolve, ms));
  for (let i = 0; i < 600; i++) {
    if (typeof showcaseAssets !== 'undefined' && history.length && showcaseAssets.length === history.length) break;
    await sleep(100);
  }
  await sleep(1800);
  await new Promise(resolve => requestAnimationFrame(() => requestAnimationFrame(resolve)));
  const canvas = document.getElementById('gl');
  const gl = canvas.getContext('webgl2');
  const display = selector => getComputedStyle(document.querySelector(selector)).display;
  const pixel = new Uint8Array(4);
  gl.readPixels(canvas.width >> 1, canvas.height >> 1, 1, 1, gl.RGBA, gl.UNSIGNED_BYTE, pixel);
  return {
    path: location.pathname, title: document.title,
    historyCount: history.length,
    loadedAssets: typeof showcaseAssets === 'undefined' ? -1 : showcaseAssets.length,
    running: typeof showcaseRunning === 'undefined' ? false : showcaseRunning,
    label: document.getElementById('showcaselabel').textContent,
    sourceClass: document.getElementById('showcasesource').className,
    chrome: {header: display('header'), side: display('#side'), timeline: display('#timeline')},
    webgl2: !!gl, glError: gl.getError(), centerPixel: [...pixel],
    canvas: {width: canvas.width, height: canvas.height,
             cssWidth: canvas.clientWidth, cssHeight: canvas.clientHeight},
  };
})()
"""


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", default="http://127.0.0.1:8742")
    parser.add_argument("--chromium", default="chromium")
    parser.add_argument("--output", default="/tmp/trellis2-headless")
    args = parser.parse_args()
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)

    with tempfile.TemporaryDirectory(prefix="trellis2-chrome-") as profile:
        command = [
            args.chromium, "--headless=new", "--no-sandbox", "--disable-dev-shm-usage",
            "--enable-webgl", "--use-angle=swiftshader", "--enable-unsafe-swiftshader",
            "--remote-debugging-port=0", "--remote-allow-origins=*",
            f"--user-data-dir={profile}", "--window-size=1280,900", "about:blank",
        ]
        browser = subprocess.Popen(command, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
        try:
            port_file = Path(profile) / "DevToolsActivePort"
            deadline = time.monotonic() + 15
            while not port_file.exists() and time.monotonic() < deadline:
                if browser.poll() is not None:
                    raise RuntimeError(browser.stderr.read().decode(errors="replace"))
                time.sleep(0.1)
            port = int(port_file.read_text().splitlines()[0])
            targets = wait_json(f"http://127.0.0.1:{port}/json/list")
            page = next(t for t in targets if t["type"] == "page")
            cdp = CDP(page["webSocketDebuggerUrl"])
            for method in ("Page.enable", "Runtime.enable", "Log.enable"):
                cdp.call(method)

            results = {}
            navigate(cdp, args.url.rstrip("/") + "/")
            results["main"] = cdp.evaluate(MAIN_CHECK, timeout=120)
            screenshot(cdp, output / "main.png")

            navigate(cdp, args.url.rstrip("/") + "/showcase")
            results["showcase"] = cdp.evaluate(SHOWCASE_CHECK, timeout=180)
            screenshot(cdp, output / "showcase.png")
            results["browserErrors"] = browser_errors(cdp.events)
            results["screenshots"] = [str(output / "main.png"), str(output / "showcase.png")]
            failures = []
            main_result, showcase_result = results["main"], results["showcase"]
            if not main_result["webgl2"] or main_result["glError"] != 0:
                failures.append("main page WebGL2 failure")
            if not main_result["mesh"]["textured"] or not main_result["pbrRange"]:
                failures.append("main page did not load PBR data")
            elif max(main_result["pbrRange"]["max"][c] - main_result["pbrRange"]["min"][c]
                     for c in range(3)) < 0.1:
                failures.append("main page base colour is effectively constant")
            if main_result["status"] != "done":
                failures.append(f"main page status is {main_result['status']!r}")
            if main_result["liveSteps"] != {
                    "text": "live steps: off", "pressed": "false", "keyframesDisabled": True}:
                failures.append(f"live steps did not default clearly off: {main_result['liveSteps']!r}")
            if showcase_result["loadedAssets"] != showcase_result["historyCount"]:
                failures.append("showcase did not load every persisted asset")
            if not showcase_result["webgl2"] or showcase_result["glError"] != 0:
                failures.append("showcase WebGL2 failure")
            if any(value != "none" for value in showcase_result["chrome"].values()):
                failures.append("showcase page chrome is visible")
            significant_errors = [e for e in results["browserErrors"] if "favicon.ico" not in e]
            if significant_errors:
                failures.extend(significant_errors)
            results["failures"] = failures
            results["passed"] = not failures
            print(json.dumps(results, indent=2))
            if failures:
                raise SystemExit(1)
        finally:
            browser.terminate()
            try:
                browser.wait(timeout=5)
            except subprocess.TimeoutExpired:
                browser.kill()


if __name__ == "__main__":
    main()
