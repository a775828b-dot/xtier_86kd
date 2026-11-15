"""XTier console UI for tf500's TP8 baseline and TP4 hybrid deployment.

Run with /data/xtier-env/bin/python. The active system unit is adopted for
status without restarting it. Start stops that process, then launches the
selected script as an independent session.
"""

import os
import re
import signal
import subprocess
import time
import urllib.error
import urllib.request
from http.server import ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

from xtier.console import core, ui
from xtier.console.__main__ import Jobs, make_handler
import tf500_benchmark
import tf500_api_proxy
import tf500_proxy_admin
import tf500_proxy_keys
import tf500_nav
import tf500_server_info

HOST = "<TF500_IP>"
PORT = 30000
TP4_PORT = 30001
SERVICE = "sglang-ds41.service"
TP4_SERVICE = "xtier-tp4.service"
MODEL_ID = "ds41-86kd"
TP4_MODEL_ID = "ds41-xtier-tp4"
API_KEY_FILE = Path("/data/sglang-api-key")
STATE_DIR = Path("/data/xtier-86kd-console")
LOG_DIR = STATE_DIR / "logs"

MODEL = {
    "id": MODEL_ID,
    "order": 10,
    "name": "DeepSeek V4.1 Flash 1017",
    "code": "DSV4.1 · 86KD",
    "port": PORT,
    "served": "DeepSeek-V4.1-Flash-1017",
    "spec": "DSpark",
    "arch": "TP8 · Humming MoE · 86KD 独立加速插件",
    "hardware": "RTX PRO 6000D × 8 · tf500",
    "ready_timeout": 3600,
    "bench": {
        "kind": "text",
        "dec": "326 / 2678",
        "pre": "12386 / 14105",
        "title": "2026-11-15 单轮开发测量：共享 32K 前缀缓存 decode，独立冷 32K prefill；数值依次为并发 1 / 32 tok/s。",
    },
    "options": [
        {"key": "packs", "label": "86KD 加速包", "default": "on", "switch": True,
         "choices": [["on", "开 · 当前配置"], ["off", "关 · SGLang 基线"]]},
        {"key": "spec", "label": "DSpark 投机解码", "default": "on", "switch": True,
         "choices": [["on", "开 · 当前配置"], ["off", "关"]]},
        {"key": "verify", "label": "DSpark 验证", "default": "compact",
         "choices": [["compact", "变长 · 本机成本表"], ["static", "固定"]],
         "off_when": [{"spec": "off"}]},
        {"key": "context", "label": "上下文长度", "default": "1048576",
         "choices": [["1048576", "1M · 当前配置"], ["524288", "512K"], ["262144", "256K"]]},
        {"key": "running", "label": "最大并行请求", "default": "64",
         "choices": [["64", "64 · 当前配置"], ["32", "32"], ["16", "16"], ["4", "4"]]},
        {"key": "graph", "label": "Decode CUDA 图批量", "default": "32",
         "choices": [["32", "32 · 当前配置"], ["16", "16"], ["8", "8"]]},
        {"key": "memory", "label": "静态显存比例", "default": "0.78",
         "choices": [["0.78", "0.78 · 当前配置"], ["0.74", "0.74"], ["0.70", "0.70"]]},
    ],
    "notes": [
        {"when": {"packs": "off"}, "text": "关闭 86KD 钩子，不改变 SGLang、TP8 和 Humming MoE。"},
        {"when": {"spec": "off"}, "text": "关闭 DSpark；验证模式选项不生效。"},
    ],
}

TP4_MODEL = {
    "id": TP4_MODEL_ID,
    "order": 20,
    "name": "DeepSeek V4.1 Flash 1017 · TP4 混合",
    "code": "DSV4.1 · XTier TP4",
    "port": TP4_PORT,
    "served": "DeepSeek-V4.1-Flash-1017",
    "spec": "DSpark",
    "arch": "TP4 · XTier CPU/GPU 混合推理 · 静态变长验证",
    "hardware": "RTX PRO 6000D × 4 · CPU0/NUMA0 · tf500",
    "ready_timeout": 3600,
    "bench": {
        "kind": "text",
        "dec": "126 单发",
        "pre": "2326 单发",
        "title": "2026-11-15 单发 32K 输入 / 1K 输出：冷 prefill 约 2326 tok/s；"
                 "KV 命中 32512 token 后 decode 125.71 tok/s。未测 TP4 并发。",
    },
    "options": [
        {"key": "engine", "label": "XTier 混合推理", "default": "on",
         "choices": [["on", "开 · 当前配置"]]},
        {"key": "spec", "label": "DSpark 投机解码", "default": "on",
         "choices": [["on", "开 · draft 6 token"]]},
        {"key": "verify", "label": "DSpark 验证", "default": "static",
         "choices": [["static", "静态变长 · 当前配置"]]},
        {"key": "engram", "label": "Engram 放置", "default": "cpu",
         "choices": [["cpu", "CPU 内存 · 不复制"]]},
        {"key": "context", "label": "上下文长度", "default": "524288",
         "choices": [["524288", "512K · 当前配置"], ["262144", "256K"]]},
        {"key": "kvpool", "label": "KV 池 Token", "default": "20000000",
         "choices": [["20000000", "20M · 当前配置"], ["16000000", "16M"], ["12000000", "12M"]]},
        {"key": "running", "label": "最大并行请求", "default": "32",
         "choices": [["32", "32 · 当前配置"], ["16", "16"], ["8", "8"], ["4", "4"]]},
        {"key": "batch", "label": "Prefill 分块", "default": "49152",
         "choices": [["49152", "48K · 当前配置"], ["24576", "24K"], ["16384", "16K"]]},
        {"key": "interleave", "label": "Prefill 中穿插 decode", "default": "72",
         "choices": [["72", "72 步 · 当前配置"]]},
        {"key": "graph", "label": "Decode CUDA 图批量", "default": "32",
         "choices": [["32", "32 · 当前配置"]]},
        {"key": "memory", "label": "静态显存比例", "default": "0.85",
         "choices": [["0.85", "0.85 · 当前配置"]]},
        {"key": "reserve", "label": "动态热专家显存预留", "default": "2",
         "choices": [["2", "2 GiB/卡 · 当前配置"]]},
    ],
    "notes": [
        {"when": {}, "text": "GPU0–3 + CPU0；每 Rank 12 线程；动态热专家初始 35 槽；GPU4–7 留给第二实例。"},
    ],
}


def build(options, port):
    env = {
        "XTIER_86KD": "1" if options["packs"] == "on" else "0",
        "USE_DSPARK": "1" if options["spec"] == "on" else "0",
        "SGLANG_RAGGED_VERIFY_MODE": options["verify"] if options["spec"] == "on" else "static",
        "SPS_TABLE": "/data/bench/86kd-sps-6000d.json" if options["spec"] == "on" and options["verify"] == "compact" else "",
        "CONTEXT_LENGTH": options["context"],
        "MAX_RUNNING": options["running"],
        "CUDA_GRAPH_MAX_BS_DECODE": options["graph"],
        "MEM_FRACTION_STATIC": options["memory"],
        "HOST": HOST,
        "PORT": str(port),
        "MOE_BACKEND": "humming",
        "API_KEY_FILE": str(API_KEY_FILE),
    }
    return env, ["/bin/bash", "/data/src/start-sglang-plan.sh"]


def build_tp4(options, port):
    env = {
        "HOST": HOST,
        "PORT": str(port),
        "CONTEXT": options["context"],
        "KV_POOL": options["kvpool"],
        "MAX_RUNNING": options["running"],
        "BATCH": options["batch"],
        "INTERLEAVE": options["interleave"],
        "MEM": options["memory"],
        "CUDA_VISIBLE_DEVICES": "0,1,2,3",
        "API_KEY_FILE": str(API_KEY_FILE),
    }
    return env, ["/bin/bash", "/data/src/xtier_86kd/deploy/start-xtier-tp4.sh"]


def tags(options):
    return ["TP8", "86KD" if options["packs"] == "on" else "基线",
            "变长验证" if options["spec"] == "on" and options["verify"] == "compact" else "固定验证"]


def tags_tp4(options):
    return ["TP4", "XTier 混合", "CPU Engram", "动态热专家", "静态变长验证"]


def _service_pid(service):
    try:
        result = subprocess.run(["systemctl", "show", service, "-p", "ActiveState", "-p", "MainPID"],
                                capture_output=True, text=True, timeout=5, check=True)
        fields = dict(line.split("=", 1) for line in result.stdout.splitlines() if "=" in line)
        if fields.get("ActiveState") == "active":
            return int(fields.get("MainPID", "0"))
    except (OSError, ValueError, subprocess.SubprocessError):
        pass
    return 0


def _models_ok(port):
    try:
        key = API_KEY_FILE.read_text(encoding="utf-8").strip()
        req = urllib.request.Request(f"http://{HOST}:{port}/v1/models",
                                     headers={"Authorization": f"Bearer {key}"})
        with urllib.request.urlopen(req, timeout=2) as response:
            return response.status == 200
    except (OSError, urllib.error.URLError):
        return False


def _journal_tail(service):
    try:
        result = subprocess.run(["journalctl", "-u", service, "-n", "350", "--no-pager", "-o", "cat"],
                                capture_output=True, text=True, timeout=5)
        return result.stdout[-131072:] if result.returncode == 0 else ""
    except (OSError, subprocess.SubprocessError):
        return ""


def _uptime(pid):
    try:
        out = subprocess.run(["ps", "-o", "etimes=", "-p", str(pid)], capture_output=True,
                             text=True, timeout=3, check=True).stdout.strip()
        return int(out)
    except (OSError, ValueError, subprocess.SubprocessError):
        return 0


def _expected_server(pid, model):
    try:
        proc = Path(f"/proc/{pid}")
        command = (proc / "cmdline").read_bytes()
        args = command.split(b"\0")
        def arg(name):
            try:
                return args[args.index(name) + 1]
            except (ValueError, IndexError):
                return b""
        if proc.stat().st_uid != os.getuid():
            return False
        if model == TP4_MODEL_ID:
            return (b"start-xtier-tp4.sh" in command or
                    (b"sglang.launch_server" in command and arg(b"--tp-size") == b"4"
                     and arg(b"--port") == str(TP4_PORT).encode()))
        if model == MODEL_ID:
            return (b"start-sglang-plan.sh" in command or
                    (b"sglang.launch_server" in command and arg(b"--tp-size") == b"8"
                     and arg(b"--port") == str(PORT).encode()))
        return False
    except OSError:
        return False


def _gpu_summary(model):
    """Show and guard only the GPUs belonging to the selected launch card."""
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=index,name,memory.used,memory.total,utilization.gpu,power.draw",
             "--format=csv,noheader,nounits"], capture_output=True, text=True, timeout=5, check=True).stdout
        rows = [[part.strip() for part in line.split(",")] for line in out.splitlines() if line.count(",") >= 5]
        indices = set(range(4)) if model == TP4_MODEL_ID else set(range(8))
        selected = [r for r in rows if int(r[0]) in indices]
        if len(selected) != len(indices):
            return {}
        return {"name": selected[0][1].replace("NVIDIA ", ""), "count": len(selected),
                "mem_used": round(sum(float(r[2]) for r in selected) / 1024, 1),
                "mem_total": round(sum(float(r[3]) for r in selected) / 1024, 1),
                "util": round(sum(float(r[4]) for r in selected) / len(selected)),
                "power": round(sum(float(r[5]) for r in selected))}
    except (OSError, ValueError, IndexError, subprocess.SubprocessError):
        return {}


class Tf500Console(core.Console):
    def __init__(self):
        models = {
            MODEL_ID: dict(MODEL, config="tf500", build=build, tags=tags),
            TP4_MODEL_ID: dict(TP4_MODEL, config="tf500-tp4", build=build_tp4, tags=tags_tp4),
        }
        super().__init__(models, STATE_DIR, log_dir=LOG_DIR, runner="setsid", guard_gib=0,
                         cwd="/data/src/sglang-xtier")

    def describe(self, model, options):
        if model == TP4_MODEL_ID:
            return ("XTier · DeepSeek V4.1 Flash 1017 / TP4 混合 / CPU Engram / "
                    f"DSpark 6 / {int(options['context']) // 1024}K")
        return super().describe(model, options)

    def read_state(self):
        saved = core.Console.read_state(self)
        if saved and self.active(saved):
            return saved
        for service, model in ((TP4_SERVICE, TP4_MODEL_ID), (SERVICE, MODEL_ID)):
            pid = _service_pid(service)
            if pid and _expected_server(pid, model):
                spec = self.models[model]
                return {"model": model, "options": self.defaults(model),
                        "label": self.describe(model, self.defaults(model)) + "（系统服务）",
                        "runner": "adopted", "service": service, "pid": pid,
                        "port": spec["port"], "served": spec["served"],
                        "started_at": time.time() - _uptime(pid), "ready_at": None, "log": None}
        return saved

    def active(self, state=None):
        state = self.read_state() if state is None else state
        if state.get("runner") == "adopted":
            service = state.get("service") or (TP4_SERVICE if state.get("model") == TP4_MODEL_ID else SERVICE)
            return state.get("pid") == _service_pid(service) and _expected_server(state["pid"], state.get("model"))
        if not super().active(state):
            return False
        return _expected_server(state["pid"], state.get("model"))

    def stop(self, reason=""):
        state = self.read_state()
        if state.get("runner") != "adopted":
            return super().stop(reason)
        pid = state["pid"]
        if not self.active(state):
            return False
        service = state.get("service") or (TP4_SERVICE if state.get("model") == TP4_MODEL_ID else SERVICE)
        try:
            os.kill(pid, signal.SIGTERM)
        except ProcessLookupError:
            return True
        except PermissionError:
            return False
        for _ in range(90):
            if not _service_pid(service):
                return True
            time.sleep(1)
        return False

    def start(self, model, values):
        if self.active():
            raise RuntimeError("现有服务仍在运行，未启动第二个实例")
        return super().start(model, values)

    def wait_gpu_free(self, model, max_gib=2.0, timeout=180):
        deadline = time.monotonic() + timeout
        while True:
            gpu = _gpu_summary(model)
            if not gpu or gpu["mem_used"] <= max_gib:
                return True
            if time.monotonic() >= deadline:
                return False
            time.sleep(2)

    def wait_ready(self, state, timeout):
        deadline = time.monotonic() + timeout
        self._waiting.add(state["port"])
        try:
            while time.monotonic() < deadline:
                if not self.active(state):
                    return False, "服务提前退出：" + " | ".join(core.log_tail(state.get("log"), 6))
                if _models_ok(state["port"]):
                    state["ready_at"] = time.time()
                    self.write_state(state)
                    return True, ""
                time.sleep(3)
            return False, f"启动超时（{timeout // 60} 分钟）"
        finally:
            self._waiting.discard(state["port"])

    def console_data(self):
        data = super().console_data()
        data["texts"] = {
            "subtitle": "tf500 · TP8 基准 + XTier TP4 混合 · DeepSeek V4.1 Flash 1017",
            "foot": "系统服务会直接显示。切换卡片并启动会先停止当前模型；服务器重启后由已启用的系统服务恢复默认配置。",
            "confirm": "启动 {name}？\n会先停止正在运行的模型，加载可能需要较长时间。API 将监听 <TF500_IP>:{port}。",
            "endpoint": "http://<TF500_IP>:{port}/v1 · {served}",
        }
        return data

    def status(self):
        with self._lock:
            if self._status["body"] is not None and time.monotonic() - self._status["t"] < 1.5:
                return self._status["body"]
        state = self.read_state()
        model = state.get("model") if state.get("model") in self.models else MODEL_ID
        spec = self.models[model]
        body = {"state": "offline", "model": state.get("model"), "label": state.get("label", ""),
                "gpu": _gpu_summary(model), "port": state.get("port", spec["port"]),
                "served": state.get("served", spec["served"])}
        if state and self.active(state):
            adopted = state.get("runner") == "adopted"
            service = state.get("service") or (TP4_SERVICE if model == TP4_MODEL_ID else SERVICE)
            log = _journal_tail(service) if adopted else core._read_tail(state.get("log"))
            elapsed = round(time.time() - state.get("started_at", time.time()))
            body.update(started_at=state.get("started_at"), elapsed=elapsed,
                        options=state.get("options"), version="XTier TP4" if model == TP4_MODEL_ID else "86KD")
            if _models_ok(state["port"]) and state["port"] not in self._waiting:
                body.update(state="online", progress=100, phase="online", phase_label="在线")
            else:
                phase, label, percent = core._progress(log)
                body.update(state="loading", progress=percent, phase=phase, phase_label=label)
            decodes = core._DECODE_AT.findall(log)[-40:]
            if decodes:
                stamp, running, tokens, accept, tps = decodes[-1]
                spark = [float(row[4]) for row in decodes]
                if core._fresh([row[0] for row in decodes], time.time(), 6.0):
                    body.update(running=int(running), tokens=int(tokens), accept=float(accept) if accept else None,
                                tps=float(tps), spark=spark)
                else:
                    body.update(running=0, tokens=int(tokens), tps=0.0, spark=spark + [0.0])
            prefills = core._PREFILL.findall(log)
            if prefills:
                body["last_prefill"] = int(prefills[-1][1])
            if not adopted and state.get("ready_at"):
                body["load_s"] = round(state["ready_at"] - state["started_at"])
            lines = [line.strip() for line in log.splitlines() if line.strip() and not core._POLL.search(line)
                     and not re.search(r"api.key|authorization|bearer |/data/sglang-api-key", line, re.I)]
            body["tail"] = [core._short(line) for line in lines[-6:]]
        elif state.get("started_at") and not state.get("stopped_at"):
            body.update(state="failed", error=state.get("error") or "进程已退出",
                        tail=core.log_tail(state.get("log"), 6))
        with self._lock:
            self._status.update(t=time.monotonic(), body=body)
        return body


class Tf500Jobs(Jobs):
    def __init__(self, console, benchmark):
        super().__init__(console)
        self.benchmark = benchmark

    def launch(self, model, values):
        if self.benchmark.status()["state"] in tf500_benchmark.ACTIVE:
            return None, "测速进行中，请先停止测速"
        return super().launch(model, values)

    def stop(self):
        if self.benchmark.status()["state"] in tf500_benchmark.ACTIVE:
            self.benchmark.stop()
        return super().stop()

    def _run(self, job, model, values):
        console, spec = self.console, self.console.models[model]
        try:
            if not console.stop():
                raise RuntimeError("当前服务停止超时，未启动第二个实例")
            if not console.wait_gpu_free(model):
                raise RuntimeError(f"目标 GPU 仍占用 {_gpu_summary(model).get('mem_used')} GiB，未启动实例")
            if job["cancelled"]:
                raise RuntimeError("已手动停止")
            state = console.start(model, values)
            console.save_settings(model, values)
            ok, detail = console.wait_ready(state, spec["ready_timeout"])
            if job["cancelled"]:
                raise RuntimeError("已手动停止")
            if not ok:
                console.stop(detail)
                raise RuntimeError(detail)
            job.update(state="done", message=f"{job['label']} 已就绪：http://{HOST}:{spec['port']}/v1，模型名 {spec['served']}")
        except Exception as exc:
            job.update(state="failed", error=str(exc))


def make_tf500_handler(console, jobs, benchmark):
    base = make_handler(console, jobs)

    class Handler(base):
        def _admin_json(self, value, status=200):
            import json
            body = json.dumps(value, ensure_ascii=False).encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.send_header("X-Content-Type-Options", "nosniff")
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self):
            path = urlparse(self.path).path
            if path == "/":
                data = {**console.console_data(), "status": console.status()}
                page = tf500_nav.decorate(ui.page_html(data, title="XTier 86KD Console"), "console")
                page = page.replace(
                    "README 实测 tok/s · 文本：64K / 512K 提示 · AGENT：编码代理会话（悬停看测法）",
                    "README 实测 tok/s · 悬停查看每张卡的输入长度、并发和测法",
                    1,
                )
                return self._send(200, page, "text/html; charset=utf-8")
            if path == "/benchmark":
                page = tf500_nav.decorate(tf500_benchmark.page_html(), "benchmark")
                return self._send(200, page, "text/html; charset=utf-8")
            if path == "/proxy":
                page = tf500_nav.decorate(tf500_proxy_admin.page_html(), "proxy").encode("utf-8")
                self.send_response(200)
                self.send_header("Content-Type", "text/html; charset=utf-8")
                self.send_header("Content-Length", str(len(page)))
                self.send_header("Cache-Control", "no-store")
                self.send_header("X-Content-Type-Options", "nosniff")
                self.end_headers()
                return self.wfile.write(page)
            if path == "/server":
                page = tf500_nav.decorate(tf500_server_info.page_html(), "server")
                return self._send(200, page, "text/html; charset=utf-8")
            if path == "/api/server/status":
                data = tf500_server_info.snapshot()
                proxy = tf500_api_proxy.status()
                data["model_state"] = proxy["backend"]
                data["model_backend"] = proxy.get("backend_label")
                data["model_port"] = proxy.get("backend_port")
                return self._admin_json(data)
            if path == "/api/proxy/status":
                return self._json(tf500_api_proxy.status())
            if path == "/api/proxy/admin":
                return self._admin_json(tf500_proxy_admin.data())
            if path == "/api/proxy/selftest":
                return self._admin_json(tf500_proxy_admin.selftest())
            if path == "/api/benchmark/status":
                return self._json(benchmark.status())
            return super().do_GET()

        def do_POST(self):
            path = urlparse(self.path).path
            if path in ("/api/proxy/keys", "/api/proxy/keys/delete"):
                if self.headers.get("X-Requested-With") != "fetch":
                    return self._admin_json({"error": "forbidden"}, 403)
                origin = self.headers.get("Origin")
                if origin and urlparse(origin).netloc != self.headers.get("Host"):
                    return self._admin_json({"error": "forbidden origin"}, 403)
                try:
                    length = int(self.headers.get("Content-Length") or 0)
                    if length <= 0 or length > 8192:
                        raise ValueError("请求体大小无效")
                    form = parse_qs(self.rfile.read(length).decode("utf-8", "replace"))
                    if path.endswith("/delete"):
                        tf500_proxy_keys.delete((form.get("id") or [""])[0])
                        return self._admin_json({"ok": True})
                    item = tf500_proxy_keys.create(
                        (form.get("name") or [""])[0],
                        (form.get("days") or ["0"])[0],
                        (form.get("max_requests") or ["0"])[0],
                    )
                    return self._admin_json(item)
                except ValueError as exc:
                    return self._admin_json({"error": str(exc)}, 400)
                except OSError:
                    return self._admin_json({"error": "密钥文件操作失败"}, 500)
            if path not in ("/api/benchmark/start", "/api/benchmark/stop"):
                return super().do_POST()
            if self.headers.get("X-Requested-With") != "fetch":
                return self._json({"error": "forbidden"}, 403)
            try:
                length = int(self.headers.get("Content-Length") or 0)
                if length < 0 or length > 65536:
                    raise ValueError("请求体过大")
                form = parse_qs(self.rfile.read(length).decode("utf-8", "replace"))
                if path.endswith("/start"):
                    if jobs.job["state"] == "running":
                        raise ValueError("模型正在启动，请等就绪后测速")
                    return self._json(benchmark.start(form))
                return self._json(benchmark.stop((form.get("id") or [""])[0]))
            except ValueError as exc:
                return self._json({"error": str(exc)}, 400)
            except OSError as exc:
                return self._json({"error": f"测速操作失败：{exc.strerror or type(exc).__name__}"}, 500)

    return Handler


def main():
    console = Tf500Console()
    benchmark = tf500_benchmark.Benchmark()
    jobs = Tf500Jobs(console, benchmark)
    server = ThreadingHTTPServer(("127.0.0.1", 8100), make_tf500_handler(console, jobs, benchmark))
    print("XTier 86KD console on http://127.0.0.1:8100/", flush=True)
    server.serve_forever()


if __name__ == "__main__":
    main()
