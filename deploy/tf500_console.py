"""XTier console UI for tf500's TP8 FP4-KV deployment.

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

from xtier.console import core
from xtier.console.__main__ import Jobs, make_handler
import tf500_console_ui as ui
import tf500_benchmark
import tf500_api_proxy
import tf500_proxy_admin
import tf500_proxy_keys
import tf500_nav
import tf500_server_info

HOST = "<TF500_IP>"
PORT = 30000
SERVICE = "sglang-ds41.service"
MODEL_ID = "ds41-86kd"
API_KEY_FILE = Path("/data/sglang-api-key")
STATE_DIR = Path("/data/xtier-86kd-console")
LOG_DIR = STATE_DIR / "logs"

MODEL = {
    'id': 'ds41-86kd',
    'order': 10,
    'name': 'DeepSeek V4.1 Flash 1017',
    'code': 'DSV4.1 · TP8 FP4 KV · MXMMA',
    'port': 30000,
    'served': 'DeepSeek-V4.1-Flash-1017',
    'spec': 'DSpark',
    'arch': 'TP8 · DP Attention 8 · DP LM Head · Humming MXMMA e3m4 · 86KD',
    'hardware': 'RTX PRO 6000D × 8 · tf500',
    'ready_timeout': 3600,
    'bench': {
        'kind': 'text',
        'dec': '4863 · 64路实测（窗口 4967）',
        'pre': '26478 · 64路实测（单发 10886）',
        'title': (
            '2026-11-17 8h 双引擎深度优化实测（sd-popt6-dp8sps，mem=0.85，tails=384，CHUNK=49152，'
            'FP8 DP Gatherv + 目标/草稿 MXMMA e3m4 pad(128,128) + 全形状 MXFP8 + DP8 SPS 表 + A1/2/3/7 + FP32-mid reader）：'
            '全局主 KV 113,569,792 token（+68.4%）；1×32K 冷 P 10674–10886 tok/s（4.6–4.7×）；'
            '64×32K 冷 P 26042–26106 tok/s（+27%）、共享前缀 D 4863 tok/s（窗口 4967，+15.4%，P50/P95 TPOT 8.89/9.40ms）；'
            '64×128K（命中99.79%）冷 P 26478 tok/s（+28.2%）、纯 D 窗口 3895 tok/s（输出 3760，47.2ms/步）；'
            '64路短提示 D 3183–3285 tok/s；3段 PPL 2.0708（+0.078%）；全程最低空闲 1996 MiB（1.949 GiB）。'
        ),
    },
    'options': [
        {'key': 'packs', 'label': '86KD 加速包', 'default': 'on', 'choices': [['on', '开']]},
        {'key': 'popt', 'label': '双引擎深度加速（Step 1–5）', 'default': 'on', 'choices': [['on', '开 · MXMMA e3m4 + FP8 Gather + 全形状 MXFP8'], ['off', '关 · 回退历史基线（W4A16）']]},
        {'key': 'spec', 'label': 'DSpark 投机解码', 'default': 'on', 'choices': [['on', '开 · draft 6']]},
        {'key': 'verify', 'label': 'DSpark 验证成本表', 'default': 'compact', 'choices': [['compact', '变长 · DP8 实测表（推荐）'], ['compact_legacy', '变长 · 旧版成本表（超短输入）']]},
        {'key': 'context', 'label': '上下文长度', 'default': '1048576', 'choices': [['1048576', '1M']]},
        {'key': 'running', 'label': '最大并行请求', 'default': '64', 'choices': [['64', '64']]},
        {'key': 'graph', 'label': 'Decode CUDA 图批量', 'default': '64', 'choices': [['64', '64']]},
        {'key': 'memory', 'label': '静态显存比例', 'default': '0.85', 'choices': [['0.85', '0.85']]},
        {'key': 'chunk', 'label': 'Prefill 分块', 'default': '49152', 'choices': [['49152', '48K · 巅峰吞吐（推荐）'], ['40960', '40K · 高余量稳健'], ['32768', '32K · 历史基线']]},
        {'key': 'interval', 'label': 'Prefill 中穿插 decode', 'default': '10', 'choices': [['10', '10 步']]},
        {'key': 'tails', 'label': 'SWA 前缀尾部预算', 'default': '384', 'choices': [['384', '384']]},
        {'key': 'kvpool', 'label': 'KV 逻辑总量（实测）', 'default': '113569792', 'choices': [['113569792', '113.57M · 启动时核实']]},
        {'key': 'align', 'label': 'DeepInfer 第一档对齐', 'default': 'on', 'choices': [['on', '开 · 开发验证（A1/A2/A3/A7）'], ['off', '关 · 开发基线（p0）']]},
    ],
    'notes': [
        {'when': {}, 'text': '启用 Step 1–5 深度加速后，Humming MoE 采用 SM120 原生 MXMMA float8e3m4 + pad(128,128)，每卡权重由 56.82 GiB 降至 51.84 GiB，全局压缩主 KV 容量达 113,569,792 token（每 DP 14,196,224，较旧基线 +68.4%）。'},
        {'when': {}, 'text': '实测单发 32K 冷 Prefill 达 10,674–10,886 tok/s（自动 16K 续块）；64×32K 冷 Prefill 26,042 tok/s、共享前缀 Decode 4,863 tok/s（窗口 4,967）；64×128K 冷 Prefill 26,478 tok/s、纯 Decode 窗口 3,895 tok/s（47.2 ms/步），最低空闲显存 1.949 GiB。'},
        {'when': {}, 'text': '可随时将「双引擎深度加速」或「DeepInfer 第一档对齐」切回「关」做无侵入回退对比。点击启动按钮才加载模型。'},
    ],
}


def _resolve_sps_table(popt_on: bool, verify_opt: str, deploy_dir: Path) -> str:
    if popt_on and verify_opt == "compact":
        for cand in (
            deploy_dir.parent / "bench" / "86kd-sps-6000d-dp8.json",
            Path("/data/bench/align/86kd-sps-6000d-dp8.json"),
        ):
            if cand.is_file():
                return str(cand)
    return "/data/bench/86kd-sps-6000d.json"


def build(options, port):
    deploy_dir = Path(__file__).resolve().parent
    spec_on = options.get("spec", "on") == "on"
    popt_on = options.get("popt", "on") == "on"
    align_on = options.get("align", "on") == "on"
    verify_opt = options.get("verify", "compact")
    verify_mode = "compact" if verify_opt in ("compact", "compact_legacy") else verify_opt
    sps_table = _resolve_sps_table(popt_on, verify_opt, deploy_dir) if (spec_on and verify_mode == "compact") else ""

    env = {
        "XTIER_86KD": "1" if options.get("packs", "on") == "on" else "0",
        "USE_DSPARK": "1" if spec_on else "0",
        "SGLANG_RAGGED_VERIFY_MODE": verify_mode if spec_on else "static",
        "SPS_TABLE": sps_table,
        "CONTEXT_LENGTH": options["context"],
        "MAX_RUNNING": options["running"],
        "CUDA_GRAPH_MAX_BS_DECODE": options["graph"],
        "MEM_FRACTION_STATIC": options["memory"],
        "HOST": HOST,
        "PORT": str(port),
        "MOE_BACKEND": "humming",
        "API_KEY_FILE": str(API_KEY_FILE),
    }

    if popt_on:
        hook_dir = deploy_dir / "prefill-opt"
        if not (hook_dir / "sitecustomize.py").is_file():
            hook_dir = Path("/data/bench/align/prefill-opt-v1")
        py_path = f"{hook_dir}:/data/src/sglang-align/python"
        popt_env = {
            "XTIER_PREFILL_OPT": "1",
            "SGLANG_ENABLE_DP_GATHER_FP8": "1",
            "XTIER_PREFILL_NCCL_SKIP_ZERO": "1",
            "XTIER_IDLE_SINGLE_CHUNK": "16384",
            "XTIER_PREFILL_MOE_MXMMA": "e3m4",
            "XTIER_DRAFT_MOE_MXMMA": "e3m4",
            "XTIER_MOE_PAD_N": "128",
            "XTIER_MOE_PAD_K": "128",
            "XTIER_86KD_MXFP8_TABLE": "1",
            "XTIER_DS_WO_A_PREFILL_MXFP8": "0",
        }
    else:
        if align_on:
            hook_dir = deploy_dir / "repair-a2-reader"
            if not (hook_dir / "sitecustomize.py").is_file():
                hook_dir = Path("/data/bench/align/repair-a2-reader")
            py_path = f"{hook_dir}:/data/src/sglang-align/python"
        else:
            py_path = "/data/src/sglang-align/python"
        popt_env = {
            "XTIER_PREFILL_OPT": "0",
            "SGLANG_ENABLE_DP_GATHER_FP8": "0",
            "XTIER_86KD_MXFP8_TABLE": "0",
            "XTIER_DS_WO_A_MAX_T": "16",
        }

    env.update({
        "PYTHONPATH": py_path,
        "CHUNK": options["chunk"],
        "INTERVAL": options["interval"],
        "SGLANG_DP_USE_GATHERV": "1",
        "SGLANG_DP_USE_REDUCE_SCATTER": "1",
        "SGLANG_DSV4_KV_LAYOUT": "v41_new",
        "SGLANG_DSV4_COMPRESSED_KV_LAYOUT": "auto",
        "SGLANG_SM120_DSV41_KV": "1",
        "SGLANG_DI_ALIGN": "0",
        **{f"SGLANG_DI_ALIGN_A{i}": ("1" if align_on and i in (1, 2, 3, 7) else "0") for i in range(1, 9)},
        "SGLANG_REPAIR_A2_FP32MID": "1" if align_on else "0",
        **popt_env,
        "EXTRA_ARGS": "--attn-dp-size 8 --enable-dp-lm-head --min-free-slots-delay 1 --swa-prefix-tails " + options["tails"],
    })
    return env, ["/bin/bash", "/data/src/xtier_86kd/deploy/start-tp8-fp4-card.sh"]


def tags(options):
    verify_opt = options.get("verify", "compact")
    is_compact = options.get("spec", "on") == "on" and verify_opt in ("compact", "compact_legacy")
    res = ["TP8", "86KD" if options.get("packs") == "on" else "基线",
           "变长验证" if is_compact else "固定验证"]
    if options.get("popt", "on") == "on":
        res.append("MXMMA·双引擎加速")
    if options.get("align") == "on":
        res.append("DI第一档·开发验证")
    return res




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
        if model == MODEL_ID:
            return (b"start-sglang-plan.sh" in command or b"start-tp8-fp4-card.sh" in command or
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
        indices = set(range(8))
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
        }
        super().__init__(models, STATE_DIR, log_dir=LOG_DIR, runner="setsid", guard_gib=0,
                         cwd="/data/src/sglang-xtier")

    def describe(self, model, options):
        return super().describe(model, options)

    def read_state(self):
        saved = core.Console.read_state(self)
        if saved and self.active(saved):
            return saved
        for service, model in ((SERVICE, MODEL_ID),):
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
            service = state.get("service") or SERVICE
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
        service = state.get("service") or SERVICE
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
            "subtitle": "tf500 · TP8 / DP8 · FP4 KV · MXMMA · DeepSeek V4.1 Flash 1017",
            "foot": "默认搭载 8h 双引擎深度加速与 DP8 实测 SPS 成本表（sd-popt6-dp8sps 实测通过，支持下拉一键回退历史基线）。模型由启动按钮手动加载。",
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
            service = state.get("service") or SERVICE
            log = _journal_tail(service) if adopted else core._read_tail(state.get("log"))
            elapsed = round(time.time() - state.get("started_at", time.time()))
            body.update(started_at=state.get("started_at"), elapsed=elapsed,
                        options=state.get("options"), version="86KD")
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

        def _html_no_store(self, html_text):
            payload = html_text.encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(payload)))
            self.send_header("Cache-Control", "no-store")
            self.send_header("X-Content-Type-Options", "nosniff")
            self.end_headers()
            return self.wfile.write(payload)

        def do_GET(self):
            path = urlparse(self.path).path
            if path == "/":
                data = {**console.console_data(), "status": console.status()}
                page = tf500_nav.decorate(ui.page_html(data, title="启动台 · TF500"), "console")
                return self._html_no_store(page)
            if path == "/benchmark":
                return self._html_no_store(tf500_nav.decorate(tf500_benchmark.page_html(), "benchmark"))
            if path == "/proxy":
                return self._html_no_store(tf500_nav.decorate(tf500_proxy_admin.page_html(), "proxy"))
            if path == "/server":
                return self._html_no_store(tf500_nav.decorate(tf500_server_info.page_html(), "server"))
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
            if path == "/api/benchmark/plan":
                try:
                    form = parse_qs(urlparse(self.path).query)
                    return self._json(benchmark.plan_preview(form))
                except ValueError as exc:
                    return self._json({"error": str(exc)}, 400)
            return super().do_GET()

        def do_POST(self):
            path = urlparse(self.path).path
            if path in ("/api/proxy/keys", "/api/proxy/keys/delete", "/api/proxy/master-key/copy"):
                if self.headers.get("X-Requested-With") != "fetch":
                    return self._admin_json({"error": "forbidden"}, 403)
                origin = self.headers.get("Origin")
                if origin and urlparse(origin).netloc != self.headers.get("Host"):
                    return self._admin_json({"error": "forbidden origin"}, 403)
                try:
                    if path == "/api/proxy/master-key/copy":
                        key = tf500_proxy_keys.MASTER_FILE.read_text(encoding="utf-8").strip()
                        if not key:
                            raise OSError("empty master key")
                        return self._admin_json({"key": key})
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
            if path not in ("/api/benchmark/start", "/api/benchmark/stop", "/api/benchmark/plan", "/api/benchmark/clear"):
                return super().do_POST()
            if self.headers.get("X-Requested-With") != "fetch":
                return self._json({"error": "forbidden"}, 403)
            try:
                length = int(self.headers.get("Content-Length") or 0)
                if length < 0 or length > 65536:
                    raise ValueError("请求体过大")
                form = parse_qs(self.rfile.read(length).decode("utf-8", "replace"))
                if path.endswith("/plan"):
                    return self._json(benchmark.plan_preview(form))
                if path.endswith("/clear"):
                    return self._json(benchmark.clear((form.get("id") or [""])[0]))
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


class ConsoleHTTPServer(ThreadingHTTPServer):
    request_queue_size = 128
    daemon_threads = True


def main():
    console = Tf500Console()
    benchmark = tf500_benchmark.Benchmark()
    jobs = Tf500Jobs(console, benchmark)
    server = ConsoleHTTPServer(("127.0.0.1", 8100), make_tf500_handler(console, jobs, benchmark))
    print("XTier 86KD console on http://127.0.0.1:8100/", flush=True)
    server.serve_forever()


if __name__ == "__main__":
    main()
