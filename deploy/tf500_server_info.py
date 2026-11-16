"""Read-only hardware telemetry for the XTier 86KD console."""

import csv
import io
import os
import platform
import shutil
import subprocess
import threading
import time
from functools import lru_cache
from pathlib import Path


GPU_FIELDS = ("index", "name", "gpu_util", "memory_util", "memory_used_mib",
              "memory_total_mib", "power_w", "power_limit_w", "temp_c",
              "fan_pct", "sm_clock_mhz")
GPU_QUERY = ",".join(("index", "name", "utilization.gpu", "utilization.memory",
                      "memory.used", "memory.total", "power.draw", "power.limit",
                      "temperature.gpu", "fan.speed", "clocks.current.sm"))
_lock = threading.Lock()
_previous = {}


def _number(value):
    value = value.strip()
    if not value or value.upper() in {"N/A", "[N/A]", "NOT SUPPORTED"}:
        return None
    try:
        return float(value)
    except ValueError:
        return None


def parse_gpus(output):
    gpus = []
    for row in csv.reader(io.StringIO(output), skipinitialspace=True):
        if len(row) != len(GPU_FIELDS):
            continue
        gpu = dict(zip(GPU_FIELDS, row))
        try:
            gpu["index"] = int(gpu["index"].strip())
        except ValueError:
            continue
        gpu["name"] = gpu["name"].strip()
        for key in GPU_FIELDS[2:]:
            gpu[key] = _number(gpu[key])
        gpus.append(gpu)
    return sorted(gpus, key=lambda item: item["index"])


def _gpus():
    result = subprocess.run(
        ["nvidia-smi", f"--query-gpu={GPU_QUERY}", "--format=csv,noheader,nounits"],
        capture_output=True, text=True, timeout=5, check=True)
    return parse_gpus(result.stdout)


def _meminfo():
    values = {}
    for line in Path("/proc/meminfo").read_text().splitlines():
        name, _, value = line.partition(":")
        if name in {"MemTotal", "MemAvailable", "SwapTotal", "SwapFree"}:
            values[name] = int(value.split()[0]) * 1024
    return values


@lru_cache(maxsize=1)
def _cpu_model():
    for line in Path("/proc/cpuinfo").read_text().splitlines():
        if line.startswith("model name"):
            return line.partition(":")[2].strip()
    return platform.processor() or "未知"


def _cpu_counters():
    fields = Path("/proc/stat").read_text().splitlines()[0].split()[1:]
    values = [int(value) for value in fields]
    return sum(values), values[3] + (values[4] if len(values) > 4 else 0)


def _network_counters():
    rx = tx = 0
    for line in Path("/proc/net/dev").read_text().splitlines()[2:]:
        name, _, rest = line.partition(":")
        name = name.strip()
        if not (Path("/sys/class/net") / name / "device").exists():
            continue
        fields = rest.split()
        if len(fields) >= 16:
            rx += int(fields[0])
            tx += int(fields[8])
    return rx, tx


def _rate(key, counters, now):
    old = _previous.get(key)
    _previous[key] = (now, counters)
    if old is None:
        return None
    elapsed = now - old[0]
    if elapsed <= 0:
        return None
    if key == "cpu":
        total = counters[0] - old[1][0]
        idle = counters[1] - old[1][1]
        return round(100 * (total - idle) / total, 1) if total > 0 else None
    return [round(max(0, counters[i] - old[1][i]) / elapsed, 1) for i in range(2)]


def snapshot():
    now = time.monotonic()
    with _lock:
        cpu_pct = _rate("cpu", _cpu_counters(), now)
        network = _rate("network", _network_counters(), now)
    mem = _meminfo()
    disk = shutil.disk_usage("/data")
    try:
        gpus = _gpus()
        gpu_error = None
    except (OSError, subprocess.SubprocessError, ValueError) as exc:
        gpus, gpu_error = [], str(exc)[:200]
    uptime = float(Path("/proc/uptime").read_text().split()[0])
    load = os.getloadavg()
    values = lambda field: [gpu[field] for gpu in gpus if gpu[field] is not None]
    power = values("power_w")
    gpu_util = values("gpu_util")
    power_limit = values("power_limit_w")
    return {
        "timestamp": time.time(), "gpu_count": len(gpus), "gpus": gpus,
        "gpu_error": gpu_error,
        "gpu": {
            "util_avg_pct": round(sum(gpu_util) / len(gpu_util), 1) if gpu_util else None,
            "memory_used_mib": sum(values("memory_used_mib")),
            "memory_total_mib": sum(values("memory_total_mib")),
            "power_w": round(sum(power), 1) if power else None,
            "power_limit_w": round(sum(power_limit), 1) if power_limit else None,
            "peak_temp_c": max(values("temp_c")) if values("temp_c") else None,
        },
        "cpu": {"util_pct": cpu_pct, "logical_cores": os.cpu_count(),
                "model": _cpu_model(), "load_1m": round(load[0], 2),
                "load_5m": round(load[1], 2), "load_15m": round(load[2], 2)},
        "memory": {"total_bytes": mem["MemTotal"], "used_bytes": mem["MemTotal"] - mem["MemAvailable"],
                   "available_bytes": mem["MemAvailable"], "swap_total_bytes": mem.get("SwapTotal", 0),
                   "swap_used_bytes": mem.get("SwapTotal", 0) - mem.get("SwapFree", 0)},
        "disk": {"path": "/data", "total_bytes": disk.total, "used_bytes": disk.used,
                 "free_bytes": disk.free},
        "network": {"rx_bytes_per_s": network[0] if network else None,
                    "tx_bytes_per_s": network[1] if network else None},
        "system": {"os": platform.system(), "kernel": platform.release(),
                   "uptime_s": uptime},
    }


CSS = r"""
.hero-4{display:grid;grid-template-columns:repeat(4,minmax(0,1fr));gap:12px}
.gpu-grid{display:grid;grid-template-columns:repeat(4,minmax(0,1fr));gap:12px}
.meter{margin-top:10px}
.meter-row{display:flex;justify-content:space-between;gap:8px;font-size:12px;color:var(--muted)}
.meter-row b{color:var(--text);font-weight:600;font-variant-numeric:tabular-nums;font-family:var(--mono)}
.meter-pair{display:grid;grid-template-columns:1fr 1fr;gap:10px}
.gpu-foot{display:flex;justify-content:space-between;gap:8px;border-top:1px solid var(--border);padding-top:9px;margin-top:11px;color:var(--muted);font-size:11.5px}
.gpu-foot b{color:var(--text-2);font-family:var(--mono);font-weight:600}
.system-grid{display:grid;grid-template-columns:repeat(3,minmax(0,1fr));gap:12px}
.sys-kv{display:flex;justify-content:space-between;gap:12px;padding:7px 0;border-bottom:1px solid var(--border);font-size:12.5px}
.sys-kv:last-child{border-bottom:0}
.sys-kv span{color:var(--muted);white-space:nowrap}
.sys-kv b{text-align:right;font-weight:600;color:var(--text);overflow-wrap:anywhere}
.error-msg{color:var(--danger);font-size:13px;margin:8px 0}
.error-msg:empty{display:none}
.foot-note{margin-top:16px;color:var(--muted);font-size:12px}
@media(max-width:1100px){.gpu-grid{grid-template-columns:repeat(2,minmax(0,1fr))}.system-grid{grid-template-columns:repeat(2,minmax(0,1fr))}}
@media(max-width:680px){.hero-4{grid-template-columns:repeat(2,minmax(0,1fr))}.gpu-grid,.system-grid{grid-template-columns:1fr}}
"""

JS = r"""
(()=>{
const $=id=>document.getElementById(id),fmt=(n,d=0)=>Number.isFinite(n)?Number(n).toLocaleString('zh-CN',{maximumFractionDigits:d}):'—';
const pct=(n,total)=>Number.isFinite(n)&&total>0?Math.min(100,Math.max(0,100*n/total)):0;
const size=(bytes)=>Number.isFinite(bytes)?(bytes/1024**3>=1024?fmt(bytes/1024**4,1)+' TiB':fmt(bytes/1024**3,1)+' GiB'):'—';
const rate=(bytes)=>Number.isFinite(bytes)?(bytes>=1024**2?fmt(bytes/1024**2,1)+' MiB/s':fmt(bytes/1024,1)+' KiB/s'):'—';
const duration=(s)=>Number.isFinite(s)?Math.floor(s/86400)+' 天 '+Math.floor(s%86400/3600)+' 小时':'—';
const esc=(s)=>String(s??'').replace(/[&<>"']/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
function gpuCard(g){
  const util=g.gpu_util,mem=g.memory_used_mib,total=g.memory_total_mib,pow=g.power_w,limit=g.power_limit_w;
  const card=document.createElement('article');
  card.className='stat';
  const shortName=(g.name||'GPU').replace(/^NVIDIA\s+/,'');
  card.innerHTML=
    `<div class="stat-label"><span>GPU ${g.index}</span><span class="stat-hint" title="${esc(g.name)}">${esc(shortName)}</span></div>`+
    `<div class="stat-value">${fmt(util)}<small>%</small></div>`+
    `<div class="bar"><span style="width:${Math.min(100,Math.max(0,util||0))}%"></span></div>`+
    `<div class="meter"><div class="meter-row"><span>显存 ${(mem==null?'—':fmt(mem/1024,1))} / ${(total==null?'—':fmt(total/1024,1))} GiB</span><b>${fmt(pct(mem,total),0)}%</b></div>`+
    `<div class="bar gpu"><span style="width:${pct(mem,total)}%"></span></div></div>`+
    `<div class="meter-pair">`+
      `<div class="meter"><div class="meter-row"><span>温度</span><b>${fmt(g.temp_c)}°C</b></div><div class="bar hot"><span style="width:${pct(g.temp_c,100)}%"></span></div></div>`+
      `<div class="meter"><div class="meter-row"><span>功耗</span><b>${fmt(pow,0)} W</b></div><div class="bar hot"><span style="width:${pct(pow,limit||600)}%"></span></div></div>`+
    `</div>`+
    `<div class="gpu-foot"><span>显存控制器 <b>${fmt(g.memory_util)}%</b></span><span>SM <b>${fmt(g.sm_clock_mhz)} MHz</b></span></div>`;
  return card;
}
function render(d){
  const g=d.gpu,c=d.cpu,m=d.memory,k=d.disk,s=d.system,n=d.network;
  $('live').className=d.gpu_error?'status-pill danger':'status-pill ok';
  $('live-text').textContent=d.gpu_error?'GPU 读取异常':'每 5 秒刷新';
  $('gpu-count').textContent=d.gpu_count+' 张在线';
  $('gpu-util').textContent=fmt(g.util_avg_pct,1)+'%';
  $('gpu-power').textContent=fmt(g.power_w)+' W';
  $('gpu-power-sub').textContent='功耗上限 '+fmt(g.power_limit_w)+' W · 峰值温度 '+fmt(g.peak_temp_c)+'°C';
  $('gpu-mem').textContent=fmt(g.memory_used_mib/1024,1)+' GiB';
  $('gpu-mem-sub').textContent='总显存 '+fmt(g.memory_total_mib/1024,1)+' GiB';
  const grid=$('gpu-grid');grid.replaceChildren(...d.gpus.map(gpuCard));
  $('gpu-error').textContent=d.gpu_error||'';
  $('cpu-util').textContent=fmt(c.util_pct,1)+'%';
  $('cpu-panel-util').textContent=fmt(c.util_pct,1)+'%';
  $('cpu-model').textContent=c.model;
  $('cpu-cores').textContent=fmt(c.logical_cores)+' 线程';
  $('load').textContent=[c.load_1m,c.load_5m,c.load_15m].map(v=>fmt(v,2)).join(' / ');
  $('ram-used').textContent=size(m.used_bytes);
  $('ram-detail').textContent=size(m.total_bytes)+' 总量 · '+size(m.available_bytes)+' 可用';
  $('ram-bar').style.width=pct(m.used_bytes,m.total_bytes)+'%';
  $('swap').textContent=size(m.swap_used_bytes)+' / '+size(m.swap_total_bytes);
  $('disk-used').textContent=size(k.used_bytes);
  $('disk-detail').textContent=size(k.total_bytes)+' 总量 · '+size(k.free_bytes)+' 可用';
  $('disk-bar').style.width=pct(k.used_bytes,k.total_bytes)+'%';
  $('disk-pct').textContent=fmt(pct(k.used_bytes,k.total_bytes),1)+'%';
  $('net-rx').textContent=rate(n.rx_bytes_per_s);
  $('net-tx').textContent=rate(n.tx_bytes_per_s);
  $('uptime').textContent=duration(s.uptime_s);
  $('kernel').textContent=s.os+' '+s.kernel;
  $('model-state').textContent=({online:'在线',offline:'离线',loading:'加载中'})[d.model_state]||d.model_state||'—';
  $('model-state').className=d.model_state==='online'?'ok-text':'';
  $('model-backend').textContent=(d.model_backend||'—')+(d.model_port?' :'+d.model_port:'');
  $('updated').textContent=' · 更新于 '+new Date(d.timestamp*1000).toLocaleTimeString('zh-CN');
}
async function poll(){
  try{
    const r=await fetch('/api/server/status',{cache:'no-store'});
    if(!r.ok)throw Error('HTTP '+r.status);
    render(await r.json());
  }catch(e){
    $('live').className='status-pill danger';
    $('live-text').textContent='连接中断';
    $('gpu-error').textContent='读取实时状态失败：'+e.message;
  }finally{setTimeout(poll,5000)}
}
poll();
})();
"""


def page_html():
    return (
        "<!doctype html><html lang=\"zh-CN\"><head><meta charset=\"utf-8\">"
        "<meta name=\"viewport\" content=\"width=device-width,initial-scale=1\">"
        "<title>服务器信息 · TF500</title><style>" + CSS
        + "</style></head><body><main>"
        "<div class=\"page-head\"><div><h1>服务器信息</h1>"
        "<p>8 张 GPU、CPU、内存、存储与网络的实时运行概览</p></div>"
        "<div id=\"live\" class=\"status-pill\"><span id=\"live-text\">正在读取…</span></div></div>"
        "<section class=\"hero-4\" aria-label=\"核心指标\">"
        "<div class=\"stat\"><div class=\"stat-label\"><span>GPU 平均负载</span><span class=\"stat-hint\" id=\"gpu-count\">—</span></div><div class=\"stat-value\" id=\"gpu-util\">—</div><div class=\"stat-sub\">8 卡平均核心利用率</div></div>"
        "<div class=\"stat\"><div class=\"stat-label\"><span>GPU 总功耗</span><span class=\"stat-hint\">实时功率</span></div><div class=\"stat-value\" id=\"gpu-power\">—</div><div class=\"stat-sub\" id=\"gpu-power-sub\">—</div></div>"
        "<div class=\"stat\"><div class=\"stat-label\"><span>已用显存</span><span class=\"stat-hint\">8 卡合计</span></div><div class=\"stat-value\" id=\"gpu-mem\">—</div><div class=\"stat-sub\" id=\"gpu-mem-sub\">—</div></div>"
        "<div class=\"stat\"><div class=\"stat-label\"><span>CPU 负载</span><span class=\"stat-hint\">逻辑核平均</span></div><div class=\"stat-value\" id=\"cpu-util\">—</div><div class=\"stat-sub\">连续采样计算</div></div>"
        "</section>"
        "<div class=\"section-title\"><h2>GPU 逐卡状态</h2><small>核心负载 / 显存 / 温度 / 功耗 / 频率</small></div>"
        "<p id=\"gpu-error\" class=\"error-msg\"></p>"
        "<section id=\"gpu-grid\" class=\"gpu-grid\" aria-label=\"GPU逐卡状态\"></section>"
        "<div class=\"section-title\"><h2>主机资源与服务状态</h2><small>本机实时数据</small></div>"
        "<section class=\"system-grid\">"
        "<article class=\"stat\"><div class=\"stat-label\"><span>CPU</span><span class=\"stat-hint\" id=\"cpu-cores\">—</span></div><div class=\"stat-value\" id=\"cpu-panel-util\">—</div>"
        "<div style=\"margin-top:10px\"><div class=\"sys-kv\"><span>处理器</span><b id=\"cpu-model\">—</b></div>"
        "<div class=\"sys-kv\"><span>系统负载 1/5/15m</span><b id=\"load\">—</b></div></div></article>"
        "<article class=\"stat\"><div class=\"stat-label\"><span>主机内存</span><span class=\"stat-hint\">RAM</span></div><div class=\"stat-value\" id=\"ram-used\">—</div><div class=\"stat-sub\" id=\"ram-detail\">—</div>"
        "<div class=\"bar gpu\"><span id=\"ram-bar\"></span></div><div style=\"margin-top:8px\"><div class=\"sys-kv\"><span>Swap 已用 / 总量</span><b id=\"swap\">—</b></div></div></article>"
        "<article class=\"stat\"><div class=\"stat-label\"><span>数据盘 /data</span><span class=\"stat-hint\" id=\"disk-pct\">—</span></div><div class=\"stat-value\" id=\"disk-used\">—</div><div class=\"stat-sub\" id=\"disk-detail\">—</div>"
        "<div class=\"bar\"><span id=\"disk-bar\"></span></div></article>"
        "<article class=\"stat\"><div class=\"stat-label\"><span>物理网卡吞吐</span><span class=\"stat-hint\">实时速率</span></div>"
        "<div style=\"margin-top:8px\"><div class=\"sys-kv\"><span>接收速率 (RX)</span><b id=\"net-rx\">—</b></div>"
        "<div class=\"sys-kv\"><span>发送速率 (TX)</span><b id=\"net-tx\">—</b></div></div><p class=\"stat-sub\">首次刷新后显示差值速率</p></article>"
        "<article class=\"stat\"><div class=\"stat-label\"><span>操作系统</span><span class=\"stat-hint\">Host</span></div>"
        "<div style=\"margin-top:8px\"><div class=\"sys-kv\"><span>连续运行时间</span><b id=\"uptime\">—</b></div>"
        "<div class=\"sys-kv\"><span>系统内核</span><b id=\"kernel\">—</b></div></div></article>"
        "<article class=\"stat\"><div class=\"stat-label\"><span>推理服务探活</span><span class=\"stat-hint\">只读监控</span></div>"
        "<div style=\"margin-top:8px\"><div class=\"sys-kv\"><span>当前状态</span><b id=\"model-state\">—</b></div>"
        "<div class=\"sys-kv\"><span>代理接入目标</span><b id=\"model-backend\">—</b></div></div>"
        "<p class=\"stat-sub\">监控页仅展示状态，不操作任何模型进程</p></article>"
        "</section>"
        "<p class=\"foot-note\">CPU 使用率和网络速率由连续采样计算；设备未提供的可选传感器不显示<span id=\"updated\"></span></p>"
        "</main><script>" + JS + "</script></body></html>"
    )
