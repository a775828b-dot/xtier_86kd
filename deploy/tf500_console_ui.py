"""XTier Launch Console (xtier.console.ui) reskinned with the clean light/dark UI theme.

Preserves the exact 3-column XTier Launch Console layout, model rail, parameter
switches/dropdowns, circular progress/TPS ring, decode sparkline, GPU bar, log
tail, and JS controller, while replacing the neon/polygon theme with clean
surface/border/accent tokens matching the rest of the WebUI.
"""

from __future__ import annotations

import html
import json

SUBTITLE = (
    "官方 SGLang main + XTier 插件 · 专家分布在显存 / 内存两层，NVMe 只放大嵌入表（Engram / PLE） · "
    "每个模型在自己的端口上提供 OpenAI 兼容接口"
)
FOOT = "默认参数 = configs/*/console.py 里的实测配置，改动过的项会标出。一次只跑一个模型：启动新的模型前会先停掉当前的。"

CSS = r"""
.xt{
  position:relative;container:xt/inline-size;margin:4px 0 18px;
  border-radius:var(--radius);overflow:hidden;
  background:var(--surface);color:var(--text);text-align:left;
  border:1px solid var(--border);box-shadow:var(--shadow);
  font:14px/1.55 var(--sans);
}
.xt *{box-sizing:border-box}
.xt-mono,.xt code,.xt-num{font-family:var(--mono);font-variant-numeric:tabular-nums}

.xt-head{
  display:flex;justify-content:space-between;align-items:center;gap:16px;
  padding:18px 22px 15px;border-bottom:1px solid var(--border);
  background:var(--surface);
}
.xt-brand{display:flex;align-items:center;gap:14px;min-width:0}
.xt-logo{width:42px;height:42px;flex:none}
.xt-logo .spin{transform-origin:24px 24px;animation:xt-rot 18s linear infinite}
@keyframes xt-rot{to{transform:rotate(360deg)}}
.xt-title{font-weight:750;letter-spacing:.08em;font-size:18px;line-height:1.2;color:var(--text)}
.xt-title span{display:inline-block;margin-left:9px;font-size:11px;font-weight:600;letter-spacing:.16em;color:var(--muted)}
.xt-sub{margin-top:3px;color:var(--muted);font-size:12.5px}

.xt-state{
  display:flex;align-items:center;gap:9px;padding:6px 14px;
  border:1px solid var(--border);border-radius:999px;
  background:var(--surface-2);white-space:nowrap;min-width:0;
}
.xt-state b{letter-spacing:.1em;font-size:12.5px;color:var(--text-2)}
.xt-state small{color:var(--muted);overflow:hidden;text-overflow:ellipsis;font-size:12px}
.xt-led{flex:none;width:8px;height:8px;border-radius:50%;background:var(--muted)}
.xt[data-state=online] .xt-state{background:var(--ok-soft);border-color:color-mix(in srgb,var(--ok) 32%,var(--border))}
.xt[data-state=online] .xt-led{background:var(--ok);box-shadow:0 0 0 3px color-mix(in srgb,var(--ok) 22%,transparent);animation:xt-pulse 2.2s ease-in-out infinite}
.xt[data-state=loading] .xt-state{background:var(--warn-soft);border-color:color-mix(in srgb,var(--warn) 32%,var(--border))}
.xt[data-state=loading] .xt-led{background:var(--warn);animation:xt-blink 1s steps(2) infinite}
.xt[data-state=failed] .xt-state{background:var(--danger-soft);border-color:color-mix(in srgb,var(--danger) 32%,var(--border))}
.xt[data-state=failed] .xt-led{background:var(--danger)}
.xt[data-state=online] .xt-state b{color:var(--ok)}
.xt[data-state=loading] .xt-state b{color:var(--warn)}
.xt[data-state=failed] .xt-state b{color:var(--danger)}
@keyframes xt-pulse{50%{box-shadow:0 0 0 6px color-mix(in srgb,var(--ok) 0%,transparent)}}
@keyframes xt-blink{50%{opacity:.4}}

.xt-body{display:grid;grid-template-columns:minmax(230px,274px) minmax(0,1fr) minmax(0,1fr);gap:14px;padding:18px 22px 20px}
.xt-panel{
  position:relative;min-width:0;padding:15px 16px;
  border:1px solid var(--border);border-radius:10px;
  background:var(--surface-2);
}
.xt-panel h4{margin:0 0 10px;font-size:12px;line-height:1.5;letter-spacing:.08em;color:var(--text-2);font-weight:700}
.xt-panel h4::before{content:"";display:inline-block;width:3px;height:12px;border-radius:2px;margin-right:7px;vertical-align:-1px;background:var(--accent)}

.xt-rail{padding:15px 13px 13px}
.xt-legend{margin:-4px 0 10px;font-size:11px;color:var(--muted);line-height:1.45}
.xt-models{
  display:grid;grid-template-columns:minmax(0,1fr);gap:8px;align-content:start;
  max-height:420px;overflow-y:auto;padding:1px;scrollbar-width:thin;
}
.xt-model{
  all:unset;cursor:pointer;position:relative;display:block;
  border-radius:9px;border:1px solid var(--border);
  background:var(--surface);transition:border-color .15s,background-color .15s,box-shadow .15s;
}
.xt-model>div{padding:9px 12px 10px;border-radius:inherit}
.xt .xt-model:hover{border-color:var(--border-strong);background:var(--surface-3);filter:none}
.xt-model:focus-visible{outline:2px solid var(--accent);outline-offset:2px}
.xt-model[aria-selected=true]{
  border-color:var(--accent);background:var(--accent-soft);
  box-shadow:0 0 0 1px color-mix(in srgb,var(--accent) 24%,transparent);
}
.xt-model .top{display:flex;align-items:center;justify-content:space-between;gap:6px;min-width:0}
.xt-model .code{min-width:0;font-size:10.5px;font-weight:600;letter-spacing:.06em;color:var(--accent-text);white-space:nowrap;overflow:hidden;text-overflow:ellipsis}
.xt-model .name{margin-top:2px;font-weight:700;font-size:13.5px;line-height:1.35;color:var(--text);white-space:nowrap;overflow:hidden;text-overflow:ellipsis}
.xt-model .perf{display:grid;grid-template-columns:34px minmax(0,1fr) auto;margin-top:5px;font-size:11.5px;line-height:1.45}
.xt-model .perf span{font-size:10px;letter-spacing:.06em;color:var(--muted);align-self:center}
.xt-model .perf b{font-weight:600;color:var(--text-2);white-space:nowrap}
.xt-model .perf .kind{
  grid-column:3;grid-row:1/span 2;align-self:center;font-style:normal;
  font-size:9.5px;font-weight:600;line-height:1.5;letter-spacing:.06em;
  padding:1px 6px;border-radius:5px;border:1px solid var(--border-strong);
  background:var(--surface);color:var(--muted);
}
.xt-model .perf .kind[data-kind=agent]{color:var(--ok);background:var(--ok-soft);border-color:color-mix(in srgb,var(--ok) 35%,var(--border))}
.xt-model .run{
  flex:none;display:none;font-size:9.5px;font-weight:700;line-height:1.5;letter-spacing:.08em;
  padding:1px 7px;border-radius:999px;color:var(--ok);background:var(--ok-soft);
  border:1px solid color-mix(in srgb,var(--ok) 35%,var(--border));
}
.xt-model[data-running=loading] .run{
  display:block;color:var(--warn);background:var(--warn-soft);
  border-color:color-mix(in srgb,var(--warn) 35%,var(--border));
}
.xt-model[data-running=online] .run{display:block}

.xt-sel{display:flex;flex-wrap:wrap;align-items:baseline;gap:3px 10px;margin:-2px 0 12px;padding-bottom:10px;border-bottom:1px dashed var(--border-strong)}
.xt-sel b{font-size:15px;color:var(--text)}
.xt-sel span{font-size:12px;color:var(--muted)}
.xt-sel .hw{font-family:var(--mono);font-size:11.5px;color:var(--accent-text);font-weight:600}

.xt-grid{display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:10px 12px}
.xt-field{display:flex;flex-direction:column;gap:4px;font-size:12px;color:var(--muted);font-weight:550}
.xt-field select{
  appearance:none;-webkit-appearance:none;width:100%;margin:0;padding:7px 28px 7px 10px;
  color:var(--text);font-family:inherit;font-size:13px;line-height:1.35;
  border-radius:8px;border:1px solid var(--border-strong);
  background:var(--surface) url("data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' width='12' height='8'%3E%3Cpath d='M1 1.5l5 5 5-5' fill='none' stroke='%236b7280' stroke-width='1.6' stroke-linecap='round' stroke-linejoin='round'/%3E%3C/svg%3E") no-repeat right 10px center;
  transition:border-color .15s,box-shadow .15s;
}
.xt-field select:hover:not(:disabled){border-color:var(--accent)}
.xt-field select:focus{outline:none;border-color:var(--accent);box-shadow:0 0 0 3px color-mix(in srgb,var(--accent) 20%,transparent)}
.xt-field select:disabled{opacity:.45;cursor:not-allowed}
.xt-field.changed select{border-color:var(--accent);background-color:color-mix(in srgb,var(--accent-soft) 45%,var(--surface))}
.xt-field.changed::after{content:"已改动 · 非实测默认";font-size:10.5px;color:var(--accent-text);font-weight:600}

.xt-switch .sw{
  position:relative;display:flex;align-items:center;gap:10px;min-height:35px;
  padding:6px 10px;border-radius:8px;border:1px solid var(--border-strong);
  background:var(--surface);cursor:pointer;transition:border-color .15s,box-shadow .15s;
}
.xt-switch .sw:hover{border-color:var(--accent)}
.xt-switch input{position:absolute;opacity:0;width:1px;height:1px;margin:0;pointer-events:none}
.xt-switch i{
  position:relative;flex:none;width:36px;height:20px;border-radius:999px;
  background:var(--border-strong);transition:background-color .18s;
}
.xt-switch i::after{
  content:"";position:absolute;top:3px;left:3px;width:14px;height:14px;border-radius:50%;
  background:#ffffff;box-shadow:0 1px 2px rgba(0,0,0,.25);transition:transform .18s;
}
.xt-switch input:checked+i{background:var(--accent)}
.xt-switch input:checked+i::after{transform:translateX(16px)}
.xt-switch input:focus-visible+i{outline:2px solid var(--accent);outline-offset:2px}
.xt-switch em{font-style:normal;font-size:13px;line-height:1.3;color:var(--text);font-weight:500}
.xt-switch input:disabled~i,.xt-switch input:disabled~em{opacity:.45}
.xt-switch .sw:has(input:disabled){cursor:not-allowed}
.xt-field.changed .sw{border-color:var(--accent);background:color-mix(in srgb,var(--accent-soft) 45%,var(--surface))}

.xt-notes{
  margin-top:11px;padding:8px 11px;border-radius:8px;
  background:var(--surface);border:1px solid var(--border);
  font-size:12px;color:var(--muted);line-height:1.5;
}
.xt-notes b{color:var(--warn);font-weight:650}

.xt-actions{display:flex;flex-wrap:wrap;align-items:center;gap:8px;margin-top:14px}
.xt-btn{
  all:unset;cursor:pointer;display:inline-flex;align-items:center;justify-content:center;gap:8px;
  padding:8px 15px;font-size:13px;font-weight:600;border-radius:8px;
  border:1px solid var(--border-strong);color:var(--text);background:var(--surface);
  transition:background-color .15s,border-color .15s,filter .15s;
}
.xt .xt-btn:hover{background:var(--surface-3);filter:none}
.xt-btn:focus-visible{outline:2px solid var(--accent);outline-offset:2px}
.xt-btn[disabled]{opacity:.45;pointer-events:none}
.xt-engage{
  padding:9px 20px;font-size:13.5px;font-weight:650;letter-spacing:.03em;
  color:#ffffff;border-color:transparent;background:var(--accent);
}
.xt .xt-engage:hover{background:var(--accent);filter:brightness(1.08);color:#ffffff}
.xt-abort{
  color:var(--danger);border-color:color-mix(in srgb,var(--danger) 40%,var(--border));
  background:var(--surface);
}
.xt .xt-abort:hover{background:var(--danger-soft);border-color:var(--danger);color:var(--danger);filter:none}

.xt-hud{display:grid;grid-template-columns:144px minmax(0,1fr);gap:14px;align-items:center}
.xt-hud>div{min-width:0}
.xt-ring{position:relative;width:144px;height:144px}
.xt-ring svg{width:144px;height:144px;transform:rotate(-90deg)}
.xt-ring .track{fill:none;stroke:var(--border);stroke-width:8}
.xt-ring .ticks{fill:none;stroke:var(--border-strong);stroke-width:2;stroke-dasharray:1 7.6}
.xt-ring .bar{fill:none;stroke:var(--accent);stroke-width:8;stroke-linecap:round;transition:stroke-dashoffset .7s ease}
.xt-ring .core{position:absolute;inset:0;display:flex;flex-direction:column;align-items:center;justify-content:center;text-align:center}
.xt-ring .core b{font-size:27px;line-height:1;color:var(--text);font-weight:750}
.xt-ring .core span{margin-top:5px;font-size:10.5px;font-weight:600;letter-spacing:.14em;color:var(--muted)}
.xt[data-state=online] .xt-ring .bar{stroke:var(--ok)}

.xt-phases{display:flex;flex-wrap:wrap;gap:5px;margin:0 0 10px;padding:0;list-style:none}
.xt-phases li{
  margin:0;font-size:11px;font-weight:550;padding:2px 7px;border-radius:6px;
  border:1px solid var(--border);background:var(--surface);color:var(--muted);white-space:nowrap;
}
.xt-phases li.done{color:var(--accent-text);background:var(--accent-soft);border-color:color-mix(in srgb,var(--accent) 30%,var(--border))}
.xt-phases li.now{color:var(--warn);background:var(--warn-soft);border-color:color-mix(in srgb,var(--warn) 40%,var(--border));font-weight:650}
.xt[data-state=online] .xt-phases li.now{color:var(--ok);background:var(--ok-soft);border-color:color-mix(in srgb,var(--ok) 40%,var(--border))}

.xt-meters{display:grid;grid-template-columns:repeat(3,minmax(0,1fr));gap:8px}
.xt-meter{padding:8px 10px;border:1px solid var(--border);border-radius:8px;background:var(--surface)}
.xt-meter span{display:block;font-size:11px;color:var(--muted)}
.xt-meter b{display:block;margin-top:2px;font-size:16px;color:var(--text);font-weight:700}
.xt-meter b small{font-size:11px;color:var(--muted);font-weight:500;margin-left:2px}

.xt-spark{margin-top:10px;height:46px;border:1px solid var(--border);border-radius:8px;background:var(--surface);overflow:hidden}
.xt-spark svg{width:100%;height:100%;display:block}
.xt-spark .ln{fill:none;stroke:var(--accent);stroke-width:1.8}
.xt-spark .ar{fill:color-mix(in srgb,var(--accent) 14%,transparent);stroke:none}

.xt-gpu{margin-top:10px;font-size:12px;color:var(--muted)}
.xt-gpu .bar{height:6px;margin-top:5px;border-radius:999px;background:var(--surface-3);overflow:hidden}
.xt-gpu .bar i{display:block;height:100%;width:0;background:var(--bar-gpu);border-radius:inherit;transition:width .5s ease}

.xt-end{display:flex;flex-wrap:wrap;align-items:center;gap:6px 8px;margin-top:12px;font-size:12px;color:var(--muted)}
.xt-end code{padding:2px 7px;border-radius:6px;border:1px solid var(--border);background:var(--surface);color:var(--text);font-size:12px;overflow-wrap:anywhere}

.xt-log{
  margin-top:10px;padding:9px 11px;max-height:106px;overflow:hidden;
  border-radius:8px;border:1px solid var(--border);
  background:var(--pre-bg);font-size:11.5px;line-height:1.55;color:var(--pre-text);
}
.xt-log div{white-space:nowrap;overflow:hidden;text-overflow:ellipsis}
.xt-log div:last-child::after{content:"▋";margin-left:3px;color:var(--accent);animation:xt-blink 1s steps(2) infinite}

.xt-msg{margin-top:10px;font-size:12.5px;color:var(--muted)}
.xt-msg:empty{display:none}
.xt-msg.err{color:var(--danger);font-weight:600}
.xt-msg.ok{color:var(--ok);font-weight:600}
.xt-foot{padding:0 22px 16px;font-size:12px;color:var(--muted)}

@container xt (max-width:1199px){
  .xt-body{grid-template-columns:minmax(0,1.1fr) minmax(0,1fr)}
  .xt-rail{grid-column:1/-1}
  .xt-models{grid-template-columns:repeat(auto-fill,minmax(220px,1fr));max-height:none}
}
@container xt (max-width:819px){.xt-body{grid-template-columns:minmax(0,1fr)}}
@container xt (max-width:559px){
  .xt-head{flex-direction:column;align-items:flex-start;padding:16px}
  .xt-state{max-width:100%}
  .xt-body{padding-left:12px;padding-right:12px}
  .xt-models,.xt-grid{grid-template-columns:minmax(0,1fr)}
  .xt-hud{grid-template-columns:minmax(0,1fr);justify-items:center}
  .xt-hud>div+div{width:100%}
  .xt-title span{display:block;margin:4px 0 0}
  .xt-actions .xt-btn{flex:1 1 auto}
  .xt-foot{padding:0 14px 14px}
}
@media (prefers-reduced-motion:reduce){
  .xt-logo .spin,.xt-led,.xt-log div:last-child::after{animation:none!important}
}
"""

LOGO = (
    "<svg class=xt-logo viewBox='0 0 48 48' aria-hidden=true>"
    "<defs><linearGradient id=xt-lg x1=0 y1=0 x2=1 y2=1>"
    "<stop offset=0 stop-color='#2563eb'/><stop offset=1 stop-color='#7c3aed'/>"
    "</linearGradient></defs>"
    "<polygon points='24,3 42,13.5 42,34.5 24,45 6,34.5 6,13.5' fill='rgba(37,99,235,.08)' stroke='url(#xt-lg)' stroke-width='2'/>"
    "<g class=spin><circle cx=24 cy=24 r=13 fill=none stroke='#2563eb' stroke-width='1.3' stroke-dasharray='3 4' opacity='.75'/></g>"
    "<path d='M16 16l16 16M32 16L16 32' stroke='url(#xt-lg)' stroke-width='3.2' stroke-linecap=round/>"
    "<circle cx=24 cy=24 r=3 fill='#2563eb'/></svg>"
)

PHASES = [
    ("boot", "BOOT"),
    ("weights", "权重"),
    ("draft", "草稿"),
    ("pool", "专家池"),
    ("graphs", "CUDA 图"),
    ("online", "ONLINE"),
]


def card_html(data: dict) -> str:
    texts = data.get("texts") or {}
    payload = html.escape(json.dumps(data, ensure_ascii=False), quote=True)
    phases = "".join(f"<li data-phase={key}>{label}</li>" for key, label in PHASES)
    return f"""<section class=xt id=xtier-console data-state=offline data-xtier="{payload}" aria-label="XTier 启动控制台">
<header class=xt-head>
  <div class=xt-brand>{LOGO}<div><div class=xt-title>XTIER<span>HYBRID MoE CONSOLE</span></div>
  <div class=xt-sub>{html.escape(texts.get('subtitle', SUBTITLE))}</div></div></div>
  <div class=xt-state><span class=xt-led></span><b id=xt-state>OFFLINE</b><small id=xt-state-sub>待命</small></div>
</header>
<div class=xt-body>
  <div class="xt-panel xt-rail" role=group aria-label="XTier 模型">
    <h4>MODELS · 模型</h4>
    <div class=xt-legend>README 实测 tok/s · 悬停查看每张卡的输入长度、并发和测法</div>
    <div class=xt-models role=tablist></div>
  </div>
  <form class=xt-panel id=xt-form autocomplete=off>
    <h4>LAUNCH PARAMETERS · 启动参数</h4>
    <div class=xt-sel id=xt-sel></div>
    <div class=xt-grid id=xt-options></div>
    <div class=xt-notes id=xt-notes></div>
    <div class=xt-actions>
      <button type=submit class="xt-btn xt-engage" id=xt-engage>启动 · ENGAGE</button>
      <button type=button class="xt-btn xt-abort" id=xt-abort disabled>停止 · ABORT</button>
      <button type=button class=xt-btn id=xt-reset>恢复实测默认</button>
    </div>
    <div class=xt-msg id=xt-msg aria-live=polite></div>
  </form>
  <div class=xt-panel>
    <h4>TELEMETRY · 运行状态</h4>
    <div class=xt-hud>
      <div class=xt-ring>
        <svg viewBox='0 0 150 150' aria-hidden=true>
        <circle class=track cx=75 cy=75 r=62 /><circle class=ticks cx=75 cy=75 r=71 /><circle class=bar id=xt-arc cx=75 cy=75 r=62 stroke-dasharray='389.6' stroke-dashoffset='389.6' /></svg>
        <div class=core><b class=xt-num id=xt-core>--</b><span id=xt-core-sub>STANDBY</span></div>
      </div>
      <div>
        <ul class=xt-phases id=xt-phases>{phases}</ul>
        <div class=xt-meters>
          <div class=xt-meter><span>DECODE</span><b class=xt-num id=xt-tps>--</b></div>
          <div class=xt-meter><span>接受长度</span><b class=xt-num id=xt-accept>--</b></div>
          <div class=xt-meter><span>运行请求</span><b class=xt-num id=xt-running>--</b></div>
        </div>
        <div class=xt-spark><svg viewBox='0 0 200 46' preserveAspectRatio=none id=xt-spark><path class=ar d=''/><path class=ln d=''/></svg></div>
      </div>
    </div>
    <div class=xt-gpu><span id=xt-gpu-text>GPU --</span><div class=bar><i id=xt-gpu-bar></i></div></div>
    <div class=xt-end id=xt-end></div>
    <div class="xt-log xt-mono" id=xt-log></div>
  </div>
</div>
<div class=xt-foot>{html.escape(texts.get('foot', FOOT))}</div>
</section>"""


JS = r"""
(() => {
  const root = document.getElementById('xtier-console');
  if (!root) return;
  const data = JSON.parse(root.dataset.xtier);
  const T = data.texts || {};
  const API = Object.assign({start: 'api/start', stop: 'api/stop', status: 'api/status', job: 'api/job'}, data.api || {});
  const hooks = window.xtierConsoleHooks || {};
  const $ = id => document.getElementById(id);
  const esc = s => String(s ?? '').replace(/[&<>"']/g, c => ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
  const fill = (tpl, vals, html) => String(tpl).replace(/\{(\w+)\}/g, (_, k) => html ? esc(vals[k]) : String(vals[k] ?? ''));
  const url = p => { const u = new URL(p, location.href); u.username = ''; u.password = ''; return u.href; };
  const post = (p, body) => fetch(url(p), {method: 'POST', headers: {'Accept': 'application/json', 'X-Requested-With': 'fetch'}, body});
  const matches = (cond, v) => Object.entries(cond || {}).every(([k, x]) => v[k] === x);
  const models = Object.keys(data.models);
  const chosen = {};
  for (const m of models) chosen[m] = {...data.defaults[m], ...(data.saved[m] || {})};
  let model = data.saved.model && data.models[data.saved.model] ? data.saved.model : models[0];
  let live = {state: 'offline'};
  let busy = false;

  const tabs = root.querySelector('.xt-models');
  tabs.innerHTML = models.map(m => {
    const d = data.models[m], b = d.bench || {};
    const kind = {agent: 'AGENT', text: '文本'}[b.kind] || '';
    const perf = b.dec || b.pre ? `<div class="perf xt-mono"><span>DEC</span><b>${esc(b.dec || '--')}</b>` +
      (kind ? `<i class=kind data-kind="${esc(b.kind)}">${kind}</i>` : '') + `<span>PRE</span><b>${esc(b.pre || '--')}</b></div>` : '';
    const tip = [d.arch, b.title].filter(Boolean).join('\n');
    return `<button type=button class=xt-model role=tab data-model="${esc(m)}" aria-selected=false title="${esc(tip)}"><div>
      <div class=top><span class="code xt-mono">${esc(d.code || m)} · :${esc(d.port)}</span><span class="run xt-mono"></span></div>
      <div class=name>${esc(d.name)}</div>${perf}</div></button>`;
  }).join('');
  tabs.addEventListener('click', e => {
    const t = e.target.closest('.xt-model');
    if (!t) return;
    model = t.dataset.model;
    paintTabs(); renderOptions();
  });

  function paintTabs() {
    for (const t of tabs.querySelectorAll('.xt-model')) {
      const here = live.model === t.dataset.model && (live.state === 'online' || live.state === 'loading');
      t.setAttribute('aria-selected', String(t.dataset.model === model));
      t.dataset.running = here ? live.state : '';
      t.querySelector('.run').textContent = here ? (live.state === 'online' ? 'RUNNING' : 'LOADING') : '';
    }
  }

  function renderOptions() {
    const d = data.models[model];
    $('xt-sel').innerHTML = `<b>${esc(d.name)}</b><span>${esc(d.arch || '')}</span>` +
      `<span class=hw>${d.hardware ? esc(d.hardware) + ' · ' : ''}:${esc(d.port)}</span>`;
    const host = $('xt-options');
    host.innerHTML = data.options[model].map(o => {
      if (o.switch) {
        const on = chosen[model][o.key] === o.choices[0][0];
        return `<label class="xt-field xt-switch" data-key="${esc(o.key)}" title="${esc(o.hint || '')}">${esc(o.label)}` +
          `<span class=sw><input type=checkbox role=switch name="${esc(o.key)}"${on ? ' checked' : ''}><i></i>` +
          `<em>${esc(o.choices[on ? 0 : 1][1])}</em></span></label>`;
      }
      const un = o.unavailable || {};
      const opts = o.choices.map(([v, label]) => `<option value="${esc(v)}"${chosen[model][o.key] === v ? ' selected' : ''}` +
        `${un[v] ? ` disabled title="${esc(un[v])}"` : ''}>${esc(label)}${un[v] ? ' · 本机不可用' : ''}</option>`).join('');
      return `<label class=xt-field data-key="${esc(o.key)}" title="${esc(o.hint || '')}">${esc(o.label)}<select name="${esc(o.key)}">${opts}</select></label>`;
    }).join('');
    for (const sel of host.querySelectorAll('select')) sel.addEventListener('change', () => { chosen[model][sel.name] = sel.value; paintOptions(); });
    for (const box of host.querySelectorAll('.xt-switch input')) box.addEventListener('change', () => {
      const c = data.options[model].find(o => o.key === box.name).choices[box.checked ? 0 : 1];
      chosen[model][box.name] = c[0];
      box.parentElement.querySelector('em').textContent = c[1];
      paintOptions();
    });
    paintOptions();
  }

  function paintOptions() {
    const v = chosen[model], d = data.defaults[model], notes = [];
    let changed = 0;
    for (const o of data.options[model]) {
      const f = $('xt-options').querySelector(`.xt-field[data-key="${o.key}"]`);
      const off = (o.off_when || []).some(c => matches(c, v));
      f.querySelector('select, input').disabled = off;
      f.classList.toggle('changed', !off && v[o.key] !== d[o.key]);
      if (!off && v[o.key] !== d[o.key]) changed++;
    }
    for (const n of data.models[model].notes || []) if (matches(n.when, v)) notes.push(n.warn ? `<b>${esc(n.text)}</b>` : esc(n.text));
    notes.push(changed ? `<b>${changed} 项不同于实测默认</b>` : '全部为实测默认配置');
    $('xt-notes').innerHTML = notes.join(' · ');
  }

  $('xt-reset').addEventListener('click', () => { chosen[model] = {...data.defaults[model]}; renderOptions(); });

  const msg = (text, kind = '') => { const el = $('xt-msg'); el.className = 'xt-msg ' + kind; el.textContent = text; };

  $('xt-form').addEventListener('submit', async e => {
    e.preventDefault();
    if (busy) return;
    const d = data.models[model];
    const ask = (hooks.confirmPrefix ? hooks.confirmPrefix() : '') +
      fill(T.confirm || '启动 XTier · {name}？\n会先停掉正在运行的 XTier 模型；就绪后在 127.0.0.1:{port} 提供 OpenAI 兼容接口（模型名 {served}）。', d, false);
    if (!confirm(ask)) return;
    busy = true; $('xt-engage').disabled = true;
    msg('正在提交启动任务…');
    try {
      const res = await post(API.start, new URLSearchParams({model, ...chosen[model]}));
      const out = await res.json().catch(() => ({ok: false, error: '接口返回非 JSON'}));
      if (!res.ok || !out.ok) throw new Error(out.error || `HTTP ${res.status}`);
      msg(`已开始：${out.label}。后台启动中，可以关闭页面。`, 'ok');
      watchJob(out.job);
    } catch (err) {
      msg('启动未提交：' + (err.message || err), 'err');
      busy = false; $('xt-engage').disabled = false;
    }
  });

  async function watchJob(job) {
    for (;;) {
      await new Promise(r => setTimeout(r, 2500));
      let st;
      try { st = await (await fetch(url(API.job), {cache: 'no-store'})).json(); } catch (err) { continue; }
      if (st.id !== job) break;
      if (st.state === 'running') continue;
      if (st.state === 'done') msg(st.message || '已就绪', 'ok'); else msg('启动未完成：' + (st.error || '未知错误'), 'err');
      break;
    }
    busy = false; poll();
  }

  $('xt-abort').addEventListener('click', async () => {
    if (!confirm('停止 XTier 服务？正在加载的启动也会中止。')) return;
    $('xt-abort').disabled = true;
    msg('正在停止…');
    try {
      const res = await post(API.stop);
      const out = await res.json().catch(() => ({ok: false, error: '接口返回非 JSON'}));
      msg(out.ok ? (out.message || '已停止') : ('停止失败：' + (out.error || res.status)), out.ok ? '' : 'err');
    } catch (err) { msg('停止失败：' + (err.message || err), 'err'); }
    poll();
  });

  function spark(values) {
    const svg = $('xt-spark');
    if (!values || values.length < 2) { svg.querySelector('.ln').setAttribute('d', ''); svg.querySelector('.ar').setAttribute('d', ''); return; }
    const hi = Math.max(...values), lo = Math.min(...values), span = Math.max(hi - lo, hi * 0.05, 1), n = values.length;
    const pts = values.map((v, i) => [i / (n - 1) * 200, 40 - (v - lo) / span * 32]);
    const line = 'M' + pts.map(p => p[0].toFixed(1) + ' ' + p[1].toFixed(1)).join(' L');
    svg.querySelector('.ln').setAttribute('d', line);
    svg.querySelector('.ar').setAttribute('d', line + ' L200 46 L0 46 Z');
  }

  function render(s) {
    live = s;
    root.dataset.state = s.state;
    const states = {offline: ['OFFLINE', '待命'], loading: ['BOOTING', s.phase_label || '启动中'], online: ['ONLINE', 'XTier 在线'], failed: ['FAULT', '上次启动失败']};
    const [big, sub] = states[s.state] || states.offline;
    $('xt-state').textContent = big;
    const mins = s.elapsed ? `${Math.floor(s.elapsed / 60)}:${String(s.elapsed % 60).padStart(2, '0')}` : '';
    $('xt-state-sub').textContent = s.state === 'offline' ? sub : `${data.models[s.model]?.name || ''} · ${sub}${mins ? ' · ' + mins : ''}`;
    const pct = s.state === 'online' ? 100 : (s.progress || 0);
    $('xt-arc').setAttribute('stroke-dashoffset', String(389.6 * (1 - pct / 100)));
    $('xt-core').textContent = s.state === 'online' ? (s.tps ? Math.round(s.tps) : 'ON') : (s.state === 'loading' ? pct + '%' : '--');
    $('xt-core-sub').textContent = s.state === 'online' ? (s.tps ? 'TOK / S' : 'READY') : (s.state === 'loading' ? 'LOADING' : (s.state === 'failed' ? 'FAULT' : 'STANDBY'));
    const order = ['boot', 'weights', 'draft', 'pool', 'graphs', 'online'];
    const at = order.indexOf(s.state === 'online' ? 'online' : (s.phase || ''));
    for (const li of $('xt-phases').children) {
      const i = order.indexOf(li.dataset.phase);
      li.className = at < 0 ? '' : (i < at ? 'done' : (i === at ? 'now' : ''));
    }
    $('xt-tps').innerHTML = typeof s.tps === 'number' ? `${s.tps.toFixed(1)}<small>tok/s</small>` : '--';
    $('xt-accept').textContent = s.accept ? s.accept.toFixed(2) : '--';
    $('xt-running').textContent = s.state === 'online' ? String(s.running ?? 0) : '--';
    spark(s.spark);
    const g = s.gpu || {};
    $('xt-gpu-text').textContent = g.mem_total ? `${g.name || 'GPU'}${g.count > 1 ? ' ×' + g.count : ''} · 显存 ${g.mem_used} / ${g.mem_total} GiB · 利用率 ${g.util}% · ${g.power} W` : 'GPU --';
    $('xt-gpu-bar').style.width = g.mem_total ? (g.mem_used / g.mem_total * 100).toFixed(1) + '%' : '0';
    if (s.state === 'online' || s.state === 'loading') {
      const opts = s.options ? Object.entries(s.options).filter(([k, v]) => data.defaults[s.model] && data.defaults[s.model][k] !== v).length : 0;
      $('xt-end').innerHTML = fill(T.endpoint || '<span>OpenAI 兼容接口</span><code>http://127.0.0.1:{port}/v1</code><span>模型名</span><code>{served}</code>', s, true) +
        (s.version ? `<span>XTier</span><code>${esc(s.version)}</code>` : '') + (s.load_s ? `<span>加载用时 ${esc(s.load_s)} s</span>` : '') +
        `<span>${opts ? opts + ' 项非默认参数' : '实测默认参数'}</span>`;
    } else if (s.state === 'failed') {
      $('xt-end').innerHTML = `<span>${esc(s.error || '')}</span>`;
    } else {
      $('xt-end').innerHTML = '<span>未运行 · 选择模型后点击启动</span>';
    }
    $('xt-log').innerHTML = (s.tail || []).map(l => `<div>${esc(l)}</div>`).join('') || '<div>— 等待指令 —</div>';
    const runningHere = s.state === 'loading' || s.state === 'online';
    $('xt-abort').disabled = !runningHere;
    $('xt-engage').disabled = busy || s.state === 'loading';
    $('xt-engage').textContent = runningHere && s.model === model ? '重启 · 应用参数' : '启动 · ENGAGE';
    paintTabs();
  }

  let timer = null;
  async function poll() {
    clearTimeout(timer);
    try {
      const res = await fetch(url(API.status), {cache: 'no-store'});
      if (res.ok) render(await res.json());
    } catch (err) { /* keep the last picture */ }
    const fast = live.state === 'loading' || live.state === 'online' || busy;
    timer = setTimeout(poll, document.hidden ? 15000 : (fast ? 2000 : 8000));
  }
  document.addEventListener('visibilitychange', () => { if (!document.hidden) poll(); });
  if (data.status && ['online', 'loading'].includes(data.status.state) && data.models[data.status.model]) model = data.status.model;
  paintTabs(); renderOptions();
  if (data.status) render(data.status);
  poll();
})();
"""


def page_html(data: dict, title: str = "XTier 86KD Console") -> str:
    return (
        "<!doctype html><html lang=\"zh-CN\"><head><meta charset=\"utf-8\">"
        "<meta name=\"viewport\" content=\"width=device-width,initial-scale=1\">"
        f"<title>{html.escape(title)}</title><style>"
        + CSS
        + "</style></head><body><main class=\"xt-page\">"
        + card_html(data)
        + "</main><script>"
        + JS
        + "</script></body></html>"
    )
