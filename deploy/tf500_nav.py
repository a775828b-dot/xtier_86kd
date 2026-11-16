"""Shared design system and sticky top navigation bar for the 8-GPU WebUI.

Provides neutral light/dark CSS tokens, sticky frosted topbar, live backend
status pill, and persistent Auto / Light / Dark theme switching.
"""

from __future__ import annotations

PAGES = (
    ("console", "/", "启动台"),
    ("benchmark", "/benchmark", "API 测速"),
    ("proxy", "/proxy", "API 代理"),
    ("server", "/server", "服务器信息"),
)

BASE_CSS = r"""
:root{
  color-scheme:light;
  --bg:#f4f5f7;--surface:#ffffff;--surface-2:#f8f9fb;--surface-3:#eef1f5;
  --border:#e3e6eb;--border-strong:#cfd5de;
  --text:#111827;--text-2:#374151;--muted:#6b7280;
  --accent:#2563eb;--accent-soft:#e8efff;--accent-text:#1d4ed8;
  --ok:#15803d;--ok-soft:#e6f6ec;
  --warn:#b45309;--warn-soft:#fff4de;
  --danger:#dc2626;--danger-soft:#fdecec;
  --code-bg:#f1f3f6;--pre-bg:#0f141a;--pre-text:#e2e8f0;
  --bar-gpu:#16a34a;--bar-hot:#ea580c;
  --shadow:0 1px 2px rgba(16,24,40,.05),0 1px 3px rgba(16,24,40,.04);
  --radius:12px;--radius-sm:8px;
  --mono:"JetBrains Mono","SFMono-Regular",ui-monospace,Menlo,Consolas,monospace;
  --sans:Inter,system-ui,-apple-system,"Segoe UI","PingFang SC","Hiragino Sans GB","Microsoft YaHei",sans-serif;
}
@media (prefers-color-scheme:dark){
  :root:not([data-theme=light]){
    color-scheme:dark;
    --bg:#0d1014;--surface:#151920;--surface-2:#1a1f27;--surface-3:#232a35;
    --border:#272e39;--border-strong:#38414f;
    --text:#e8ebf0;--text-2:#c5cbd5;--muted:#8d96a5;
    --accent:#5b8cff;--accent-soft:#1b2642;--accent-text:#a3bdff;
    --ok:#4ade80;--ok-soft:#0f2a1b;
    --warn:#fbbf24;--warn-soft:#2b2310;
    --danger:#f87171;--danger-soft:#2d1618;
    --code-bg:#1f252f;--pre-bg:#0a0d11;--pre-text:#d6dde8;
    --bar-gpu:#22c55e;--bar-hot:#f97316;
    --shadow:0 1px 2px rgba(0,0,0,.45);
  }
}
:root[data-theme=dark]{
  color-scheme:dark;
  --bg:#0d1014;--surface:#151920;--surface-2:#1a1f27;--surface-3:#232a35;
  --border:#272e39;--border-strong:#38414f;
  --text:#e8ebf0;--text-2:#c5cbd5;--muted:#8d96a5;
  --accent:#5b8cff;--accent-soft:#1b2642;--accent-text:#a3bdff;
  --ok:#4ade80;--ok-soft:#0f2a1b;
  --warn:#fbbf24;--warn-soft:#2b2310;
  --danger:#f87171;--danger-soft:#2d1618;
  --code-bg:#1f252f;--pre-bg:#0a0d11;--pre-text:#d6dde8;
  --bar-gpu:#22c55e;--bar-hot:#f97316;
  --shadow:0 1px 2px rgba(0,0,0,.45);
}
*{box-sizing:border-box}
html{-webkit-text-size-adjust:100%}
html,body{margin:0;padding:0;min-height:100%;background:var(--bg);color:var(--text);font:14px/1.6 var(--sans);-webkit-font-smoothing:antialiased}
a{color:var(--accent-text);text-decoration:none}
a:hover{text-decoration:underline}
main{max-width:1440px;margin:0 auto;padding:22px 24px 64px;min-width:0}
h1{font-size:22px;line-height:1.3;margin:0;font-weight:700;letter-spacing:-.01em;color:var(--text)}
h2{font-size:16px;line-height:1.4;margin:0;font-weight:650;color:var(--text)}
h3{font-size:14px;line-height:1.4;margin:0;font-weight:650;color:var(--text)}
p{margin:0}
small,.muted{color:var(--muted)}
code{font-family:var(--mono);font-size:12.5px;background:var(--code-bg);color:var(--text);border-radius:5px;padding:1px 5px;overflow-wrap:anywhere}
pre{font-family:var(--mono);font-size:12.5px;line-height:1.55;white-space:pre-wrap;overflow:auto;max-height:360px;margin:0;background:var(--pre-bg);color:var(--pre-text);border-radius:var(--radius-sm);padding:14px 16px}

.topbar{position:sticky;top:0;z-index:50;background:color-mix(in srgb,var(--surface) 90%,transparent);-webkit-backdrop-filter:saturate(1.5) blur(12px);backdrop-filter:saturate(1.5) blur(12px);border-bottom:1px solid var(--border)}
.topbar-inner{max-width:1440px;margin:0 auto;padding:0 24px;display:flex;align-items:center;gap:18px;min-height:56px}
.brand{display:flex;align-items:center;gap:9px;font-weight:700;font-size:15px;color:var(--text);white-space:nowrap;user-select:none}
.brand-mark{display:grid;place-items:center;width:26px;height:26px;border-radius:7px;background:linear-gradient(135deg,#2563eb,#7c3aed);color:#fff;font-size:11px;font-weight:700;letter-spacing:.02em}
.topnav{display:flex;gap:4px;overflow-x:auto;scrollbar-width:none;flex:1;min-width:0}
.topnav::-webkit-scrollbar{display:none}
.topnav a{padding:7px 12px;border-radius:8px;color:var(--text-2);white-space:nowrap;font-weight:550;font-size:13.5px;transition:background-color .14s,color .14s}
.topnav a:hover{background:var(--surface-3);text-decoration:none}
.topnav a.active-tab{background:var(--accent-soft);color:var(--accent-text);font-weight:650}

.status-pill{display:inline-flex;align-items:center;gap:8px;max-width:290px;padding:5px 12px;border-radius:999px;background:var(--surface-3);color:var(--text-2);font-size:12.5px;font-weight:600;white-space:nowrap}
.status-pill>span{overflow:hidden;text-overflow:ellipsis}
.status-pill::before{content:"";flex:none;width:8px;height:8px;border-radius:50%;background:var(--muted)}
.status-pill.ok{background:var(--ok-soft);color:var(--ok)}
.status-pill.ok::before{background:var(--ok);box-shadow:0 0 0 3px color-mix(in srgb,var(--ok) 22%,transparent);animation:xtNavPulse 2.4s ease-in-out infinite}
.status-pill.warn{background:var(--warn-soft);color:var(--warn)}
.status-pill.warn::before{background:var(--warn)}
.status-pill.danger{background:var(--danger-soft);color:var(--danger)}
.status-pill.danger::before{background:var(--danger)}
@keyframes xtNavPulse{50%{box-shadow:0 0 0 6px color-mix(in srgb,var(--ok) 0%,transparent)}}

.theme-toggle{display:inline-flex;align-items:center;gap:6px;flex:none;padding:5px 11px;border-radius:999px;background:var(--surface);color:var(--text-2);border:1px solid var(--border-strong);font:600 12.5px/1.4 var(--sans);cursor:pointer;white-space:nowrap}
.theme-toggle:hover{background:var(--surface-3);filter:none}
.theme-toggle svg{width:15px;height:15px;display:none}
.theme-toggle[data-mode=auto] .i-auto,.theme-toggle[data-mode=light] .i-sun,.theme-toggle[data-mode=dark] .i-moon{display:block}

.page-head{display:flex;justify-content:space-between;align-items:flex-end;gap:16px;flex-wrap:wrap;margin:4px 0 18px}
.page-head p{color:var(--muted);margin-top:4px;max-width:960px;font-size:13.5px}
.card{background:var(--surface);border:1px solid var(--border);border-radius:var(--radius);padding:18px 20px;box-shadow:var(--shadow);min-width:0}
section.card,details.card,.card+.card{margin-top:16px}
.card-head{display:flex;justify-content:space-between;align-items:flex-start;gap:12px;flex-wrap:wrap;margin-bottom:14px}
.card-head p{color:var(--muted);font-size:13px;margin-top:3px;max-width:900px}
.section-title{display:flex;align-items:baseline;justify-content:space-between;flex-wrap:wrap;gap:4px 12px;margin:26px 2px 12px}
.section-title h2{font-size:15px}
.section-title p,.section-title small{color:var(--muted);font-size:12.5px}

.badge,.service-pill,.pill{display:inline-flex;align-items:center;gap:6px;border-radius:999px;padding:3px 10px;background:var(--surface-3);color:var(--text-2);font-size:12px;font-weight:600;white-space:nowrap}
.badge.ok,.service-pill.ok,.pill.ok{background:var(--ok-soft);color:var(--ok)}
.badge.warn,.pill.warn{background:var(--warn-soft);color:var(--warn)}
.badge.danger,.pill.danger{background:var(--danger-soft);color:var(--danger)}
.ok-text{color:var(--ok)!important}.warn-text{color:var(--warn)!important}.danger-text{color:var(--danger)!important}

label{display:grid;gap:5px;font-size:12.5px;color:var(--muted);font-weight:550}
input,select,textarea{font:inherit;font-size:13.5px;width:100%;padding:7px 10px;border:1px solid var(--border-strong);border-radius:8px;background:var(--surface);color:var(--text);outline:none;transition:border-color .15s,box-shadow .15s}
input:focus,select:focus,textarea:focus{border-color:var(--accent);box-shadow:0 0 0 3px color-mix(in srgb,var(--accent) 20%,transparent)}
button{font:inherit;font-size:13px;font-weight:600;line-height:1.4;border:1px solid transparent;border-radius:8px;padding:7px 14px;background:var(--accent);color:#fff;cursor:pointer;white-space:nowrap;transition:filter .15s,background-color .15s,border-color .15s}
button:hover:not(:disabled){filter:brightness(1.07)}
button:disabled{opacity:.5;cursor:not-allowed}
button.secondary,.btn-secondary{display:inline-flex;align-items:center;justify-content:center;gap:6px;font:inherit;font-size:13px;font-weight:600;line-height:1.4;border-radius:8px;padding:7px 14px;background:var(--surface);color:var(--text);border:1px solid var(--border-strong);cursor:pointer;text-decoration:none}
button.secondary:hover:not(:disabled),.btn-secondary:hover{background:var(--surface-3);filter:none;text-decoration:none}
button.stop,button.danger{background:var(--surface);color:var(--danger);border:1px solid color-mix(in srgb,var(--danger) 40%,var(--border))}
button.stop:hover:not(:disabled),button.danger:hover:not(:disabled){background:var(--danger-soft);filter:none}
button.is-loading{cursor:wait;opacity:.9}
button.is-loading::before{content:"";display:inline-block;width:11px;height:11px;margin-right:7px;border:2px solid currentColor;border-right-color:transparent;border-radius:50%;vertical-align:-1px;animation:xtSpin .8s linear infinite}
@keyframes xtSpin{to{transform:rotate(360deg)}}

.table-wrap{overflow-x:auto;max-width:100%;border:1px solid var(--border);border-radius:var(--radius-sm);background:var(--surface)}
table{width:100%;border-collapse:collapse}
th,td{padding:9px 12px;text-align:left;vertical-align:middle;font-size:13px;border-bottom:1px solid var(--border)}
th{background:var(--surface-2);color:var(--muted);font-weight:600;font-size:12px;white-space:nowrap}
td{color:var(--text-2)}
tr:last-child td{border-bottom:0}
tbody tr:hover td{background:var(--surface-2)}
td.num,th.num{text-align:right;white-space:nowrap;font-variant-numeric:tabular-nums;font-family:var(--mono);font-size:12.5px}

details>summary{cursor:pointer;color:var(--text-2);font-weight:600;list-style:none;user-select:none}
details>summary::-webkit-details-marker{display:none}
details>summary::before{content:"\25B8";display:inline-block;width:15px;color:var(--muted);transition:transform .15s}
details[open]>summary::before{transform:rotate(90deg)}
details.card>summary{font-size:15px}
details.card[open]>summary{margin-bottom:14px}

.bar{height:6px;border-radius:999px;background:var(--surface-3);overflow:hidden;margin-top:5px}
.bar>span,.bar>i{display:block;height:100%;width:0;border-radius:inherit;background:var(--accent);transition:width .4s ease}
.bar.gpu>span,.bar.gpu>i{background:var(--bar-gpu)}
.bar.hot>span,.bar.hot>i{background:var(--bar-hot)}

.stats{display:grid;grid-template-columns:repeat(auto-fill,minmax(240px,1fr));gap:12px}
.stat{background:var(--surface);border:1px solid var(--border);border-radius:var(--radius);padding:15px 16px;box-shadow:var(--shadow);min-width:0}
.stat-label{display:flex;justify-content:space-between;align-items:center;gap:8px;font-size:12.5px;font-weight:650;color:var(--text-2)}
.stat-label>span:first-child{flex:none;white-space:nowrap}
.stat-hint{font-weight:500;color:var(--muted);font-size:11.5px;overflow:hidden;text-overflow:ellipsis;white-space:nowrap;min-width:0}
.stat-value{font-size:26px;font-weight:700;margin:6px 0 0;font-variant-numeric:tabular-nums;letter-spacing:-.02em;line-height:1.2;color:var(--text)}
.stat-value small{font-size:13px;font-weight:600;margin-left:4px;color:var(--muted)}
.stat-sub{font-size:12px;color:var(--muted);margin-top:6px;overflow-wrap:anywhere}

.kv-grid{display:grid;grid-template-columns:repeat(auto-fill,minmax(270px,1fr));gap:14px 16px}
.kv{min-width:0}
.kv>label{margin-bottom:5px}
code.config{display:block;padding:8px 11px;font-size:12.5px;border:1px solid var(--border);border-radius:8px;background:var(--surface-2);color:var(--text);white-space:nowrap;overflow:hidden;text-overflow:ellipsis}
.field-copy{position:relative}
.field-copy code.config{padding-right:64px}
.copy-inline-btn{position:absolute;right:4px;top:50%;transform:translateY(-50%);padding:3px 9px;font-size:12px;font-weight:600;background:var(--surface);color:var(--text-2);border:1px solid var(--border-strong);border-radius:6px}
.copy-inline-btn:hover{background:var(--surface-3);filter:none}
.copy-inline-btn.done{color:var(--ok);border-color:color-mix(in srgb,var(--ok) 45%,var(--border))}

@media (max-width:900px){
  .topbar-inner{flex-wrap:wrap;padding:8px 14px;gap:6px 12px;min-height:0}
  .topnav{order:3;flex-basis:100%;margin:0 -4px}
  .status-pill{max-width:52vw;margin-left:auto}
  .theme-label{display:none}
  main{padding:16px 14px 48px}
  .card{padding:16px}
}
"""

THEME_ICONS = (
    '<svg class="i-auto" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">'
    '<circle cx="12" cy="12" r="9"/><path d="M12 3a9 9 0 0 1 0 18z" fill="currentColor"/></svg>'
    '<svg class="i-sun" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round">'
    '<circle cx="12" cy="12" r="4"/><path d="M12 2v2M12 20v2M4.9 4.9l1.4 1.4M17.7 17.7l1.4 1.4M2 12h2M20 12h2M4.9 19.1l1.4-1.4M17.7 6.3l1.4-1.4"/></svg>'
    '<svg class="i-moon" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linejoin="round">'
    '<path d="M21 12.8A9 9 0 1 1 11.2 3a7 7 0 0 0 9.8 9.8z"/></svg>'
)

THEME_BOOT_JS = (
    "<script>(function(){try{var t=localStorage.getItem('xt86-theme');"
    "if(t==='light'||t==='dark')document.documentElement.dataset.theme=t;}catch(e){}})()</script>"
)

NAV_RUNTIME_JS = r"""<script>(function(){
  var order=['auto','light','dark'],names={auto:'跟随系统',light:'浅色',dark:'深色'};
  var root=document.documentElement,b=document.getElementById('theme-toggle');
  function cur(){return root.dataset.theme||'auto';}
  function paint(){
    if(!b)return;
    var t=cur();
    b.dataset.mode=t;
    b.title='主题：'+names[t]+'（点击切换）';
    b.setAttribute('aria-label',b.title);
    var lbl=b.querySelector('.theme-label');
    if(lbl)lbl.textContent=names[t];
  }
  if(b){
    b.addEventListener('click',function(){
      var n=order[(order.indexOf(cur())+1)%order.length];
      if(n==='auto'){delete root.dataset.theme;}else{root.dataset.theme=n;}
      try{if(n==='auto'){localStorage.removeItem('xt86-theme');}else{localStorage.setItem('xt86-theme',n);}}catch(e){}
      paint();
      window.dispatchEvent(new CustomEvent('xt-theme-change',{detail:{theme:n}}));
    });
    paint();
  }
  var pill=document.getElementById('nav-active-badge');
  async function pollBadge(){
    if(!pill)return;
    try{
      var r=await fetch('/api/proxy/status',{cache:'no-store'});
      if(r.ok){
        var d=await r.json();
        var span=pill.querySelector('span');
        if(d.backend==='online'){
          var txt=(d.backend_label||'推理在线')+(d.backend_port?' :'+d.backend_port:'');
          pill.className='status-pill ok';
          pill.title=txt;
          if(span)span.textContent=txt;
        }else{
          var offTxt=d.proxy==='online'?'未启动模型 · 代理在线':'服务待命';
          pill.className='status-pill';
          pill.title=offTxt;
          if(span)span.textContent=offTxt;
        }
      }
    }catch(e){}
    setTimeout(pollBadge,5000);
  }
  pollBadge();
})();</script>"""


def decorate(page: str, active: str) -> str:
    if active not in {item[0] for item in PAGES}:
        raise ValueError("unknown page")
    if "</style></head><body>" not in page:
        raise ValueError("unexpected page layout")

    links = []
    for key, href, label in PAGES:
        selected = key == active
        cls = "active-tab" if selected else ""
        aria = ' aria-current="page"' if selected else ""
        links.append(f'<a href="{href}" class="{cls}"{aria}>{label}</a>')

    topbar = (
        THEME_BOOT_JS
        + '<header class="topbar"><div class="topbar-inner">'
        + '<div class="brand"><span class="brand-mark">8G</span><span>TF500 运行管理</span></div>'
        + '<nav class="topnav xt-nav" aria-label="页面导航">'
        + "".join(links)
        + "</nav>"
        + '<span id="nav-active-badge" class="status-pill" title="正在读取运行状态"><span>检测中…</span></span>'
        + f'<button id="theme-toggle" class="theme-toggle" type="button" data-mode="auto">{THEME_ICONS}<span class="theme-label">跟随系统</span></button>'
        + "</div></header>"
        + NAV_RUNTIME_JS
    )
    return page.replace("</style></head><body>", BASE_CSS + "</style></head><body>" + topbar, 1)
