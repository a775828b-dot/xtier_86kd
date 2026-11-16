"""WebUI page renderer for the 8-GPU Agent API Benchmark (/benchmark).

Uses the shared neutral light/dark design system tokens from tf500_nav.BASE_CSS.
Supports 1/4/8/16/32/64/100 concurrency tiers, up to 512K/1M input lengths,
Total & Per-Session Prefill/Decode speeds, dual-axis SVG curves, CSV/JSON
export, and a 1..100 per-session speed inspector.
"""

from __future__ import annotations

CSS = r"""
.head-badges{display:flex;align-items:center;gap:8px;flex-wrap:wrap}

.b-top-grid{display:grid;grid-template-columns:minmax(0,300px) minmax(0,1fr) minmax(0,330px);gap:16px;align-items:stretch}
.b-top-grid>.card{margin-top:0;display:flex;flex-direction:column}

.cap-list{display:grid;gap:0}
.cap-row{display:flex;justify-content:space-between;align-items:baseline;gap:10px;padding:7px 0;border-bottom:1px solid var(--border);font-size:12.5px}
.cap-row:last-child{border-bottom:0}
.cap-row span{color:var(--muted);white-space:nowrap}
.cap-row b{font:600 12.5px var(--mono);color:var(--text);text-align:right;overflow-wrap:anywhere}
.cap-tags{display:flex;flex-wrap:wrap;gap:6px;margin-top:auto;padding-top:12px}

#bench-form{display:flex;flex-direction:column;gap:13px;min-width:0}
.f-group{display:flex;flex-direction:column;gap:6px;min-width:0}
.f-label-row{display:flex;justify-content:space-between;align-items:center;gap:8px;flex-wrap:wrap}
.f-label{font-size:12.5px;font-weight:650;color:var(--text-2)}
.f-hint{font-size:11.5px;color:var(--muted)}
.f-quick{display:inline-flex;gap:5px;flex-wrap:wrap}
.f-quick button{padding:3px 9px;border-radius:6px;font-size:11.5px;font-weight:600;background:var(--surface-2);color:var(--text-2);border:1px solid var(--border-strong)}
.f-quick button:hover{background:var(--surface-3);filter:none}

.mode-tabs{display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:8px}
.mode-btn{padding:10px 12px;border-radius:10px;border:1px solid var(--border-strong);background:var(--surface-2);color:var(--text);text-align:left;cursor:pointer;transition:border-color .14s,background-color .14s,box-shadow .14s;min-width:0;white-space:normal}
.mode-btn:hover:not(:disabled){background:var(--surface-3);filter:none}
.mode-btn strong{display:block;font-size:13.5px;color:var(--text);font-weight:650}
.mode-btn small{display:block;font-size:11.5px;color:var(--muted);font-weight:500;margin-top:2px;line-height:1.4}
.mode-btn.active{border-color:var(--accent);background:var(--accent-soft);box-shadow:0 0 0 1px color-mix(in srgb,var(--accent) 30%,transparent)}
.mode-btn.active strong{color:var(--accent-text)}

.pill-grid{display:grid;gap:6px;min-width:0}
.pill-grid.c7{grid-template-columns:repeat(7,minmax(0,1fr))}
.pill-grid.c5{grid-template-columns:repeat(5,minmax(0,1fr))}
.pill-grid.c6{grid-template-columns:repeat(6,minmax(0,1fr))}
.pill-btn{padding:7px 4px;border-radius:8px;border:1px solid var(--border-strong);background:var(--surface);color:var(--text-2);font:600 12.5px var(--mono);text-align:center;cursor:pointer;transition:all .12s;min-width:0}
.pill-btn:hover:not(:disabled){background:var(--surface-3);border-color:var(--accent);filter:none}
.pill-btn.active{border-color:var(--accent);background:var(--accent-soft);color:var(--accent-text);box-shadow:inset 0 0 0 1px color-mix(in srgb,var(--accent) 25%,transparent)}
.pill-btn.warn-cap{border-style:dashed}

.f-row-3{display:grid;grid-template-columns:repeat(3,minmax(0,1fr));gap:10px}
.f-actions{display:flex;gap:8px;align-items:center;margin-top:2px;flex-wrap:wrap}
.f-actions #btn-start{flex:1.4;min-width:150px;padding:9px 16px}

.prog-box{padding:12px 14px;border-radius:10px;background:var(--surface-2);border:1px solid var(--border)}
.prog-head{display:flex;justify-content:space-between;align-items:baseline;gap:8px}
.prog-pct{font:700 22px var(--mono);color:var(--text)}
.prog-sub{font-size:12px;color:var(--muted)}
.prog-msg{margin-top:8px;font-size:12px;color:var(--text-2);line-height:1.45;word-break:break-word}

.phase-strip{display:grid;grid-template-columns:repeat(4,minmax(0,1fr));gap:6px;margin-top:10px}
.phase-pill{padding:5px 4px;border-radius:6px;border:1px solid var(--border);background:var(--surface);text-align:center;font-size:11px;color:var(--muted);font-weight:550}
.phase-pill.active{border-color:color-mix(in srgb,var(--warn) 45%,var(--border));background:var(--warn-soft);color:var(--warn);font-weight:650}
.phase-pill.done{border-color:color-mix(in srgb,var(--ok) 40%,var(--border));background:var(--ok-soft);color:var(--ok);font-weight:600}

.live-box{margin-top:10px;padding:9px 11px;border-radius:8px;background:var(--surface-2);border:1px solid var(--border);font:500 11.5px var(--mono);color:var(--muted);display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:5px 8px}
.live-box b{color:var(--text)}

.plan-box{margin-top:10px;padding:10px 12px;border-radius:8px;background:var(--surface-2);border:1px solid var(--border);font-size:12px;flex:1}
.plan-head{display:flex;justify-content:space-between;gap:8px;color:var(--text-2);margin-bottom:7px;font-weight:650}
.plan-chips{display:flex;flex-wrap:wrap;gap:5px;max-height:84px;overflow-y:auto}
.plan-chip{padding:2px 8px;border-radius:6px;background:var(--surface);border:1px solid var(--border-strong);color:var(--text-2);font:600 11px var(--mono)}
.plan-chip.adj{border-color:color-mix(in srgb,var(--warn) 45%,var(--border));background:var(--warn-soft);color:var(--warn)}
.plan-notes{margin:8px 0 0;padding-left:16px;color:var(--muted);font-size:11.5px;line-height:1.5}
.plan-notes li{margin:3px 0}
.err-banner{margin-top:4px;padding:9px 12px;border-radius:8px;border:1px solid color-mix(in srgb,var(--danger) 40%,var(--border));background:var(--danger-soft);color:var(--danger);font-size:12.5px;display:none}
.err-banner.show{display:block}

.pt-selector{display:flex;gap:6px;flex-wrap:wrap;align-items:center}
.pt-chip{padding:4px 11px;border-radius:999px;border:1px solid var(--border-strong);background:var(--surface);color:var(--text-2);font:600 11.5px var(--mono)}
.pt-chip:hover{background:var(--surface-3);filter:none}
.pt-chip.active{border-color:var(--accent);background:var(--accent-soft);color:var(--accent-text)}

.hero-6{display:grid;grid-template-columns:repeat(6,minmax(0,1fr));gap:12px}
.hero-6 .stat.pre-card .stat-value{color:var(--accent-text)}
.hero-6 .stat.dec-card .stat-value{color:var(--ok)}
.hero-6 .stat-value{font-size:24px;font-family:var(--mono);white-space:nowrap;overflow:hidden;text-overflow:ellipsis}

.chart-toolbar{display:flex;justify-content:space-between;align-items:center;gap:12px;margin-bottom:12px;flex-wrap:wrap}
.chart-tabs{display:flex;gap:6px;flex-wrap:wrap}
.c-tab{padding:6px 12px;border-radius:8px;border:1px solid var(--border-strong);background:var(--surface);color:var(--text-2);font-size:12.5px;font-weight:600}
.c-tab:hover{background:var(--surface-3);filter:none}
.c-tab.active{border-color:var(--accent);background:var(--accent-soft);color:var(--accent-text)}
.chart-legend{display:flex;gap:14px;align-items:center;flex-wrap:wrap;font-size:12px;color:var(--muted)}
.leg-item{display:inline-flex;align-items:center;gap:6px}
.leg-swatch{width:12px;height:3px;border-radius:2px;display:inline-block}
.leg-band{width:12px;height:8px;border-radius:2px;display:inline-block;background:color-mix(in srgb,var(--accent) 18%,transparent);border:1px solid color-mix(in srgb,var(--accent) 50%,transparent)}

.chart-stage{position:relative;width:100%;min-height:296px;border-radius:10px;background:var(--surface-2);border:1px solid var(--border);padding:10px 8px 6px;overflow:hidden}
.chart-svg{width:100%;height:284px;display:block;overflow:visible}
.chart-empty{position:absolute;inset:0;display:flex;flex-direction:column;align-items:center;justify-content:center;color:var(--muted);font-size:13px;pointer-events:none;text-align:center;padding:20px;gap:4px}
.chart-empty strong{color:var(--text-2)}
.chart-tip{position:absolute;z-index:10;min-width:225px;max-width:305px;padding:10px 12px;border-radius:10px;background:var(--surface);border:1px solid var(--border-strong);box-shadow:0 10px 28px rgba(0,0,0,.18);color:var(--text);font-size:12px;pointer-events:none;display:none}
.chart-tip strong{display:block;color:var(--accent-text);font-size:12.5px;margin-bottom:5px}
.tip-row{display:flex;justify-content:space-between;gap:10px;margin:2px 0}
.tip-row span{color:var(--muted)}
.tip-row b{font-family:var(--mono);font-variant-numeric:tabular-nums}

.b-table{min-width:980px}
.b-table th{text-align:right}
.b-table th:first-child,.b-table td:first-child{text-align:left}
.b-table td{text-align:right;font-family:var(--mono);font-size:12.5px;white-space:nowrap}
.b-table tbody tr{cursor:pointer}
.b-table tbody tr.selected td{background:var(--accent-soft)}
.b-table td .sub-cell{display:block;font-family:var(--sans);font-size:11px;color:var(--muted);margin-top:1px}
.td-pre{color:var(--accent-text)!important;font-weight:650}
.td-dec{color:var(--ok)!important;font-weight:650}

.lanes-summary{display:flex;flex-wrap:wrap;gap:8px;margin-bottom:12px}
.l-stat{padding:6px 11px;border-radius:8px;background:var(--surface-2);border:1px solid var(--border);font-size:12px;color:var(--muted)}
.l-stat b{font:600 12.5px var(--mono);color:var(--text);margin-left:5px}
.lanes-grid{display:grid;grid-template-columns:repeat(auto-fill,minmax(235px,1fr));gap:10px;max-height:430px;overflow-y:auto;padding-right:4px}
.lane-item{padding:12px 13px;border-radius:10px;border:1px solid var(--border);background:var(--surface-2);min-width:0}
.lane-top{display:flex;justify-content:space-between;align-items:center;gap:6px;margin-bottom:6px}
.lane-id{font:700 12px var(--mono);color:var(--accent-text)}
.lane-tools{font-size:11px;color:var(--muted);overflow:hidden;text-overflow:ellipsis;white-space:nowrap;max-width:148px}
.lane-metrics{display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:4px 10px;margin-top:8px;font-size:11.5px}
.lane-metrics span{color:var(--muted)}
.lane-metrics b{font:600 11.5px var(--mono);color:var(--text);float:right}

.method-grid{display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:12px 20px;line-height:1.6}
.method-grid h4{margin:0 0 4px;color:var(--text);font-size:13px}
.method-grid p{margin:0;font-size:12.5px;color:var(--muted)}

@media(max-width:1200px){
  .b-top-grid{grid-template-columns:1fr}
  .hero-6{grid-template-columns:repeat(3,minmax(0,1fr))}
}
@media(max-width:740px){
  .hero-6{grid-template-columns:repeat(2,minmax(0,1fr))}
  .pill-grid.c7{grid-template-columns:repeat(4,minmax(0,1fr))}
  .pill-grid.c5{grid-template-columns:repeat(3,minmax(0,1fr))}
  .f-row-3,.method-grid{grid-template-columns:1fr}
}
"""

BODY = r"""
<main class="xt-bench">
  <div class="page-head">
    <div>
      <h1>API 性能测速</h1>
      <p>基于真实编码 Agent 工具链与隔离沙盒测速，支持 <b>1 / 4 / 8 / 16 / 32 / 64 / 100</b> 并发与最高 <b>512K</b>（及 1M）上下文，精确拆分 <b>Prefill 总速度 / 每会话速度</b> 与 <b>Decode 总速度 / 每会话速度</b>。</p>
    </div>
    <div class="head-badges">
      <span id="backend-pill" class="status-pill"><span id="backend-pill-text">正在检测后端…</span></span>
      <span id="run-pill" class="pill">空闲待命</span>
    </div>
  </div>

  <section class="b-top-grid" aria-label="测速控制面板">
    <!-- Card 1: Target & KV Capacity -->
    <article class="card">
      <div class="card-head">
        <div>
          <h2>接入后端与实时容量</h2>
          <p>只读读取当前容量，不重启推理服务</p>
        </div>
        <button type="button" id="btn-refresh-cap" class="secondary">↻ 刷新</button>
      </div>
      <div class="cap-list">
        <div class="cap-row"><span>代理入口</span><b><TF500_IP>:8000/v1</b></div>
        <div class="cap-row"><span>当前后端</span><b id="cap-backend">检测中…</b></div>
        <div class="cap-row"><span>在线模型</span><b id="cap-model">—</b></div>
        <div class="cap-row"><span>模型最大上下文</span><b id="cap-ctx">—</b></div>
        <div class="cap-row"><span>共享 KV 池总容量</span><b id="cap-kv" class="ok-text">—</b></div>
        <div class="cap-row"><span>DP 分片 / 单池容量</span><b id="cap-dp">—</b></div>
        <div class="cap-row"><span>最大并行请求槽位</span><b id="cap-running">—</b></div>
      </div>
      <div id="cap-tags" class="cap-tags">
        <span class="badge">Agent 6 工具沙盒</span>
        <span class="badge">连续流式计时</span>
      </div>
    </article>

    <!-- Card 2: Benchmark Configuration Form -->
    <article class="card">
      <div class="card-head">
        <div>
          <h2>并发挡位与上下文配置</h2>
          <p>选择自动连跑多个并发挡位，或只选单个并发挡位测试</p>
        </div>
      </div>

      <form id="bench-form" novalidate>
        <div class="f-group">
          <div class="f-label-row">
            <span class="f-label">测试方式</span>
            <span class="f-hint">可勾选多个并发挡位自动连跑，或只选一个并发挡位单独测试</span>
          </div>
          <div class="mode-tabs" role="radiogroup" aria-label="测速模式">
            <button type="button" class="mode-btn active" data-mode="concurrency">
              <strong>自动跑多个并发挡位测试</strong>
              <small>勾选几个并发就按顺序自动跑完几个（如 1 → 100 连测）</small>
            </button>
            <button type="button" class="mode-btn" data-mode="context">
              <strong>选一个并发挡位测试</strong>
              <small>只测选中的单个并发（可单跑 1 次，或按长度梯度连跑）</small>
            </button>
          </div>
        </div>

        <div class="f-group">
          <div class="f-label-row">
            <span class="f-label" id="conc-label">选择要自动连跑的并发挡位（可多选）</span>
            <div class="f-quick" id="conc-quick">
              <button type="button" data-preset="all">全选 7 档 (1→100)</button>
              <button type="button" data-preset="low">1→32</button>
              <button type="button" data-preset="high">32 / 64 / 100</button>
            </div>
          </div>
          <div id="conc-grid" class="pill-grid c7" role="group" aria-label="并发挡位">
            <button type="button" class="pill-btn active" data-conc="1">1</button>
            <button type="button" class="pill-btn active" data-conc="4">4</button>
            <button type="button" class="pill-btn active" data-conc="8">8</button>
            <button type="button" class="pill-btn active" data-conc="16">16</button>
            <button type="button" class="pill-btn active" data-conc="32">32</button>
            <button type="button" class="pill-btn active" data-conc="64">64</button>
            <button type="button" class="pill-btn active" data-conc="100">100</button>
          </div>
        </div>

        <div class="f-group">
          <div class="f-label-row">
            <span class="f-label" id="input-label">目标输入长度（真实项目源码 + Agent 历史）</span>
            <span class="f-hint" id="input-hint">已选 512K (524,288 tokens)</span>
          </div>
          <div id="input-grid" class="pill-grid c5" role="group" aria-label="输入长度挡位">
            <button type="button" class="pill-btn" data-max="4096">4K</button>
            <button type="button" class="pill-btn" data-max="8192">8K</button>
            <button type="button" class="pill-btn" data-max="16384">16K</button>
            <button type="button" class="pill-btn" data-max="32768">32K</button>
            <button type="button" class="pill-btn" data-max="65536">64K</button>
            <button type="button" class="pill-btn" data-max="131072">128K</button>
            <button type="button" class="pill-btn" data-max="262144">256K</button>
            <button type="button" class="pill-btn" data-max="393216">384K</button>
            <button type="button" class="pill-btn active" data-max="524288">512K</button>
            <button type="button" class="pill-btn" data-max="1048576">1M</button>
          </div>
        </div>

        <div class="f-group" id="points-group" style="display:none">
          <div class="f-label-row">
            <span class="f-label">上下文采样点数</span>
            <span class="f-hint">1 点直测目标长度；多点自动按比例均分测出上下文衰减曲线</span>
          </div>
          <div id="points-grid" class="pill-grid c6">
            <button type="button" class="pill-btn active" data-pts="1">1 点直测</button>
            <button type="button" class="pill-btn" data-pts="2">2 点</button>
            <button type="button" class="pill-btn" data-pts="4">4 点均分</button>
            <button type="button" class="pill-btn" data-pts="8">8 点均分</button>
            <button type="button" class="pill-btn" data-pts="12">12 点</button>
            <button type="button" class="pill-btn" data-pts="16">16 点</button>
          </div>
        </div>

        <div class="f-row-3">
          <label>
            单轮最大输出
            <select id="sel-output" name="output">
              <option value="512">512 tokens</option>
              <option value="1024" selected>1K (1,024 tokens)</option>
              <option value="2048">2K (2,048 tokens)</option>
              <option value="4096">4K (4,096 tokens)</option>
            </select>
          </label>
          <label>
            Agent 采样策略
            <select id="sel-strategy" name="strategy">
              <option value="agent_auto" selected>标准多轮（工具沙盒 + 长代码交付）</option>
              <option value="agent_fast">快速单波（冷预填充 + 直接长代码交付）</option>
            </select>
          </label>
          <label>
            有效长解码轮数
            <select id="sel-turns" name="turns">
              <option value="1" selected>1 轮有效并行采样</option>
              <option value="2">2 轮有效并行采样</option>
              <option value="3">3 轮有效并行采样</option>
            </select>
          </label>
        </div>

        <div class="f-actions">
          <button type="submit" id="btn-start">开始 Agent 测速</button>
          <button type="button" id="btn-stop" class="stop" disabled>停止测速</button>
          <button type="button" id="btn-clear" class="secondary">清除结果</button>
        </div>
        <div id="form-error" class="err-banner" role="alert"></div>
      </form>
    </article>

    <!-- Card 3: Live Progress & Execution Plan -->
    <article class="card">
      <div class="card-head">
        <div>
          <h2>实时进度与容量规划</h2>
          <p>按实时 KV 池校验每路会话安全上限</p>
        </div>
        <span id="elapsed-label" class="badge">00:00</span>
      </div>

      <div class="prog-box">
        <div class="prog-head">
          <span id="dial-pct-text" class="prog-pct">0%</span>
          <span id="dial-point-text" class="prog-sub">待命</span>
        </div>
        <div class="bar"><span id="prog-bar"></span></div>
        <div id="dial-msg" class="prog-msg">选择并发挡位与输入长度后即可启动测试。</div>
      </div>

      <div class="phase-strip" aria-label="执行阶段">
        <div class="phase-pill" data-ph="prepare">1. 构造会话</div>
        <div class="phase-pill" data-ph="prefill">2. 冷预填充</div>
        <div class="phase-pill" data-ph="decode">3. 并行解码</div>
        <div class="phase-pill" data-ph="complete">4. 聚合完成</div>
      </div>

      <div class="live-box" id="live-box">
        <span>当前波次：<b id="lv-wave">—</b></span>
        <span>首字到达：<b id="lv-prefill">—</b></span>
        <span>完成会话：<b id="lv-done">—</b></span>
        <span>实时解码：<b id="lv-tps">—</b></span>
      </div>

      <div class="plan-box">
        <div class="plan-head">
          <span id="plan-summary">执行计划预览</span>
          <span id="plan-points-badge">0 点</span>
        </div>
        <div id="plan-chips" class="plan-chips"></div>
        <ul id="plan-notes" class="plan-notes"></ul>
      </div>
    </article>
  </section>

  <!-- Point Switcher & 6-Card Speed HUD -->
  <div class="section-title">
    <div>
      <h2>核心吞吐与每会话分配速度</h2>
      <p id="hud-point-desc">尚未运行测速 · 点击图表或表格任意行可切换查看对应点位</p>
    </div>
    <div id="pt-selector" class="pt-selector"></div>
  </div>

  <section class="hero-6" aria-label="核心速度指标">
    <article class="stat pre-card">
      <div class="stat-label"><span>Prefill 总速度</span><span class="stat-hint">tok/s</span></div>
      <div id="m-pre-total" class="stat-value">—</div>
      <div id="m-pre-total-sub" class="stat-sub">整组独立冷预填充总吞吐</div>
    </article>
    <article class="stat pre-card">
      <div class="stat-label"><span>Prefill 每会话速度</span><span class="stat-hint">tok/s/会话</span></div>
      <div id="m-pre-sess" class="stat-value">—</div>
      <div id="m-pre-sess-sub" class="stat-sub">总速度 ÷ 并发 · 单路均速 —</div>
    </article>
    <article class="stat dec-card">
      <div class="stat-label"><span>Decode 总速度</span><span class="stat-hint">tok/s</span></div>
      <div id="m-dec-total" class="stat-value">—</div>
      <div id="m-dec-total-sub" class="stat-sub">并行活跃窗口总吞吐 · 合计 —</div>
    </article>
    <article class="stat dec-card">
      <div class="stat-label"><span>Decode 每会话速度</span><span class="stat-hint">tok/s/会话</span></div>
      <div id="m-dec-sess" class="stat-value">—</div>
      <div id="m-dec-sess-sub" class="stat-sub">总速度 ÷ 并发 · 单路中位 —</div>
    </article>
    <article class="stat">
      <div class="stat-label"><span>冷预填充首字 TTFT</span><span class="stat-hint">秒 (s)</span></div>
      <div id="m-ttft" class="stat-value">—</div>
      <div id="m-ttft-sub" class="stat-sub">P95 — · 多轮缓存命中 —</div>
    </article>
    <article class="stat">
      <div class="stat-label"><span>端到端吞吐 / DSpark</span><span class="stat-hint">tok/s</span></div>
      <div id="m-e2e" class="stat-value">—</div>
      <div id="m-e2e-sub" class="stat-sub">每会话 — · 接受步长 —</div>
    </article>
  </section>

  <!-- Dual-Axis Interactive Chart -->
  <section class="card" aria-label="性能趋势图">
    <div class="chart-toolbar">
      <div class="chart-tabs" role="tablist">
        <button type="button" class="c-tab active" data-chart="decode">Decode 速度（总速度 + 每会话）</button>
        <button type="button" class="c-tab" data-chart="prefill">Prefill 速度（总速度 + 每会话）</button>
        <button type="button" class="c-tab" data-chart="ttft">首字延迟 TTFT（均值 / P95）</button>
        <button type="button" class="c-tab" data-chart="e2e">端到端吞吐（总速度 + 每会话）</button>
      </div>
      <div id="chart-legend" class="chart-legend"></div>
    </div>
    <div class="chart-stage" id="chart-stage">
      <svg id="chart-svg" class="chart-svg" viewBox="0 0 1120 284" preserveAspectRatio="none"></svg>
      <div id="chart-empty" class="chart-empty">
        <strong>等待测速数据</strong>
        <span>启动 Agent 测速后，每个完成点位将实时绘制总速度（左轴）与每会话速度（右轴）双曲线</span>
      </div>
      <div id="chart-tip" class="chart-tip"></div>
    </div>
  </section>

  <!-- Full Results Table + Export -->
  <section class="card" aria-label="各挡位测速结果对比表">
    <div class="card-head">
      <div>
        <h2>各挡位总速度与每会话速度对比表</h2>
        <p>点击任意行可切换上方指标卡与下方每会话速度分配明细</p>
      </div>
      <div style="display:flex;gap:8px;flex-wrap:wrap">
        <button type="button" id="btn-export-csv" class="secondary" disabled>导出 CSV</button>
        <button type="button" id="btn-export-json" class="secondary" disabled>导出 JSON</button>
      </div>
    </div>
    <div class="table-wrap">
      <table class="b-table">
        <thead>
          <tr>
            <th>测试点 / 挡位</th>
            <th>并发会话</th>
            <th>实际输入</th>
            <th>Prefill 总速度</th>
            <th>Prefill 每会话</th>
            <th>Decode 总速度</th>
            <th>Decode 每会话</th>
            <th>首字 TTFT (均/P95)</th>
            <th>端到端 (总/每会话)</th>
            <th>DSpark / 缓存命中</th>
            <th>状态</th>
          </tr>
        </thead>
        <tbody id="results-tbody">
          <tr><td colspan="11" style="text-align:center;color:var(--muted);padding:24px;font-family:var(--sans)">暂无测速记录</td></tr>
        </tbody>
      </table>
    </div>
  </section>

  <!-- Per-Session Speed Inspector (1..100 Sessions) -->
  <section class="card" aria-label="每会话速度明细">
    <div class="card-head">
      <div>
        <h2 id="lanes-title">每会话 Prefill / Decode 速度分配明细（支持 1–100 会话）</h2>
        <p>展示当前选中点位下全部并发会话的分配均速、单路实测速率、首字延迟与工具调用轨迹</p>
      </div>
      <div style="display:flex;gap:8px;align-items:center;flex-wrap:wrap">
        <select id="lanes-sort" style="width:auto;font-size:12.5px">
          <option value="lane">按会话序号 (S001→S100)</option>
          <option value="dec_desc">按 Decode 实测降序</option>
          <option value="dec_asc">按 Decode 实测升序</option>
          <option value="ttft_desc">按首字延迟降序</option>
        </select>
      </div>
    </div>
    <div id="lanes-summary" class="lanes-summary">
      <span class="l-stat">完成第一个测试点后，此处展示全部并发会话（最高 100 路）的分配速度与单路实测明细</span>
    </div>
    <div id="lanes-grid" class="lanes-grid"></div>
  </section>

  <!-- Methodology Reference -->
  <details class="card">
    <summary>Agent 测速口径与公式说明（Prefill / Decode 总速度与每会话分配速度）</summary>
    <div class="method-grid">
      <div>
        <h4>1. 真实编码 Agent 负载与最高 512K+ 输入</h4>
        <p>每次请求携带完整编码代理系统提示词、6 个 OpenAI Function 工具 Schema（<code>read_file</code> / <code>list_dir</code> / <code>grep</code> / <code>edit_file</code> / <code>write_file</code> / <code>check_python</code>）及真实项目源码阅读历史，按 <code>/v1/tokenize</code> 精确对齐至 4K–512K（或 1M）。每路并发拥有独立会话前缀与零拷贝内存隔离沙盒，杜绝跨并发虚假缓存命中，且绝不调用 <code>/flush_cache</code>。</p>
      </div>
      <div>
        <h4>2. Prefill 总速度 &amp; 分配到每个会话的速度</h4>
        <p>第 1 波所有并发会话通过同步屏障同时发出冷前缀请求。<b>Prefill 总速度</b> = <code>Σ(各路未命中缓存输入 tokens) ÷ (最晚首字时间 − 最早发包时间)</code>；<b>Prefill 每会话分配速度</b> = <code>Prefill 总速度 ÷ 并发会话数</code>（同时展示单路独立 <code>uncached_tokens ÷ ttft</code> 均值供对照）。</p>
      </div>
      <div>
        <h4>3. Decode 总速度 &amp; 分配到每个会话的速度</h4>
        <p>开启 SGLang <code>continuous_usage_stats</code> 逐包精确计量首包后生成的 token 数。<b>Decode 总速度</b> = <code>并行波次内有效解码总 token 数 ÷ 各会话 [首字, 末字] 流式时间区间并集</code>；<b>Decode 每会话分配速度</b> = <code>Decode 总速度 ÷ 并发会话数</code>，并在明细区列出每路会话各自的实测速率、P10 / P50 / P90 分位数及各路速率算术和。</p>
      </div>
      <div>
        <h4>4. 多轮工具执行与自动长代码交付</h4>
        <p>在「标准多轮」模式下，第 1 波允许模型自主调用工具操作隔离沙盒并观测真实工具回合，若模型仅返回短工具调用，第 2 波自动保留完整前缀（观测 RadixCache 命中率与缓存首字延迟）并进入长代码模块交付阶段，确保高并发下获得充足、稳定的持续解码窗口。</p>
      </div>
    </div>
  </details>
</main>
"""

JS = r"""
(() => {
  const $ = (id) => document.getElementById(id);
  const ALL_CONCS = [1, 4, 8, 16, 32, 64, 100];

  const ui = {
    mode: 'concurrency',
    concurrency: 16,
    concurrencies: [1, 4, 8, 16, 32, 64, 100],
    maximum: 524288,
    points: 1,
    output: 1024,
    strategy: 'agent_auto',
    turns: 1,
    fingerprint: '',
    online: false,
    capacity: null,
    plan: null,
    status: null,
    chartTab: 'decode',
    selectedPointIdx: -1,
    laneSort: 'lane',
    planTimer: null,
    pollTimer: null
  };

  function fmt(v, d = 1) {
    return Number.isFinite(v) ? Number(v).toLocaleString('zh-CN', { minimumFractionDigits: d, maximumFractionDigits: d }) : '—';
  }
  function fmtInt(v) {
    return Number.isFinite(v) ? Math.round(v).toLocaleString('zh-CN') : '—';
  }
  function fmtTokTier(n) {
    if (!Number.isFinite(n) || n <= 0) return '—';
    if (n >= 1048576 && n % 1048576 === 0) return (n / 1048576) + 'M';
    if (n >= 1024) return Math.round(n / 1024) + 'K';
    return String(n);
  }
  function fmtSec(s) {
    if (!Number.isFinite(s)) return '—';
    return s >= 10 ? fmt(s, 1) + 's' : fmt(s, 2) + 's';
  }
  function fmtClock(sec) {
    if (!Number.isFinite(sec) || sec < 0) return '00:00';
    const m = Math.floor(sec / 60), s = Math.floor(sec % 60);
    return String(m).padStart(2, '0') + ':' + String(s).padStart(2, '0');
  }
  function esc(str) {
    const d = document.createElement('div');
    d.textContent = str == null ? '' : String(str);
    return d.innerHTML;
  }

  function showError(msg) {
    const box = $('form-error');
    if (!msg) {
      box.textContent = '';
      box.classList.remove('show');
      return;
    }
    box.textContent = msg;
    box.classList.add('show');
  }

  function buildFormParams(withFp = false) {
    const p = new URLSearchParams();
    p.set('mode', ui.mode);
    p.set('concurrency', String(ui.concurrency));
    p.set('concurrencies', ui.concurrencies.join(','));
    p.set('maximum', String(ui.maximum));
    p.set('points', String(ui.mode === 'concurrency' ? 1 : ui.points));
    p.set('output', String(ui.output));
    p.set('strategy', ui.strategy);
    p.set('turns', String(ui.turns));
    if (withFp && ui.fingerprint) p.set('fingerprint', ui.fingerprint);
    return p;
  }

  function syncControls() {
    const busy = ui.status && ['preparing', 'running', 'stopping'].includes(ui.status.state);

    document.querySelectorAll('.mode-btn').forEach((btn) => {
      btn.classList.toggle('active', btn.dataset.mode === ui.mode);
      btn.disabled = busy;
    });

    $('points-group').style.display = ui.mode === 'context' ? '' : 'none';
    $('conc-quick').style.display = ui.mode === 'concurrency' ? 'inline-flex' : 'none';
    $('conc-label').textContent = ui.mode === 'concurrency'
      ? '选择要自动连跑的并发挡位（可多选）'
      : '选择一个要测试的并发挡位（单选）';

    document.querySelectorAll('#conc-grid .pill-btn').forEach((btn) => {
      const c = Number(btn.dataset.conc);
      const on = ui.mode === 'concurrency' ? ui.concurrencies.includes(c) : ui.concurrency === c;
      btn.classList.toggle('active', on);
      btn.disabled = busy;
    });

    document.querySelectorAll('#input-grid .pill-btn').forEach((btn) => {
      const m = Number(btn.dataset.max);
      btn.classList.toggle('active', m === ui.maximum);
      const overCtx = ui.capacity && ui.capacity.context && m > ui.capacity.context;
      btn.classList.toggle('warn-cap', !!overCtx);
      btn.title = overCtx ? `超过当前在线模型上下文 (${fmtTokTier(ui.capacity.context)})，规划时将自动收敛` : '';
      btn.disabled = busy;
    });
    $('input-hint').textContent = `已选 ${fmtTokTier(ui.maximum)} (${fmtInt(ui.maximum)} tokens)`;

    document.querySelectorAll('#points-grid .pill-btn').forEach((btn) => {
      btn.classList.toggle('active', Number(btn.dataset.pts) === ui.points);
      btn.disabled = busy;
    });

    $('sel-output').value = String(ui.output);
    $('sel-strategy').value = ui.strategy;
    $('sel-turns').value = String(ui.turns);
    $('sel-output').disabled = busy;
    $('sel-strategy').disabled = busy;
    $('sel-turns').disabled = busy;

    $('btn-start').disabled = busy || !ui.online;
    $('btn-start').classList.toggle('is-loading', !!busy);
    $('btn-start').textContent = !ui.online
      ? '等待推理后端上线…'
      : (busy ? 'Agent 测速进行中' : '开始 Agent 测速');
    $('btn-stop').disabled = !busy || ui.status?.state === 'stopping';
    $('btn-clear').disabled = busy;
  }

  function renderCapacityAndPlan(plan) {
    ui.plan = plan;
    ui.online = !!plan.online;
    ui.capacity = plan.capacity || null;
    ui.fingerprint = plan.capacity?.fingerprint || '';

    const pill = $('backend-pill');
    pill.className = 'status-pill ' + (ui.online ? 'ok' : '');
    if (ui.online && ui.capacity) {
      const c = ui.capacity;
      $('backend-pill-text').textContent = `后端在线 · ${c.backend_label || (':' + c.backend_port)}`;
      $('cap-backend').textContent = `${c.backend_label || 'SGLang'} (:${c.backend_port})`;
      $('cap-backend').className = 'ok-text';
      $('cap-model').textContent = c.model;
      $('cap-ctx').textContent = `${fmtTokTier(c.context)} (${fmtInt(c.context)})`;
      $('cap-kv').textContent = `${fmtInt(c.kv)} tok`;
      $('cap-dp').textContent = c.dp_ranks > 1
        ? `DP=${c.dp_ranks} · 单池 ${fmtInt(c.single_kv)}`
        : `DP=1 · 共享池 ${fmtInt(c.single_kv)}`;
      $('cap-running').textContent = `${c.max_running} 并发槽位`;
      const tags = [
        `<span class="badge">Agent 6 工具沙盒</span>`,
        `<span class="badge">页大小 ${c.page_size}</span>`,
        c.spec_algo ? `<span class="badge ok">投机解码: ${esc(c.spec_algo)}</span>` : `<span class="badge">标准自回归</span>`,
        c.chunked_prefill_size ? `<span class="badge">Chunk: ${fmtInt(c.chunked_prefill_size)}</span>` : ''
      ].filter(Boolean);
      $('cap-tags').innerHTML = tags.join('');
    } else {
      $('backend-pill-text').textContent = '推理后端未在线（只读待命）';
      $('cap-backend').textContent = '未检测到 :30000';
      $('cap-backend').className = '';
      $('cap-model').textContent = 'DeepSeek-V4.1-Flash-1017 (待上线)';
      $('cap-ctx').textContent = '—';
      $('cap-kv').textContent = '—';
      $('cap-dp').textContent = '—';
      $('cap-running').textContent = '—';
    }

    const steps = plan.steps || [];
    $('plan-points-badge').textContent = `共 ${steps.length} 点`;
    $('plan-summary').textContent = plan.mode === 'concurrency'
      ? `多并发挡位连测计划（输入 ${fmtTokTier(ui.maximum)}）`
      : `单并发挡位测试计划（固定 ${ui.concurrency} 并发）`;

    $('plan-chips').innerHTML = steps.map((s) => {
      const label = plan.mode === 'concurrency'
        ? `${s.concurrency}并发 · ${fmtTokTier(s.target)}`
        : `#${s.index} · ${fmtTokTier(s.target)} (${s.concurrency}并发)`;
      return `<span class="plan-chip ${s.adjusted ? 'adj' : ''}" title="单路安全容量上限: ${fmtInt(s.per_lane)} tokens">${esc(label)}</span>`;
    }).join('');

    const notes = [...(plan.notes || [])];
    if (!plan.online && plan.offline_reason && !notes.length) notes.push(plan.offline_reason);
    $('plan-notes').innerHTML = notes.map((n) => `<li>${esc(n)}</li>`).join('');
    syncControls();
  }

  async function refreshPlan() {
    try {
      const resp = await fetch('/api/benchmark/plan', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/x-www-form-urlencoded;charset=UTF-8',
          'X-Requested-With': 'fetch'
        },
        body: buildFormParams(false).toString(),
        cache: 'no-store'
      });
      const data = await resp.json();
      if (!resp.ok) {
        showError(data.error || '规划参数无效');
        return;
      }
      showError('');
      renderCapacityAndPlan(data);
    } catch (_err) {
      $('backend-pill').className = 'status-pill danger';
      $('backend-pill-text').textContent = '控制台连接异常';
    }
  }

  function schedulePlan() {
    clearTimeout(ui.planTimer);
    ui.planTimer = setTimeout(refreshPlan, 140);
  }

  function getDisplayPoints(st) {
    const done = Array.isArray(st?.results) ? st.results.slice() : [];
    if (st?.partial && ['preparing', 'running', 'stopping'].includes(st.state)) {
      const exists = done.some((r) => r.point === st.partial.point);
      if (!exists) done.push({ ...st.partial, _live: true });
    }
    return done;
  }

  function renderProgress(st) {
    const state = st?.state || 'idle';
    const busy = ['preparing', 'running', 'stopping'].includes(state);
    const runPill = $('run-pill');
    const stateLabels = {
      idle: '空闲待命',
      preparing: '准备会话中',
      running: '流式测速中',
      stopping: '正在停止…',
      done: '测速已完成',
      stopped: '已手动停止',
      failed: '测速中止',
      interrupted: '进程中断'
    };
    runPill.className = 'pill ' + (busy ? 'warn' : (state === 'done' ? 'ok' : (state === 'failed' || state === 'interrupted' ? 'danger' : '')));
    runPill.textContent = stateLabels[state] || state;
    $('elapsed-label').textContent = fmtClock(st?.elapsed_s || 0);

    const totalPts = st?.total_points || ui.plan?.points || 0;
    const donePts = Array.isArray(st?.results) ? st.results.length : 0;
    const lp = st?.live_progress;
    let frac = 0;
    if (state === 'done') {
      frac = 1;
    } else if (totalPts > 0 && busy) {
      const sub = lp && lp.total > 0 ? Math.min(0.95, (lp.prefilled * 0.35 + lp.done * 0.65) / lp.total) : 0.08;
      frac = Math.min(0.99, (donePts + sub) / totalPts);
    } else if (totalPts > 0 && donePts > 0) {
      frac = donePts / totalPts;
    }
    const pct = Math.round(frac * 100);
    $('prog-bar').style.width = pct + '%';
    $('dial-pct-text').textContent = pct + '%';
    $('dial-point-text').textContent = totalPts > 0 ? `完成 ${donePts}/${totalPts} 点` : '待命';
    $('dial-msg').textContent = st?.message || '选择并发挡位与输入长度后即可启动测试。';

    const phaseOrder = ['prepare', 'prefill', 'decode', 'complete'];
    const curPhase = st?.phase || 'idle';
    const curIdx = phaseOrder.indexOf(curPhase);
    document.querySelectorAll('.phase-pill').forEach((el, idx) => {
      el.classList.remove('active', 'done');
      if (state === 'done') {
        el.classList.add('done');
      } else if (curIdx >= 0) {
        if (idx < curIdx) el.classList.add('done');
        else if (idx === curIdx) el.classList.add('active');
      }
    });

    if (lp) {
      $('lv-wave').textContent = `第 ${lp.turn} 波`;
      $('lv-prefill').textContent = `${lp.prefilled} / ${lp.total}`;
      $('lv-done').textContent = `${lp.done} / ${lp.total} (${fmtInt(lp.streamed_tokens)} tok)`;
      $('lv-tps').textContent = lp.live_tps ? `${fmt(lp.live_tps, 1)} tok/s` : '计时中…';
    } else {
      $('lv-wave').textContent = busy ? '初始化…' : '—';
      $('lv-prefill').textContent = '—';
      $('lv-done').textContent = '—';
      $('lv-tps').textContent = '—';
    }
  }

  function pointTag(pt, mode) {
    const inStr = fmtTokTier(pt.actual_input || pt.target);
    return mode === 'concurrency'
      ? `${pt.concurrency} 并发 · ${inStr}`
      : `${inStr} · ${pt.concurrency} 并发`;
  }

  function renderHudAndLanes(points, runMode) {
    const selBar = $('pt-selector');
    if (!points.length) {
      selBar.innerHTML = '';
      $('hud-point-desc').textContent = '尚未运行测速 · 点击图表或表格任意行可切换查看对应点位';
      ['m-pre-total', 'm-pre-sess', 'm-dec-total', 'm-dec-sess', 'm-ttft', 'm-e2e'].forEach((id) => { $(id).textContent = '—'; });
      $('m-pre-total-sub').textContent = '整组独立冷预填充总吞吐';
      $('m-pre-sess-sub').textContent = '总速度 ÷ 并发 · 单路均速 —';
      $('m-dec-total-sub').textContent = '并行活跃窗口总吞吐 · 合计 —';
      $('m-dec-sess-sub').textContent = '总速度 ÷ 并发 · 单路中位 —';
      $('m-ttft-sub').textContent = 'P95 — · 多轮缓存命中 —';
      $('m-e2e-sub').textContent = '每会话 — · 接受步长 —';
      $('lanes-title').textContent = '每会话 Prefill / Decode 速度分配明细（支持 1–100 会话）';
      $('lanes-summary').innerHTML = '<span class="l-stat">完成第一个测试点后，此处展示全部并发会话（最高 100 路）的分配速度与单路实测明细</span>';
      $('lanes-grid').innerHTML = '';
      return;
    }

    if (ui.selectedPointIdx < 0 || ui.selectedPointIdx >= points.length) {
      ui.selectedPointIdx = points.length - 1;
    }
    const p = points[ui.selectedPointIdx];

    selBar.innerHTML = points.map((item, idx) => {
      const lbl = `#${item.point} ${pointTag(item, runMode)}${item._live ? ' · 采样中' : ''}`;
      return `<button type="button" class="pt-chip ${idx === ui.selectedPointIdx ? 'active' : ''}" data-ptidx="${idx}">${esc(lbl)}</button>`;
    }).join('');

    $('hud-point-desc').textContent = `当前展示：第 #${p.point} 点（${p.concurrency} 并发会话 · 实际输入 ${fmtInt(p.actual_input || p.target)} tokens${p._live ? ' · 实时滚动更新' : ''}）`;

    $('m-pre-total').textContent = fmt(p.prefill_total_tps, 1);
    $('m-pre-total-sub').textContent = `组首字窗口 ${fmtSec(p.group_ttft_s)} · 成功 ${p.success}/${p.concurrency} 路`;

    $('m-pre-sess').textContent = fmt(p.prefill_per_session_tps, 1);
    $('m-pre-sess-sub').textContent = `总速度÷${p.concurrency} · 单路均速 ${fmt(p.prefill_session_mean_tps, 1)}`;

    $('m-dec-total').textContent = fmt(p.decode_total_tps, 1);
    $('m-dec-total-sub').textContent = `各路速率和 ${fmt(p.decode_lane_sum_tps, 1)} · 重叠 ${fmt(p.decode_overlap, 2)}x`;

    $('m-dec-sess').textContent = fmt(p.decode_per_session_tps, 1);
    $('m-dec-sess-sub').textContent = `总速度÷${p.concurrency} · 单路 P50 ${fmt(p.decode_lane_p50_tps, 1)}`;

    $('m-ttft').textContent = fmtSec(p.ttft_mean_s);
    const hitStr = p.cache_hit_pct != null ? `缓存命中 ${fmt(p.cache_hit_pct, 1)}% (${fmtSec(p.cached_ttft_p50_s)})` : '单波冷启动';
    $('m-ttft-sub').textContent = `P95 ${fmtSec(p.ttft_p95_s)} · ${hitStr}`;

    $('m-e2e').textContent = fmt(p.e2e_total_tps, 1);
    const specStr = p.accept_len != null ? `DSpark ${fmt(p.accept_len, 2)} tok/步` : '标准解码';
    $('m-e2e-sub').textContent = `每会话 ${fmt(p.e2e_per_session_tps, 1)} · ${specStr}`;

    $('lanes-title').textContent = `每会话速度明细 · 第 #${p.point} 点（共 ${p.concurrency} 路并发 · 实际输入 ${fmtTokTier(p.actual_input || p.target)}）`;
    const summaryItems = [
      `<span class="l-stat">Prefill 分配均速<b>${fmt(p.prefill_per_session_tps, 1)} tok/s</b></span>`,
      `<span class="l-stat">Decode 分配均速<b>${fmt(p.decode_per_session_tps, 1)} tok/s</b></span>`,
      `<span class="l-stat">单路 Decode 最小 / P10 / P50 / P90 / 最大<b>${fmt(p.decode_lane_min_tps, 1)} / ${fmt(p.decode_lane_p10_tps, 1)} / ${fmt(p.decode_lane_p50_tps, 1)} / ${fmt(p.decode_lane_p90_tps, 1)} / ${fmt(p.decode_lane_max_tps, 1)} tok/s</b></span>`,
      `<span class="l-stat">累计输出<b>${fmtInt(p.total_output_tokens)} tokens</b></span>`
    ];
    if (p.decode_note) {
      summaryItems.push(`<span class="l-stat warn-text">提示：${esc(p.decode_note)}</span>`);
    }
    $('lanes-summary').innerHTML = summaryItems.join('');

    const lanes = Array.isArray(p.lanes) ? p.lanes.slice() : [];
    if (ui.laneSort === 'dec_desc') lanes.sort((a, b) => (b.decode_tps ?? -1) - (a.decode_tps ?? -1));
    else if (ui.laneSort === 'dec_asc') lanes.sort((a, b) => (a.decode_tps ?? 1e12) - (b.decode_tps ?? 1e12));
    else if (ui.laneSort === 'ttft_desc') lanes.sort((a, b) => (b.ttft_s ?? -1) - (a.ttft_s ?? -1));
    else lanes.sort((a, b) => a.lane - b.lane);

    const maxDec = Math.max(1, ...lanes.map((l) => l.decode_tps || 0), p.decode_per_session_tps || 1);
    $('lanes-grid').innerHTML = lanes.map((l) => {
      const code = 'S' + String(l.lane).padStart(3, '0');
      const pctBar = Math.min(100, Math.max(4, Math.round(((l.decode_tps || l.decode_alloc_tps || 0) / maxDec) * 100)));
      const toolTxt = l.tools && l.tools.length ? l.tools.join(' → ') : `共 ${l.turns} 轮采样`;
      return `<div class="lane-item">
        <div class="lane-top">
          <span class="lane-id">${code}</span>
          <span class="lane-tools" title="${esc(toolTxt)}">${esc(toolTxt)}</span>
        </div>
        <div class="bar gpu"><span style="width:${pctBar}%"></span></div>
        <div class="lane-metrics">
          <div><span>Prefill 分配</span><b>${fmt(l.prefill_alloc_tps, 1)}</b></div>
          <div><span>Prefill 单路</span><b>${fmt(l.prefill_ind_tps, 1)}</b></div>
          <div><span>Decode 分配</span><b>${fmt(l.decode_alloc_tps, 1)}</b></div>
          <div><span>Decode 实测</span><b>${fmt(l.decode_tps, 1)}</b></div>
          <div><span>首字 TTFT</span><b>${fmtSec(l.ttft_s)}</b></div>
          <div><span>输出 Token</span><b>${fmtInt(l.output_tokens)}</b></div>
        </div>
      </div>`;
    }).join('');
  }

  function renderTable(points, runMode) {
    const tbody = $('results-tbody');
    const hasDone = points.some((p) => !p._live);
    $('btn-export-csv').disabled = !hasDone;
    $('btn-export-json').disabled = !hasDone;

    if (!points.length) {
      tbody.innerHTML = '<tr><td colspan="11" style="text-align:center;color:var(--muted);padding:24px;font-family:var(--sans)">暂无测速记录</td></tr>';
      return;
    }

    tbody.innerHTML = points.map((p, idx) => {
      const title = runMode === 'concurrency'
        ? `#${p.point} · ${p.concurrency} 并发挡`
        : `#${p.point} · ${fmtTokTier(p.target)} 挡`;
      const badge = p._live
        ? '<span class="pill warn">采样中</span>'
        : (p.failed_requests > 0
          ? `<span class="pill warn">成功 ${p.success}/${p.concurrency}</span>`
          : '<span class="pill ok">已完成</span>');
      return `<tr class="${idx === ui.selectedPointIdx ? 'selected' : ''}" data-rowidx="${idx}">
        <td><strong>${esc(title)}</strong><span class="sub-cell">目标 ${fmtTokTier(p.target)}</span></td>
        <td><b>${p.concurrency}</b> 会话</td>
        <td>${fmtInt(p.actual_input || p.target)}<span class="sub-cell">${fmtTokTier(p.actual_input || p.target)}</span></td>
        <td class="td-pre">${fmt(p.prefill_total_tps, 1)}<span class="sub-cell">组首字 ${fmtSec(p.group_ttft_s)}</span></td>
        <td class="td-pre">${fmt(p.prefill_per_session_tps, 1)}<span class="sub-cell">单路均 ${fmt(p.prefill_session_mean_tps, 1)}</span></td>
        <td class="td-dec">${fmt(p.decode_total_tps, 1)}<span class="sub-cell">各路和 ${fmt(p.decode_lane_sum_tps, 1)}</span></td>
        <td class="td-dec">${fmt(p.decode_per_session_tps, 1)}<span class="sub-cell">P50 ${fmt(p.decode_lane_p50_tps, 1)} [${fmt(p.decode_lane_p10_tps, 0)}–${fmt(p.decode_lane_p90_tps, 0)}]</span></td>
        <td>${fmtSec(p.ttft_mean_s)}<span class="sub-cell">P95 ${fmtSec(p.ttft_p95_s)}</span></td>
        <td>${fmt(p.e2e_total_tps, 1)}<span class="sub-cell">每会话 ${fmt(p.e2e_per_session_tps, 1)}</span></td>
        <td>${p.accept_len != null ? fmt(p.accept_len, 2) + ' tok/步' : '—'}<span class="sub-cell">${p.cache_hit_pct != null ? '命中 ' + fmt(p.cache_hit_pct, 1) + '%' : '冷前缀'}</span></td>
        <td>${badge}</td>
      </tr>`;
    }).join('');
  }

  function renderChart(points, runMode) {
    const svg = $('chart-svg');
    const empty = $('chart-empty');
    const legend = $('chart-legend');
    const tip = $('chart-tip');
    tip.style.display = 'none';

    if (!points.length) {
      svg.innerHTML = '';
      empty.style.display = 'flex';
      legend.innerHTML = '';
      return;
    }
    empty.style.display = 'none';

    const W = 1120, H = 284;
    const padL = 74, padR = 78, padT = 22, padB = 40;
    const plotW = W - padL - padR, plotH = H - padT - padB;

    let leftKey, rightKey, bandLoKey, bandHiKey, leftName, rightName, leftColor, rightColor, unit, dualAxis;
    if (ui.chartTab === 'decode') {
      leftKey = 'decode_total_tps';
      rightKey = 'decode_per_session_tps';
      bandLoKey = 'decode_lane_p10_tps';
      bandHiKey = 'decode_lane_p90_tps';
      leftName = 'Decode 总速度 (左轴)';
      rightName = 'Decode 每会话速度 (右轴)';
      leftColor = 'var(--ok)';
      rightColor = 'var(--accent)';
      unit = 'tok/s';
      dualAxis = true;
    } else if (ui.chartTab === 'prefill') {
      leftKey = 'prefill_total_tps';
      rightKey = 'prefill_per_session_tps';
      bandLoKey = null;
      bandHiKey = null;
      leftName = 'Prefill 总速度 (左轴)';
      rightName = 'Prefill 每会话速度 (右轴)';
      leftColor = 'var(--accent)';
      rightColor = 'var(--warn)';
      unit = 'tok/s';
      dualAxis = true;
    } else if (ui.chartTab === 'ttft') {
      leftKey = 'ttft_mean_s';
      rightKey = 'ttft_p95_s';
      bandLoKey = null;
      bandHiKey = null;
      leftName = '冷首字均值 TTFT';
      rightName = '冷首字 P95 TTFT';
      leftColor = 'var(--warn)';
      rightColor = 'var(--danger)';
      unit = 's';
      dualAxis = false;
    } else {
      leftKey = 'e2e_total_tps';
      rightKey = 'e2e_per_session_tps';
      bandLoKey = null;
      bandHiKey = null;
      leftName = '端到端总吞吐 (左轴)';
      rightName = '端到端每会话吞吐 (右轴)';
      leftColor = 'var(--accent)';
      rightColor = 'var(--ok)';
      unit = 'tok/s';
      dualAxis = true;
    }

    legend.innerHTML = [
      `<span class="leg-item"><i class="leg-swatch" style="background:${leftColor}"></i>${esc(leftName)}</span>`,
      `<span class="leg-item"><i class="leg-swatch" style="background:${rightColor}"></i>${esc(rightName)}</span>`,
      bandLoKey ? `<span class="leg-item"><i class="leg-band"></i>单会话 P10–P90 区间 (右轴)</span>` : ''
    ].join('');

    const n = points.length;
    const xAt = (i) => (n === 1 ? padL + plotW / 2 : padL + (i / (n - 1)) * plotW);

    const leftVals = points.map((p) => p[leftKey]).filter(Number.isFinite);
    const rightVals = points.map((p) => p[rightKey]).filter(Number.isFinite);
    const bandVals = bandHiKey ? points.map((p) => p[bandHiKey]).filter(Number.isFinite) : [];

    const maxL = Math.max(1e-3, ...(dualAxis ? leftVals : [...leftVals, ...rightVals])) * 1.16;
    const maxR = dualAxis ? Math.max(1e-3, ...rightVals, ...bandVals) * 1.18 : maxL;

    const yL = (v) => padT + plotH - Math.max(0, Math.min(1, v / maxL)) * plotH;
    const yR = (v) => padT + plotH - Math.max(0, Math.min(1, v / maxR)) * plotH;

    const parts = [];
    for (let g = 0; g <= 4; g++) {
      const gy = padT + (g / 4) * plotH;
      const vL = maxL * (1 - g / 4);
      const vR = maxR * (1 - g / 4);
      parts.push(`<line x1="${padL}" y1="${gy.toFixed(1)}" x2="${W - padR}" y2="${gy.toFixed(1)}" stroke="var(--border)" stroke-width="1"/>`);
      parts.push(`<text x="${padL - 10}" y="${(gy + 4).toFixed(1)}" text-anchor="end" fill="${leftColor}" font-size="11" font-family="var(--mono)">${unit === 's' ? fmt(vL, 2) + 's' : fmtInt(vL)}</text>`);
      if (dualAxis) {
        parts.push(`<text x="${W - padR + 10}" y="${(gy + 4).toFixed(1)}" text-anchor="start" fill="${rightColor}" font-size="11" font-family="var(--mono)">${fmt(vR, vR >= 100 ? 0 : 1)}</text>`);
      }
    }

    points.forEach((p, i) => {
      const cx = xAt(i);
      const xLbl = runMode === 'concurrency' ? `${p.concurrency}并发` : fmtTokTier(p.actual_input || p.target);
      parts.push(`<line x1="${cx.toFixed(1)}" y1="${padT}" x2="${cx.toFixed(1)}" y2="${padT + plotH}" stroke="var(--border)" stroke-dasharray="3 3"/>`);
      parts.push(`<text x="${cx.toFixed(1)}" y="${H - 12}" text-anchor="middle" fill="var(--muted)" font-size="11.5" font-family="var(--mono)">${esc(xLbl)}</text>`);
    });

    if (bandLoKey && bandHiKey && points.length >= 2) {
      const validBand = points.map((p, i) => ({ i, lo: p[bandLoKey], hi: p[bandHiKey] }))
        .filter((item) => Number.isFinite(item.lo) && Number.isFinite(item.hi));
      if (validBand.length >= 2) {
        const upper = validBand.map((b) => `${xAt(b.i).toFixed(1)},${yR(b.hi).toFixed(1)}`);
        const lower = validBand.slice().reverse().map((b) => `${xAt(b.i).toFixed(1)},${yR(b.lo).toFixed(1)}`);
        parts.push(`<polygon points="${[...upper, ...lower].join(' ')}" fill="var(--accent)" fill-opacity="0.13" stroke="none"/>`);
      }
    }

    function drawSeries(key, yFn, color, dashed) {
      const coords = [];
      points.forEach((p, i) => {
        if (Number.isFinite(p[key])) coords.push({ i, x: xAt(i), y: yFn(p[key]), v: p[key] });
      });
      if (coords.length >= 2) {
        const d = coords.map((c, k) => `${k === 0 ? 'M' : 'L'}${c.x.toFixed(1)},${c.y.toFixed(1)}`).join(' ');
        parts.push(`<path d="${d}" fill="none" stroke="${color}" stroke-width="${dashed ? '2.2' : '2.8'}" ${dashed ? 'stroke-dasharray="6 4"' : ''}/>`);
      }
      coords.forEach((c) => {
        const sel = c.i === ui.selectedPointIdx;
        parts.push(`<circle cx="${c.x.toFixed(1)}" cy="${c.y.toFixed(1)}" r="${sel ? 6 : 4.2}" fill="${color}" stroke="var(--surface)" stroke-width="2"/>`);
      });
    }

    drawSeries(rightKey, yR, rightColor, true);
    drawSeries(leftKey, yL, leftColor, false);

    const colW = n === 1 ? plotW : Math.max(28, plotW / n);
    points.forEach((p, i) => {
      const cx = xAt(i);
      parts.push(`<rect class="pt-hit" data-ptidx="${i}" x="${(cx - colW / 2).toFixed(1)}" y="${padT}" width="${colW.toFixed(1)}" height="${plotH}" fill="transparent" style="cursor:pointer"/>`);
    });

    svg.innerHTML = parts.join('');

    svg.querySelectorAll('.pt-hit').forEach((rect) => {
      rect.addEventListener('mouseenter', (ev) => {
        const idx = Number(rect.dataset.ptidx);
        const p = points[idx];
        if (!p) return;
        tip.innerHTML = `
          <strong>#${p.point} · ${p.concurrency} 并发 · 输入 ${fmtTokTier(p.actual_input || p.target)}</strong>
          <div class="tip-row"><span>Prefill 总速度</span><b class="td-pre">${fmt(p.prefill_total_tps, 1)} tok/s</b></div>
          <div class="tip-row"><span>Prefill 每会话</span><b class="td-pre">${fmt(p.prefill_per_session_tps, 1)} tok/s</b></div>
          <div class="tip-row"><span>Decode 总速度</span><b class="td-dec">${fmt(p.decode_total_tps, 1)} tok/s</b></div>
          <div class="tip-row"><span>Decode 每会话</span><b class="td-dec">${fmt(p.decode_per_session_tps, 1)} tok/s</b></div>
          <div class="tip-row"><span>冷首字 TTFT (均/P95)</span><b>${fmtSec(p.ttft_mean_s)} / ${fmtSec(p.ttft_p95_s)}</b></div>
          <div class="tip-row"><span>端到端总吞吐</span><b>${fmt(p.e2e_total_tps, 1)} tok/s</b></div>
        `;
        tip.style.display = 'block';
        const stageRect = $('chart-stage').getBoundingClientRect();
        const relX = ev.clientX - stageRect.left;
        const leftPx = relX > stageRect.width * 0.62 ? Math.max(10, relX - 255) : Math.min(stageRect.width - 255, relX + 16);
        tip.style.left = leftPx + 'px';
        tip.style.top = '16px';
      });
      rect.addEventListener('mouseleave', () => {
        tip.style.display = 'none';
      });
      rect.addEventListener('click', () => {
        ui.selectedPointIdx = Number(rect.dataset.ptidx);
        renderAllFromStatus();
      });
    });
  }

  function renderAllFromStatus() {
    const st = ui.status || { state: 'idle', results: [] };
    const runMode = st?.options?.mode || ui.mode;
    const points = getDisplayPoints(st);
    renderProgress(st);
    renderHudAndLanes(points, runMode);
    renderChart(points, runMode);
    renderTable(points, runMode);
    syncControls();
  }

  async function pollStatus() {
    clearTimeout(ui.pollTimer);
    let nextMs = 3500;
    try {
      const resp = await fetch('/api/benchmark/status', { cache: 'no-store' });
      if (resp.ok) {
        const prevCount = getDisplayPoints(ui.status).length;
        ui.status = await resp.json();
        const curPoints = getDisplayPoints(ui.status);
        if (curPoints.length > prevCount) {
          ui.selectedPointIdx = curPoints.length - 1;
        }
        renderAllFromStatus();
        if (['preparing', 'running', 'stopping'].includes(ui.status.state)) {
          nextMs = 1000;
        }
      }
    } catch (_e) {
      nextMs = 4000;
    } finally {
      ui.pollTimer = setTimeout(pollStatus, nextMs);
    }
  }

  function downloadBlob(filename, mime, content) {
    const blob = new Blob([content], { type: mime });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = filename;
    document.body.appendChild(a);
    a.click();
    a.remove();
    setTimeout(() => URL.revokeObjectURL(url), 1000);
  }

  document.querySelectorAll('.mode-btn').forEach((btn) => {
    btn.addEventListener('click', () => {
      ui.mode = btn.dataset.mode;
      syncControls();
      schedulePlan();
    });
  });

  document.querySelectorAll('#conc-grid .pill-btn').forEach((btn) => {
    btn.addEventListener('click', () => {
      const c = Number(btn.dataset.conc);
      if (ui.mode === 'concurrency') {
        if (ui.concurrencies.includes(c)) {
          if (ui.concurrencies.length > 1) ui.concurrencies = ui.concurrencies.filter((x) => x !== c);
        } else {
          ui.concurrencies = [...ui.concurrencies, c].sort((a, b) => a - b);
        }
      } else {
        ui.concurrency = c;
      }
      syncControls();
      schedulePlan();
    });
  });

  document.querySelectorAll('#conc-quick button').forEach((btn) => {
    btn.addEventListener('click', () => {
      const p = btn.dataset.preset;
      if (p === 'all') ui.concurrencies = ALL_CONCS.slice();
      else if (p === 'low') ui.concurrencies = [1, 4, 8, 16, 32];
      else if (p === 'high') ui.concurrencies = [32, 64, 100];
      syncControls();
      schedulePlan();
    });
  });

  document.querySelectorAll('#input-grid .pill-btn').forEach((btn) => {
    btn.addEventListener('click', () => {
      ui.maximum = Number(btn.dataset.max);
      syncControls();
      schedulePlan();
    });
  });

  document.querySelectorAll('#points-grid .pill-btn').forEach((btn) => {
    btn.addEventListener('click', () => {
      ui.points = Number(btn.dataset.pts);
      syncControls();
      schedulePlan();
    });
  });

  $('sel-output').addEventListener('change', (e) => { ui.output = Number(e.target.value); schedulePlan(); });
  $('sel-strategy').addEventListener('change', (e) => { ui.strategy = e.target.value; schedulePlan(); });
  $('sel-turns').addEventListener('change', (e) => { ui.turns = Number(e.target.value); schedulePlan(); });
  $('btn-refresh-cap').addEventListener('click', refreshPlan);

  document.querySelectorAll('.c-tab').forEach((tab) => {
    tab.addEventListener('click', () => {
      ui.chartTab = tab.dataset.chart;
      document.querySelectorAll('.c-tab').forEach((t) => t.classList.toggle('active', t === tab));
      renderAllFromStatus();
    });
  });

  $('pt-selector').addEventListener('click', (e) => {
    const chip = e.target.closest('[data-ptidx]');
    if (!chip) return;
    ui.selectedPointIdx = Number(chip.dataset.ptidx);
    renderAllFromStatus();
  });

  $('results-tbody').addEventListener('click', (e) => {
    const tr = e.target.closest('tr[data-rowidx]');
    if (!tr) return;
    ui.selectedPointIdx = Number(tr.dataset.rowidx);
    renderAllFromStatus();
  });

  $('lanes-sort').addEventListener('change', (e) => {
    ui.laneSort = e.target.value;
    renderAllFromStatus();
  });

  $('bench-form').addEventListener('submit', async (e) => {
    e.preventDefault();
    showError('');
    try {
      const resp = await fetch('/api/benchmark/start', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/x-www-form-urlencoded;charset=UTF-8',
          'X-Requested-With': 'fetch'
        },
        body: buildFormParams(true).toString()
      });
      const data = await resp.json();
      if (!resp.ok) {
        showError(data.error || '启动测速失败');
        refreshPlan();
        return;
      }
      ui.selectedPointIdx = -1;
      ui.status = data;
      renderAllFromStatus();
      pollStatus();
    } catch (err) {
      showError('请求启动接口失败：' + err.message);
    }
  });

  $('btn-stop').addEventListener('click', async () => {
    showError('');
    try {
      const body = new URLSearchParams();
      if (ui.status?.id) body.set('id', ui.status.id);
      const resp = await fetch('/api/benchmark/stop', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/x-www-form-urlencoded;charset=UTF-8',
          'X-Requested-With': 'fetch'
        },
        body: body.toString()
      });
      const data = await resp.json();
      if (!resp.ok) {
        showError(data.error || '停止测速失败');
        return;
      }
      ui.status = data;
      renderAllFromStatus();
    } catch (err) {
      showError('停止请求失败：' + err.message);
    }
  });

  $('btn-clear').addEventListener('click', async () => {
    showError('');
    try {
      const body = new URLSearchParams();
      if (ui.status?.id) body.set('id', ui.status.id);
      const resp = await fetch('/api/benchmark/clear', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/x-www-form-urlencoded;charset=UTF-8',
          'X-Requested-With': 'fetch'
        },
        body: body.toString()
      });
      const data = await resp.json();
      if (!resp.ok) {
        showError(data.error || '清除结果失败');
        return;
      }
      ui.selectedPointIdx = -1;
      ui.status = data;
      renderAllFromStatus();
    } catch (err) {
      showError('清除请求失败：' + err.message);
    }
  });

  $('btn-export-json').addEventListener('click', () => {
    if (!ui.status?.results?.length) return;
    const stamp = ui.status.id || 'benchmark';
    downloadBlob(`xtier-86kd-agent-bench-${stamp}.json`, 'application/json;charset=utf-8', JSON.stringify(ui.status, null, 2));
  });

  $('btn-export-csv').addEventListener('click', () => {
    const rows = ui.status?.results || [];
    if (!rows.length) return;
    const headers = [
      'point', 'concurrency', 'target', 'actual_input',
      'prefill_total_tps', 'prefill_per_session_tps', 'prefill_session_mean_tps',
      'decode_total_tps', 'decode_per_session_tps', 'decode_lane_sum_tps',
      'decode_lane_p10_tps', 'decode_lane_p50_tps', 'decode_lane_p90_tps',
      'ttft_mean_s', 'ttft_p95_s', 'cached_ttft_p50_s', 'cache_hit_pct',
      'e2e_total_tps', 'e2e_per_session_tps', 'accept_len', 'step_ms'
    ];
    const lines = [headers.join(',')];
    rows.forEach((r) => {
      lines.push(headers.map((h) => (r[h] == null ? '' : String(r[h]))).join(','));
    });
    const stamp = ui.status.id || 'benchmark';
    downloadBlob(`xtier-86kd-agent-bench-${stamp}.csv`, 'text/csv;charset=utf-8', '\uFEFF' + lines.join('\n'));
  });

  syncControls();
  refreshPlan();
  pollStatus();
})();
"""


def page_html() -> str:
    return (
        "<!doctype html><html lang=\"zh-CN\"><head><meta charset=\"utf-8\">"
        "<meta name=\"viewport\" content=\"width=device-width,initial-scale=1\">"
        "<title>API 测速 · TF500</title><style>"
        + CSS
        + "</style></head><body>"
        + BODY
        + "<script>"
        + JS
        + "</script></body></html>"
    )
