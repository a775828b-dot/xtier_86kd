"""Loopback-only management page for the 8-GPU API proxy (/proxy)."""

from __future__ import annotations

import json
import time
import urllib.error
import urllib.request

import tf500_api_proxy as proxy
import tf500_proxy_keys as keys

MODEL = "DeepSeek-V4.1-Flash-1017"


def data():
    traffic = keys.usage()
    today = time.strftime("%Y-%m-%d")
    return {
        "status": proxy.status(),
        "base_url": f"http://{proxy.HOST}:{proxy.PORT}/v1",
        "model": MODEL,
        "keys": keys.public_entries(),
        "today": traffic["days"].get(today, {}),
        "total": {
            "requests": sum(row["requests"] for row in traffic["days"].values()),
            "input": sum(row["input"] for row in traffic["days"].values()),
            "output": sum(row["output"] for row in traffic["days"].values()),
        },
        "days": traffic["days"],
        "recent": traffic["recent"][-30:][::-1],
    }


def selftest():
    result = {}
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    started = time.monotonic()
    for label, headers in (
        ("authenticated", {"Authorization": "Bearer " + keys.MASTER_FILE.read_text().strip()}),
        ("unauthenticated", {}),
    ):
        request = urllib.request.Request(f"http://{proxy.HOST}:{proxy.PORT}/v1/models", headers=headers)
        try:
            with opener.open(request, timeout=5) as response:
                result[label] = response.status
                if label == "authenticated":
                    result["models"] = [
                        str(row.get("id", ""))
                        for row in json.load(response).get("data", [])
                        if isinstance(row, dict)
                    ]
        except urllib.error.HTTPError as exc:
            result[label] = exc.code
        except (OSError, urllib.error.URLError, ValueError):
            result[label] = 0
    result["elapsed_ms"] = round((time.monotonic() - started) * 1000)
    return result


CSS = r"""
.proxy-stats{display:grid;grid-template-columns:repeat(auto-fit,minmax(175px,1fr));gap:12px}
.proxy-stat{border:1px solid var(--border);border-radius:var(--radius-sm);padding:11px 13px;background:var(--surface-2);min-width:0}
.proxy-stat span{display:block;color:var(--muted);font-size:12px}
.proxy-stat b{display:block;font-size:18px;margin-top:3px;font-variant-numeric:tabular-nums;color:var(--text)}
.proxy-actions{display:flex;gap:10px;flex-wrap:wrap;align-items:center;margin-top:14px}
.selftest-out{font-size:13px;color:var(--text-2)}
.key-create{display:flex;gap:10px;flex-wrap:wrap;align-items:flex-end;margin-bottom:14px}
.key-create label{min-width:150px}
.key-create input[name=name]{min-width:230px}
.new-key{margin:0 0 14px;padding:12px 14px;border-radius:10px;background:var(--ok-soft);border:1px solid color-mix(in srgb,var(--ok) 32%,transparent)}
.new-key[hidden]{display:none}
.new-key strong{display:block;color:var(--ok);margin-bottom:6px;font-size:13px}
.new-key code{display:block;padding:8px 11px;margin-bottom:8px;border-radius:7px;background:var(--surface);border:1px solid var(--border);overflow-wrap:anywhere}
.message{min-height:20px;margin:0 0 12px;font-size:13px;font-weight:600;color:var(--ok)}
.message:empty{display:none}
.message.error{color:var(--danger)}
.usage-stack{display:grid;grid-template-columns:minmax(0,1fr);gap:16px}
.usage-stack h3{margin-bottom:8px}
.usage-stack>div:last-child table{min-width:860px}
ul.rules{margin:0;padding-left:20px;color:var(--text-2);font-size:13px;line-height:1.7}
ul.rules li{margin:4px 0}
"""

JS = r"""
(() => {
  const $ = (id) => document.getElementById(id);
  const fmt = (n) => Number(n || 0).toLocaleString('zh-CN');
  const stamp = (s) => s ? new Date(s * 1000).toLocaleString('zh-CN') : '—';
  const expiry = (s) => s ? stamp(s) : '永久';

  async function writeClip(value) {
    try {
      await navigator.clipboard.writeText(value);
    } catch (_err) {
      const t = document.createElement('textarea');
      t.value = value;
      t.style.position = 'fixed';
      t.style.opacity = '0';
      document.body.appendChild(t);
      t.select();
      document.execCommand('copy');
      t.remove();
    }
  }

  document.querySelectorAll('[data-copy-target]').forEach((btn) => {
    btn.addEventListener('click', async () => {
      const target = $(btn.dataset.copyTarget);
      if (!target) return;
      await writeClip(target.textContent.trim());
      const old = btn.textContent;
      btn.textContent = '已复制';
      btn.classList.add('done');
      setTimeout(() => { btn.textContent = old; btn.classList.remove('done'); }, 1200);
    });
  });

  function row(table, values, action, numCols = []) {
    const tr = document.createElement('tr');
    values.forEach((value, idx) => {
      const td = document.createElement('td');
      if (numCols.includes(idx)) td.className = 'num';
      td.textContent = String(value);
      tr.append(td);
    });
    if (action) {
      const td = document.createElement('td');
      td.append(action);
      tr.append(td);
    }
    table.append(tr);
  }

  async function request(path, body) {
    const r = await fetch(
      path,
      body
        ? { method: 'POST', headers: { 'X-Requested-With': 'fetch' }, body: new URLSearchParams(body), cache: 'no-store' }
        : { cache: 'no-store' }
    );
    const d = await r.json();
    if (!r.ok) throw Error(d.error || 'HTTP ' + r.status);
    return d;
  }

  function message(value, error = false) {
    $('message').textContent = value;
    $('message').classList.toggle('error', error);
  }

  async function refresh() {
    try {
      const d = await request('/api/proxy/admin'), s = d.status;
      const pOk = s.proxy === 'online', bOk = s.backend === 'online';
      $('proxy-state').textContent = pOk ? '在线' : '离线';
      $('proxy-state').className = pOk ? 'ok-text' : 'danger-text';
      $('backend-state').textContent = bOk ? '在线' : '离线';
      $('backend-state').className = bOk ? 'ok-text' : 'danger-text';
      $('backend-label').textContent = '推理后端 · ' + (s.backend_label || '待选择') + (s.backend_port ? ' :' + s.backend_port : '');
      $('today-requests').textContent = fmt(d.today.requests) + ' 次';
      $('total-requests').textContent = fmt(d.total.requests) + ' 次';
      $('total-tokens').textContent = fmt((d.total.input || 0) + (d.total.output || 0));
      $('base-url').textContent = d.base_url;
      $('model-name').textContent = d.model;

      const names = {};
      $('keys').replaceChildren();
      for (const key of d.keys) {
        names[key.id] = key.name;
        let action;
        if (!key.master) {
          action = document.createElement('button');
          action.className = 'danger';
          action.textContent = '删除';
          action.onclick = async () => {
            if (!confirm('删除这个 Key？使用它的客户端会立即失效。')) return;
            try {
              await request('/api/proxy/keys/delete', { id: key.id });
              message('已删除 ' + key.name);
              refresh();
            } catch (e) {
              message(e.message, true);
            }
          };
        } else {
          action = document.createElement('button');
          action.type = 'button';
          action.className = 'secondary';
          action.textContent = '复制 Key';
          action.title = '复制系统主 Key';
          action.onclick = async () => {
            action.disabled = true;
            try {
              const res = await request('/api/proxy/master-key/copy', { copy: '1' });
              await writeClip(res.key);
              message('已复制系统主 Key');
            } catch (e) {
              message('复制失败：' + e.message, true);
            } finally {
              action.disabled = false;
            }
          };
        }
        const used = key.usage || {};
        row(
          $('keys'),
          [
            key.name,
            key.masked_key,
            expiry(key.expires_at),
            key.max_requests ? fmt(used.requests) + ' / ' + fmt(key.max_requests) : fmt(used.requests) + ' / 不限',
            fmt(used.input) + ' / ' + fmt(used.output),
            stamp(used.last || 0)
          ],
          action,
          [3, 4]
        );
      }

      $('days').replaceChildren();
      const dayEntries = Object.entries(d.days || {}).sort((a, b) => b[0].localeCompare(a[0])).slice(0, 14);
      if (!dayEntries.length) {
        $('days').innerHTML = '<tr><td colspan="5" class="muted">暂无按天统计记录</td></tr>';
      } else {
        for (const [day, v] of dayEntries) row($('days'), [day, fmt(v.requests), fmt(v.input), fmt(v.output), fmt(v.errors)], null, [1, 2, 3, 4]);
      }

      $('recent').replaceChildren();
      if (!(d.recent || []).length) {
        $('recent').innerHTML = '<tr><td colspan="8" class="muted">暂无最近调用记录</td></tr>';
      } else {
        for (const v of d.recent) {
          row(
            $('recent'),
            [stamp(v.time), names[v.key_id] || '已删除 Key', v.model || '—', v.path, v.status, fmt(v.input), fmt(v.output), v.elapsed_s + ' s'],
            null,
            [4, 5, 6, 7]
          );
        }
      }
    } catch (e) {
      message('读取配置失败：' + e.message, true);
    }
  }

  $('create-form').onsubmit = async (e) => {
    e.preventDefault();
    const button = $('create-button');
    button.disabled = true;
    button.classList.add('is-loading');
    try {
      const d = await request('/api/proxy/keys', Object.fromEntries(new FormData(e.target)));
      $('new-key').hidden = false;
      $('new-key-value').textContent = d.key;
      message('已创建 ' + d.name + '；完整 Key 只显示这一次，请立即保存。');
      e.target.reset();
      await refresh();
    } catch (err) {
      message(err.message, true);
    } finally {
      button.disabled = false;
      button.classList.remove('is-loading');
    }
  };

  $('copy-key').onclick = async () => {
    try {
      await writeClip($('new-key-value').textContent);
      message('已复制新 Key');
    } catch (_e) {
      message('复制失败，请手动选择 Key', true);
    }
  };

  $('selftest').onclick = async () => {
    const button = $('selftest');
    button.disabled = true;
    button.classList.add('is-loading');
    try {
      const d = await request('/api/proxy/selftest');
      $('selftest-result').textContent =
        '带 Key：HTTP ' + d.authenticated +
        '；不带 Key：HTTP ' + d.unauthenticated +
        '；耗时 ' + d.elapsed_ms + ' ms' +
        (d.models?.length ? '；模型 ' + d.models.join('、') : (d.authenticated === 502 ? '（后端模型未运行）' : ''));
    } catch (e) {
      $('selftest-result').textContent = '自检失败：' + e.message;
    } finally {
      button.disabled = false;
      button.classList.remove('is-loading');
    }
  };

  refresh();
  setInterval(refresh, 10000);
})();
"""


def page_html():
    return f"""<!doctype html><html lang="zh-CN"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>API 代理 · TF500</title><style>{CSS}</style></head><body><main>
<div class="page-head"><div><h1>API 代理</h1><p>OpenAI 兼容入口 :8000：接入地址、API Key 管理、按 Key 请求用量统计与连通性自检。</p></div></div>

<section class="card">
  <div class="card-head"><div><h2>代理状态</h2><p>客户端 → :8000 代理 → 自动转发至当前健康的 TP8 (:30000) 推理后端。</p></div></div>
  <div class="proxy-stats">
    <div class="proxy-stat"><span>API 代理 :8000</span><b id="proxy-state">检测中</b></div>
    <div class="proxy-stat"><span id="backend-label">推理后端</span><b id="backend-state">检测中</b></div>
    <div class="proxy-stat"><span>今日请求</span><b id="today-requests">—</b></div>
    <div class="proxy-stat"><span>累计请求</span><b id="total-requests">—</b></div>
    <div class="proxy-stat"><span>累计输入 + 输出 tokens</span><b id="total-tokens">—</b></div>
  </div>
  <div class="proxy-actions">
    <button id="selftest" class="secondary" type="button">连通性自检</button>
    <span id="selftest-result" class="selftest-out"></span>
  </div>
</section>

<section class="card">
  <div class="card-head"><div><h2>接入地址</h2><p>在任意 OpenAI 兼容客户端（Claude Code / Cursor / OpenCode 等）填写 Base URL、模型名与下方创建的 API Key。</p></div></div>
  <div class="kv-grid">
    <div class="kv">
      <label>客户端 Base URL</label>
      <div class="field-copy"><code id="base-url" class="config">—</code><button type="button" class="copy-inline-btn" data-copy-target="base-url">复制</button></div>
    </div>
    <div class="kv">
      <label>当前模型名</label>
      <div class="field-copy"><code id="model-name" class="config">—</code><button type="button" class="copy-inline-btn" data-copy-target="model-name">复制</button></div>
    </div>
  </div>
</section>

<section class="card">
  <div class="card-head"><div><h2>API Key 管理</h2><p>有效期可选永久、1、7、30、90 天；请求额度为最大调用次数，0 表示不限。系统主 Key 仅供内部测速与管理复制，不可删除。</p></div></div>
  <form id="create-form" class="key-create">
    <label>名称<input name="name" maxlength="40" required placeholder="例如 本机客户端、OpenCode"></label>
    <label>有效期<select name="days"><option value="0">永久有效</option><option value="1">1 天</option><option value="7">7 天</option><option value="30">30 天</option><option value="90">90 天</option></select></label>
    <label>请求次数额度（0=不限）<input name="max_requests" type="number" min="0" max="100000000" value="0" required></label>
    <button id="create-button" type="submit">新建 Key</button>
  </form>
  <div id="new-key" class="new-key" hidden>
    <strong>已创建新 Key（完整值仅显示这一次，请立即复制保存）</strong>
    <code id="new-key-value"></code>
    <button id="copy-key" class="secondary" type="button">复制 Key</button>
  </div>
  <p id="message" class="message" role="status"></p>
  <div class="table-wrap">
    <table>
      <thead><tr><th>名称</th><th>Key 掩码</th><th>有效期</th><th class="num">已用 / 额度</th><th class="num">输入 / 输出 tokens</th><th>最近使用</th><th>操作</th></tr></thead>
      <tbody id="keys"></tbody>
    </table>
  </div>
</section>

<section class="card">
  <div class="card-head"><div><h2>用量统计</h2><p>代理仅记录时间、Key ID、接口路径、模型名、HTTP 状态、耗时与返回的 usage tokens；不记录任何请求或响应正文。</p></div></div>
  <div class="usage-stack">
    <div>
      <h3>按天汇总（最近 14 天）</h3>
      <div class="table-wrap"><table><thead><tr><th>日期</th><th class="num">请求</th><th class="num">输入 tokens</th><th class="num">输出 tokens</th><th class="num">失败</th></tr></thead><tbody id="days"></tbody></table></div>
    </div>
    <div>
      <h3>最近调用</h3>
      <div class="table-wrap"><table><thead><tr><th>时间</th><th>Key</th><th>模型</th><th>接口</th><th class="num">状态</th><th class="num">输入</th><th class="num">输出</th><th class="num">耗时</th></tr></thead><tbody id="recent"></tbody></table></div>
    </div>
  </div>
</section>

<section class="card">
  <div class="card-head"><div><h2>转发规则</h2></div></div>
  <ul class="rules">
    <li>仅转发 <code>/v1/</code> 下的标准接口路径；流式 SSE 响应逐块零缓冲转发。</li>
    <li>客户端支持通过 <code>Authorization: Bearer &lt;Key&gt;</code> 或 <code>x-api-key</code> 传参，代理校验后统一改写为后端内部密钥。</li>
    <li>自动探活并接入健康的 TP8 (<code>:30000</code>) 推理服务，不会主动启停模型。</li>
    <li>Key 失效或过期返回 <code>401</code>；请求次数额度耗尽返回 <code>402</code>（正在进行中的流式请求照常完成）。</li>
  </ul>
</section>
</main><script>{JS}</script></body></html>"""
