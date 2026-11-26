const app = document.querySelector('#app');
const taskNav = document.querySelector('#task-nav');
const icons = {classification: '◩', regression: '╱', clustering: '⠿', dimensionality: '◇', anomaly: '◎', sequence: '∿', recommendation: '☆'};
const labels = {
  predicted_label: '预测类别', probability_1: '类别 1 概率', label: '生成标签',
  prediction: '预测值', target: '生成目标值', cluster: '分组', distance: '到中心的距离',
  source_group: '生成分组', z1: '降维坐标 1', z2: '降维坐标 2', reconstruction_error: '重建误差',
  is_injected_anomaly: '注入异常', next_value: '下一步预测值', actual_next: '下一步真实值',
  user_id: '用户编号', item_id: '物品编号', predicted_rating: '预测评分', rating: '生成评分',
};
for (let i = 1; i <= 4; i++) labels[`reconstructed_x${i}`] = `重建特征 ${i}`;
const state = new Map();
let catalog;
let byId;
let dataTask;

function escapeHTML(value) {
  return String(value).replace(/[&<>"']/g, char => ({'&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;'})[char]);
}

function route() {
  const hash = location.hash.slice(1);
  if (byId.has(hash)) return {page: 'task', task: byId.get(hash)};
  if (hash === 'data' || hash.startsWith('data/')) {
    return {page: 'data', task: byId.get(hash.slice(5)) || dataTask};
  }
  return {page: 'home'};
}

function icon(task) {
  return `<span class="task-icon ${task.id}" aria-hidden="true">${icons[task.id] || '◇'}</span>`;
}

function notice(message, role = 'status') {
  return message ? `<p class="message ${role === 'alert' ? 'error' : ''}" role="${role}">${escapeHTML(message)}</p>` : '';
}

function render() {
  const {page, task} = route();
  if (page === 'data') dataTask = task;
  const active = page === 'task' ? task.id : page;
  for (const link of document.querySelectorAll('nav a')) {
    if (link.hash === `#${active}`) link.setAttribute('aria-current', 'page');
    else link.removeAttribute('aria-current');
  }
  const missing = catalog.missing_dependencies;
  const warning = notice(missing.length ? `运行模型和生成数据前，请安装 Python 依赖 ${missing.join('、')}，然后刷新页面。` : '');
  app.innerHTML = warning + (page === 'home' ? home() : page === 'data' ? dataPage(task) : taskPage(task));
}

function home() {
  const count = catalog.tasks.flatMap(task => task.models).filter(model => model.has_weights).length;
  return `<header class="topline"><span>工作台 / 任务目录</span><span class="chip"><i class="local-dot"></i>本地运行</span></header>
    <section class="hero">
      <div><h1>选择一个任务</h1><p>使用本地模型分析表格或时间序列。可以先运行示例，再上传自己的数据。</p>
      <a class="button primary" href="#data">生成示例数据</a></div>
      <div class="hero-art" aria-hidden="true"><span></span><span></span><span></span><span></span><span></span><span></span><span></span><span></span><span></span><span></span><span></span><span></span></div>
    </section>
    <div class="section-heading"><h2>任务目录</h2><p>${catalog.tasks.length} 类任务 · ${count} 个本地模型或聚类结果</p></div>
    <div class="task-grid">${catalog.tasks.map(task => `<a class="task-card" href="#${task.id}">${icon(task)}<span class="method">${escapeHTML(task.method)}</span><h3>${escapeHTML(task.title)}</h3><p>${escapeHTML(task.summary)}</p><span class="card-action">打开任务</span></a>`).join('')}</div>`;
}

function modelPicker(task, selected, busy) {
  if (task.models.length === 1) return `<div class="model-detail"><span>当前模型</span><strong>${escapeHTML(task.models[0].name)}</strong></div>`;
  return `<label class="model-detail" for="model-picker"><span>选择模型</span><select id="model-picker" ${busy ? 'disabled' : ''}>${task.models.map(model => `<option value="${model.id}" ${model.id === selected ? 'selected' : ''}>${escapeHTML(model.name)}</option>`).join('')}</select></label>`;
}

function resultPanel(result, task, generation) {
  if (!result) return `<section class="panel result-panel"><h2>${generation ? '数据预览' : '运行结果'}</h2><div class="empty"><span aria-hidden="true">▤</span><p>${generation ? '生成后可预览数据并下载 CSV。' : '运行完成后，结果会显示在这里。'}</p></div></section>`;
  const columns = result.columns;
  const rows = result.rows;
  const shown = rows.slice(0, 10);
  return `<section class="panel result-panel" aria-label="${generation ? '数据预览' : '运行结果'}">
    <div class="section-heading"><div><h2>${generation ? '数据预览' : '运行结果'}</h2><p role="status">${rows.length} 行${rows.length > shown.length ? ` · 预览前 ${shown.length} 行` : ''}</p></div>
      <div class="actions"><button data-action="download" ${rows.length ? '' : 'disabled'}>下载 CSV</button>${generation ? `<button class="primary" data-action="transfer" ${rows.length ? '' : 'disabled'}>在任务中使用</button>` : ''}</div></div>
    ${!generation && task.id === 'anomaly' ? '<button class="sort" data-action="sort">按重建误差从高到低排序</button>' : ''}
    ${!generation ? scatter(task, result) : ''}
    <div class="table-scroll" tabindex="0" role="region" aria-label="结果表格"><table><thead><tr><th scope="col">行</th>${columns.map(column => `<th scope="col" title="${escapeHTML(column)}">${escapeHTML(labels[column] || column)}</th>`).join('')}</tr></thead>
      <tbody>${shown.map((row, index) => `<tr><th scope="row">${index + 1}</th>${row.map(value => `<td title="${value}">${format(value)}</td>`).join('')}</tr>`).join('')}</tbody></table></div>
    ${rows.length ? '' : notice('本次没有生成评分记录。可以调高评分覆盖率后重试。')}</section>`;
}

function format(value) {
  if (value !== 0 && Math.abs(value) < 0.0001) return value.toExponential(2);
  return value.toLocaleString('zh-CN', {maximumFractionDigits: 5, useGrouping: false});
}

function scatter(task, result) {
  if (!['clustering', 'dimensionality'].includes(task.id)) return '';
  const x = result.columns.indexOf(task.id === 'clustering' ? 'x1' : 'z1');
  const y = x + 1;
  const group = result.columns.indexOf('cluster');
  const points = result.rows.slice(0, 200);
  const xs = points.map(row => row[x]);
  const ys = points.map(row => row[y]);
  const lx = Math.min(...xs), hx = Math.max(...xs), ly = Math.min(...ys), hy = Math.max(...ys);
  const scale = (v, l, h) => h === l ? 50 : 4 + 92 * (v - l) / (h - l);
  return `<figure class="figure"><div class="scatter" role="img" aria-label="前 ${points.length} 行的二维散点图">${points.map(row => `<i class="point group-${group < 0 ? 0 : row[group]}" style="left:${scale(row[x], lx, hx)}%;bottom:${scale(row[y], ly, hy)}%"></i>`).join('')}</div><figcaption>${result.columns[x]} × ${result.columns[y]} · 前 ${points.length} 行${group < 0 ? '' : ' · 不同颜色代表不同分组'}</figcaption></figure>`;
}

function taskPage(task) {
  const current = state.get(task.id);
  const {model, busy, tab, error, info, csv, manual, result} = current;
  return `<header class="topline"><a href="#home">任务目录</a><span>${escapeHTML(task.title)}</span></header>
    <div class="page-heading">${icon(task)}<div><h1>${escapeHTML(task.title)}</h1><p>${escapeHTML(task.summary)}</p></div></div>
    <section class="panel"><div class="section-heading"><h2>准备数据</h2><a class="button" href="#data/${task.id}">生成数据</a></div>
      ${modelPicker(task, model, busy)}<p class="hint">${escapeHTML(task.input_hint)} 现有权重基于合成样本，可用于任务体验。</p>
      <div class="tabs" aria-label="数据输入方式"><button type="button" data-action="manual" aria-pressed="${tab === 'manual'}" ${busy ? 'disabled' : ''}>填写一条</button>
      <button type="button" data-action="csv" aria-pressed="${tab === 'csv'}" ${busy ? 'disabled' : ''}>批量 CSV</button></div>
      <form id="predict-form">
        ${tab === 'manual' ? `<div class="fields">${task.columns.map(column => `<label>${labels[column] || column}<input name="${column}" type="number" step="any" value="${escapeHTML(manual[column] || '')}" required ${busy ? 'disabled' : ''}></label>`).join('')}</div>` :
          `<label class="csv-label" for="csv-input">上传或粘贴 CSV</label><p class="hint">必需列 ${task.columns.join('、')}。每次最多 ${catalog.max_rows} 行。</p><textarea id="csv-input" name="csv" rows="5" placeholder="${task.columns.join(',')}" ${busy ? 'disabled' : ''}>${escapeHTML(csv)}</textarea><label class="file-label">选择 CSV 文件 <input type="file" id="csv-file" accept=".csv,text/csv" ${busy ? 'disabled' : ''}></label>`}
        <div class="actions"><button class="primary" type="submit" ${busy ? 'disabled' : ''}>${busy === 'predict' ? '正在运行…' : '运行模型'}</button><button type="button" data-action="example" ${busy ? 'disabled' : ''}>${busy === 'example' ? '正在加载…' : '加载示例'}</button><button type="button" data-action="template" ${busy ? 'disabled' : ''}>下载 CSV 模板</button></div>
      </form>${notice(error, 'alert')}${notice(info)}</section>${resultPanel(result, task, false)}`;
}

function dataPage(task) {
  const current = state.get(task.id);
  const {parameters, model, busy, generated, generationError} = current;
  return `<header class="topline"><a href="#home">任务目录</a><span>数据生成</span></header>
    <div class="page-heading"><span class="task-icon dimensionality" aria-hidden="true">▤</span><div><h1>生成合成数据</h1><p>选择任务，生成一份可预览、下载或直接运行的示例数据。</p></div></div>
    <section class="panel"><h2>生成设置</h2><label class="task-picker" for="data-task">任务类型<select id="data-task">${catalog.tasks.map(item => `<option value="${item.id}" ${task.id === item.id ? 'selected' : ''}>${escapeHTML(item.title)}</option>`).join('')}</select></label>
    ${modelPicker(task, model, busy)}<form id="generate-form"><div class="fields">${task.parameters.map(parameter => `<label>${escapeHTML(parameter.label)}<input type="number" name="${parameter.key}" min="${parameter.minimum}" max="${parameter.maximum}" step="${parameter.step}" value="${escapeHTML(parameters[parameter.key])}" required ${busy ? 'disabled' : ''}></label>`).join('')}</div>
    <p class="hint">相同设置和随机种子可以复现数据。</p><button class="primary" type="submit" ${busy ? 'disabled' : ''}>${busy === 'generate' ? '正在生成…' : '生成数据'}</button></form>${notice(generationError, 'alert')}</section>${resultPanel(generated, task, true)}`;
}

function toCSV(columns, rows) {
  return `${columns.join(',')}\n${rows.map(row => row.join(',')).join('\n')}\n`;
}

function download(filename, csv) {
  const url = URL.createObjectURL(new Blob(['\ufeff', csv], {type: 'text/csv;charset=utf-8'}));
  const link = document.createElement('a');
  link.href = url;
  link.download = filename;
  link.click();
  setTimeout(() => URL.revokeObjectURL(url), 1000);
}

async function request(task, action, body) {
  const response = await fetch(`/api/v1/tasks/${task.id}/${action}`, {
    method: 'POST', headers: {'Content-Type': 'application/json'}, body: JSON.stringify(body),
  });
  const result = await response.json();
  if (!response.ok) throw new Error(result.error);
  return result;
}

async function run(task, action) {
  const current = state.get(task.id);
  if (current.busy) return;
  const {model, csv, manual, tab, parameters} = current;
  current.busy = action;
  if (action === 'generate') {
    current.generationError = '';
    current.generated = null;
  } else {
    current.error = '';
    current.info = '';
    current.result = null;
  }
  render();
  try {
    if (action === 'predict') {
      const input = tab === 'csv' ? csv : toCSV(task.columns, [task.columns.map(column => manual[column])]);
      current.result = await request(task, 'predict', {model, csv: input});
    } else if (action === 'example') {
      const generated = await request(task, 'generate', {model});
      const n = Math.min(12, generated.rows.length);
      const rows = task.id === 'anomaly' ? [...generated.rows.slice(0, 8), ...generated.rows.slice(-4)] :
        Array.from({length: n}, (_, i) => generated.rows[Math.floor(i * generated.rows.length / n)]);
      current.csv = toCSV(generated.columns, rows);
      current.tab = 'csv';
      current.info = `已加载 ${rows.length} 条示例记录。点击「运行模型」查看结果。`;
    } else {
      current.generated = await request(task, 'generate', {model, parameters: Object.fromEntries(Object.entries(parameters).map(([key, value]) => [key, Number(value)]))});
    }
  } catch (error) {
    current[action === 'generate' ? 'generationError' : 'error'] = error.message || '请求失败，请检查本地服务是否正在运行。';
  } finally {
    current.busy = '';
    if (route().task === task) render();
  }
}

app.addEventListener('input', event => {
  const {task} = route();
  if (!task) return;
  const current = state.get(task.id);
  const {target} = event;
  if (target.closest('#generate-form')) current.parameters[target.name] = target.value;
  else if (target.id === 'csv-input') current.csv = target.value;
  else if (target.closest('#predict-form') && target.type === 'number') current.manual[target.name] = target.value;
});

app.addEventListener('change', async event => {
  const {task} = route();
  if (!task) return;
  const current = state.get(task.id);
  const {target} = event;
  if (target.id === 'data-task') location.hash = `data/${target.value}`;
  if (target.id === 'model-picker') {
    current.model = target.value;
    current.result = null;
    current.generated = null;
    current.info = '';
    current.error = '';
    render();
    document.querySelector('#model-picker')?.focus();
  }
  if (target.id === 'csv-file' && target.files.length) {
    const file = target.files[0];
    current.csv = '';
    current.error = '';
    current.info = '';
    current.result = null;
    if (file.size >= catalog.max_bytes) current.error = '文件超过 2 MiB，请拆分后再上传。';
    else {
      try {
        current.csv = await file.text();
        current.info = `已读取 ${file.name}。点击「运行模型」查看结果。`;
      } catch {
        current.error = '文件读取失败，请重试。';
      }
    }
    if (route().task === task) render();
  }
});

app.addEventListener('submit', event => {
  event.preventDefault();
  const {page, task} = route();
  run(task, page === 'data' ? 'generate' : 'predict');
});

app.addEventListener('click', event => {
  const button = event.target.closest('button[data-action]');
  if (!button) return;
  const {page, task} = route();
  const current = state.get(task.id);
  const action = button.dataset.action;
  if (action === 'example') run(task, 'example');
  if (action === 'template') download(`${task.id}-template.csv`, toCSV(task.columns, []));
  if (action === 'download') {
    const result = page === 'data' ? current.generated : current.result;
    download(`${task.id}-${page === 'data' ? 'data' : 'result'}.csv`, toCSV(result.columns, result.rows));
  }
  if (action === 'transfer') {
    current.csv = toCSV(current.generated.columns, current.generated.rows);
    current.tab = 'csv';
    current.result = null;
    current.error = '';
    current.info = `已载入 ${current.generated.rows.length} 行合成数据。`;
    location.hash = task.id;
  }
  if (action === 'sort') {
    const index = current.result.columns.indexOf('reconstruction_error');
    current.result.rows.sort((a, b) => b[index] - a[index]);
    render();
    document.querySelector('[data-action="sort"]')?.focus();
  }
  if (action === 'manual' || action === 'csv') {
    current.tab = action;
    current.error = '';
    current.info = '';
    render();
    document.querySelector(`[data-action="${action}"]`)?.focus();
  }
});

async function initialize() {
  try {
    const response = await fetch('/api/v1/tasks');
    if (!response.ok) throw new Error('任务目录请求失败。');
    catalog = await response.json();
    byId = new Map(catalog.tasks.map(task => [task.id, task]));
    dataTask = catalog.tasks[0];
    for (const task of catalog.tasks) {
      state.set(task.id, {model: task.models[0].id, tab: 'manual', csv: '', manual: {},
        parameters: Object.fromEntries(task.parameters.map(parameter => [parameter.key, parameter.default]))});
    }
    taskNav.innerHTML = catalog.tasks.map(task => `<a href="#${task.id}"><span aria-hidden="true">${icons[task.id] || '◇'}</span>${escapeHTML(task.title)}</a>`).join('');
    render();
    window.addEventListener('hashchange', () => {
      render();
      window.scrollTo(0, 0);
      app.focus();
    });
  } catch (error) {
    app.innerHTML = notice('无法读取任务目录。请确认本地工作台服务已启动，再刷新此页面。', 'alert');
  }
}

initialize();
