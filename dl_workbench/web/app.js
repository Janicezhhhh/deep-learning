const app = document.querySelector('#app');
const taskNav = document.querySelector('#task-nav');
const icons = {classification: '◩', regression: '╱', clustering: '⠿', dimensionality: '◇', anomaly: '◎', sequence: '∿', recommendation: '☆', digit: '✎', 'object-detection': '⊞', 'semantic-segmentation': '▧', 'instance-segmentation': '▥'};
const labels = {
  predicted_label: '预测类别', probability_1: '类别 1 概率', label: '生成标签',
  prediction: '预测值', target: '生成目标值', cluster: '分组', distance: '到中心的距离',
  source_group: '生成分组', z1: '降维坐标 1', z2: '降维坐标 2', reconstruction_error: '重建误差',
  is_injected_anomaly: '注入异常', next_value: '下一步预测值', actual_next: '下一步真实值',
  user_id: '用户编号', item_id: '物品编号', predicted_rating: '预测评分', rating: '生成评分',
  probability_0: '类别 0 概率', probability_2: '类别 2 概率', observed: '观测值', absolute_error: '预测偏差',
  query: '输入行号', k: '推荐结果数', rank: '排名',
  probability: '概率', instance: '实例编号', class_name: '类别', pixel_fraction: '像素占比',
  x_min: '左边界', y_min: '上边界', x_max: '右边界', y_max: '下边界',
  entity: '实体', entity_type: '实体类型', start: '起始位置', end: '结束位置',
  answer: '答案', translation: '译文', summary: '摘要', generated_text: '生成文本', transcript: '转写文本',
  positive_probability: '正向概率', negative_probability: '负向概率',
};
const scoreColumn = {'anomaly-scores': 'reconstruction_error', 'forecast-errors': 'absolute_error'};
for (const [i, key] of ['a', 'b', 'c'].entries()) {
  labels[`label_${key}`] = `标签 ${key.toUpperCase()}`;
  labels[`probability_${key}`] = `标签 ${key.toUpperCase()} 概率`;
  labels[`forecast_${i + 1}`] = `第 ${i + 1} 步预测`;
  labels[`actual_${i + 1}`] = `第 ${i + 1} 步真实值`;
}
for (let i = 1; i <= 4; i++) labels[`reconstructed_x${i}`] = `重建特征 ${i}`;
const state = new Map();
let filter = '';
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
  const visible = catalog.tasks.filter(task => !filter || task.input_type === filter);
  const grouped = Object.entries(catalog.groups).map(([group, label]) => {
    const tasks = visible.filter(task => task.group === group);
    if (!tasks.length) return '';
    return `<section class="task-group" aria-labelledby="group-${group}"><h3 id="group-${group}">${escapeHTML(label)}</h3><div class="task-grid">
      ${tasks.map(task => `<a class="task-card" href="#${task.id}">${icon(task)}<span class="method">${escapeHTML(catalog.input_types[task.input_type])}</span><h4>${escapeHTML(task.title)}</h4><p>${escapeHTML(task.summary)}</p><span class="card-action">${task.models.some(model => model.has_weights) ? '打开任务' : '查看模型安装说明'}</span></a>`).join('')}
      </div></section>`;
  }).join('');
  return `<header class="topline"><span>工作台 / 任务目录</span><span class="chip"><i class="local-dot"></i>本地运行</span></header>
    <section class="hero">
      <div><h1>选择一个任务</h1><p>按你想完成的目标查找本地模型。可以先运行示例，再使用自己的数据。</p>
      <a class="button primary" href="#data">生成示例数据</a></div>
      <div class="hero-art" aria-hidden="true"><span></span><span></span><span></span><span></span><span></span><span></span><span></span><span></span><span></span><span></span><span></span><span></span></div>
    </section>
    <div class="section-heading"><h2>任务目录</h2><p>${catalog.tasks.length} 项任务 · ${count} 个可用模型选项</p></div>
    <div class="catalog-filters" aria-label="按输入类型筛选">${[['', '全部输入'], ...Object.entries(catalog.input_types)].filter(([type]) => !type || catalog.tasks.some(task => task.input_type === type)).map(([type, label]) => `<button type="button" data-filter="${type}" aria-pressed="${filter === type}">${escapeHTML(label)}</button>`).join('')}</div>
    ${grouped}`;
}

function modelPicker(task, selected, busy) {
  if (task.models.length === 1) return `<div class="model-detail"><span>当前模型</span><strong>${escapeHTML(task.models[0].name)}</strong>${task.models[0].name !== task.method ? `<small class="method">${escapeHTML(task.method)}</small>` : ''}</div>`;
  return `<label class="model-detail" for="model-picker"><span>选择模型</span><select id="model-picker" ${busy ? 'disabled' : ''}>${task.models.map(model => `<option value="${model.id}" ${model.id === selected ? 'selected' : ''}>${escapeHTML(model.name)}</option>`).join('')}</select><small class="method">${escapeHTML(task.method)}</small></label>`;
}

function resultPanel(result, task, generation) {
  if (!result) return `<section class="panel result-panel"><h2>${generation ? '数据预览' : '运行结果'}</h2><div class="empty"><span aria-hidden="true">▤</span><p>${generation ? '生成后可预览和下载数据。' : '运行完成后，结果会显示在这里。'}</p></div></section>`;
  if (generation && result.samples) return samplePanel(result);
  const columns = result.columns;
  const rows = result.rows;
  const matchColumn = columns.indexOf('label_match');
  const errorsOnly = matchColumn >= 0 && state.get(task.id).errorsOnly;
  const visible = errorsOnly ? rows.filter(row => row[matchColumn] === 0) : rows;
  const shown = visible.slice(0, 10);
  return `<section class="panel result-panel" aria-label="${generation ? '数据预览' : '运行结果'}">
    <div class="section-heading"><div><h2>${generation ? '数据预览' : '运行结果'}</h2><p role="status">${columns.length ? `${visible.length} 行${visible.length > shown.length ? ` · 预览前 ${shown.length} 行` : ''}` : '已生成'}</p></div>
      <div class="actions">${result.preview ? `<button type="button" data-action="download-preview">下载${result.preview.type === 'audio/wav' ? '音频' : '图像'}</button>` : ''}${columns.length ? `<button data-action="download" ${rows.length ? '' : 'disabled'}>下载 CSV</button>` : ''}${generation ? `<button class="primary" data-action="transfer" ${rows.length ? '' : 'disabled'}>在任务中使用</button>` : ''}</div></div>
    ${!generation && scoreColumn[task.result_type] ? `<button class="sort" data-action="sort">按${labels[scoreColumn[task.result_type]]}从高到低排序</button>` : ''}
    ${result.evaluation ? evaluationPanel(result.evaluation) : ''}
    ${matchColumn >= 0 ? `<button type="button" class="sort" data-action="errors-only" aria-pressed="${!!errorsOnly}">${errorsOnly ? '显示全部样本' : '只看预测错误'}</button>` : ''}
    ${!generation ? scatter(task, result) : ''}
    ${result.preview ? previewMarkup(result.preview, '模型输出') : ''}
    ${columns.length ? `<div class="table-scroll" tabindex="0" role="region" aria-label="结果表格"><table><thead><tr><th scope="col">行</th>${columns.map(column => `<th scope="col" title="${escapeHTML(column)}">${escapeHTML(columnLabel(column))}</th>`).join('')}</tr></thead>
      <tbody>${shown.map((row, index) => `<tr><th scope="row">${index + 1}</th>${row.map((value, cell) => cell === matchColumn ? `<td class="${value === 0 ? 'wrong' : ''}">${value === null ? '未标注' : value ? '正确' : '错误'}</td>` : `<td${typeof value === 'string' ? ' class="text-cell"' : value === null ? '' : ` title="${value}"`}>${format(value)}</td>`).join('')}</tr>`).join('')}</tbody></table></div>` : ''}
    ${errorsOnly && !visible.length ? notice('本次已标注样本中没有预测错误。') : ''}
    ${rows.length || !columns.length ? '' : notice(generation ? '本次没有生成评分记录。可以调高评分覆盖率后重试。' : '未找到符合模型类别的结果。')}</section>`;
}

function columnLabel(column) {
  if (column === 'sample_index') return '原样本编号';
  if (column === 'label_match') return '对照结果';
  if (column === 'row_mae') return '该行平均绝对误差';
  if (column.startsWith('reference_')) {
    const key = column.slice(10);
    return `参考 ${key === 'label' ? '类别' : (labels[key] || key)}`;
  }
  return labels[column] || column;
}

function evaluationPanel(evaluation) {
  return `<div class="evaluation"><h3>参考答案对照</h3><p class="hint">已对照 ${evaluation.count} 个样本</p>
    <dl>${evaluation.metrics.map(item => `<div><dt>${escapeHTML(item.label)}</dt><dd>${item.unit === 'percent' ? `${(item.value * 100).toFixed(1)}%` : item.unit === 'match' ? item.value ? '正确' : '错误' : format(item.value)}</dd></div>`).join('')}</dl></div>`;
}

function format(value) {
  if (value === null) return '—';
  if (typeof value === 'string') return escapeHTML(value);
  if (value !== 0 && Math.abs(value) < 0.0001) return value.toExponential(2);
  return value.toLocaleString('zh-CN', {maximumFractionDigits: 5, useGrouping: false});
}

function previewMarkup(media, label) {
  const src = `data:${escapeHTML(media.type)};base64,${escapeHTML(media.data)}`;
  return media.type === 'audio/wav' ? `<audio controls preload="none" src="${src}" aria-label="${label}"></audio>` :
    `<img class="media-preview" alt="${label}" src="${src}">`;
}

function samplePanel(dataset) {
  return `<section class="panel result-panel" aria-label="数据预览"><div class="section-heading"><div><h2>数据预览</h2><p role="status">${dataset.samples.length} 条样本</p></div>
    <button data-action="download-dataset">下载数据集 ZIP</button></div>
    <div class="sample-grid">${dataset.samples.map((sample, index) => `<article class="sample">
      ${sample.data ? previewMarkup(sample, `样本 ${index + 1}`) : `<p class="sample-text">${escapeHTML(Object.values(sample.fields).join('\n'))}</p>`}
      <p>${escapeHTML(sample.target.label === undefined ? sample.target.objects ? `${sample.target.objects.length} 个物体` : sample.name : `标签 ${sample.target.label}`)}</p>
      ${sample.target.reference || sample.target.transcript ? `<p class="hint">参考内容　${escapeHTML(sample.target.reference || sample.target.transcript)}</p>` : ''}
      <button data-action="sample" data-index="${index}">在任务中使用</button>
    </article>`).join('')}</div></section>`;
}

function scatter(task, result) {
  if (!['cluster-scatter', 'embedding-scatter'].includes(task.result_type)) return '';
  const x = result.columns.indexOf(task.result_type === 'cluster-scatter' ? 'x1' : 'z1');
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
  if (['image', 'text', 'audio'].includes(task.input_type)) return mediaTaskPage(task);
  const current = state.get(task.id);
  const {model, busy, tab, error, info, csv, manual, result} = current;
  return `<header class="topline"><a href="#home">任务目录</a><span>${escapeHTML(task.title)}</span></header>
    <div class="page-heading">${icon(task)}<div><h1>${escapeHTML(task.title)}</h1><p>${escapeHTML(task.summary)}</p></div></div>
    <section class="panel"><div class="section-heading"><h2>准备数据</h2><a class="button" href="#data/${task.id}">生成数据</a></div>
      ${modelPicker(task, model, busy)}<p class="hint">${escapeHTML(task.input_hint)} 现有权重基于合成样本，可用于任务体验。</p>
      <div class="tabs" aria-label="数据输入方式"><button type="button" data-action="manual" aria-pressed="${tab === 'manual'}" ${busy ? 'disabled' : ''}>填写一条</button>
      <button type="button" data-action="csv" aria-pressed="${tab === 'csv'}" ${busy ? 'disabled' : ''}>批量 CSV</button></div>
      <form id="predict-form">
        ${tab === 'manual' ? `<div class="fields">${task.columns.map(column => {
          const range = task.integer_ranges[column];
          return `<label>${labels[column] || column}<input name="${column}" type="number" ${range ? `min="${range[0]}" max="${range[1]}" step="1"` : 'step="any"'} value="${escapeHTML(manual[column] || '')}" required ${busy ? 'disabled' : ''}></label>`;
        }).join('')}</div>` :
          `<label class="csv-label" for="csv-input">上传或粘贴 CSV</label><p class="hint">必需列 ${task.columns.join('、')}。每次最多 ${catalog.max_rows} 行。${task.reference_columns.length ? `带 ${task.reference_columns.join('、')} 列时，会对照参考答案。` : ''}</p><textarea id="csv-input" name="csv" rows="5" placeholder="${task.columns.join(',')}" ${busy ? 'disabled' : ''}>${escapeHTML(csv)}</textarea><label class="file-label">选择 CSV 文件 <input type="file" id="csv-file" accept=".csv,text/csv" ${busy ? 'disabled' : ''}></label>`}
        <div class="actions"><button class="primary" type="submit" ${busy ? 'disabled' : ''}>${busy === 'predict' ? '正在运行…' : '运行模型'}</button><button type="button" data-action="example" ${busy ? 'disabled' : ''}>${busy === 'example' ? '正在加载…' : '加载示例'}</button><button type="button" data-action="template" ${busy ? 'disabled' : ''}>下载 CSV 模板</button></div>
      </form>${notice(error, 'alert')}${notice(info)}</section>${resultPanel(result, task, false)}`;
}

function mediaTaskPage(task) {
  const {model, busy, error, info, media, fields, result, tab, csv, predictParameters} = state.get(task.id);
  const selected = task.models.find(item => item.id === model);
  const isText = task.input_type === 'text';
  const isImage = task.input_type === 'image';
  const title = isImage ? '选择图片' : '选择 WAV 音频';
  const isBatch = task.batch_limit > 0 && tab === 'csv';
  return `<header class="topline"><a href="#home">任务目录</a><span>${escapeHTML(task.title)}</span></header>
    <div class="page-heading">${icon(task)}<div><h1>${escapeHTML(task.title)}</h1><p>${escapeHTML(task.summary)}</p></div></div>
    <section class="panel"><div class="section-heading"><h2>准备数据</h2><a class="button" href="#data/${task.id}">生成数据</a></div>
      ${modelPicker(task, model, busy)}<p class="hint">${escapeHTML(task.input_hint)}</p>
      ${selected.has_weights ? '' : notice(`本机还未准备这个模型的权重。${selected.asset_key ? `按 README 安装可选依赖，运行 python -m dl_workbench.download_models ${selected.asset_key} 后刷新。` : '请确认仓库中的权重文件完整。'}`)}
      ${task.batch_limit ? `<div class="tabs" aria-label="数据输入方式"><button type="button" data-action="manual" aria-pressed="${!isBatch}" ${busy ? 'disabled' : ''}>填写一条</button><button type="button" data-action="csv" aria-pressed="${isBatch}" ${busy ? 'disabled' : ''}>批量 CSV</button></div>` : ''}
      <form id="predict-form" class="media-form">${isBatch ? `<label class="csv-label" for="csv-input">上传或粘贴 CSV</label><p class="hint">必需列 text。每次最多 ${task.batch_limit} 行。</p><textarea id="csv-input" name="csv" rows="5" placeholder="text" ${busy ? 'disabled' : ''} required>${escapeHTML(csv)}</textarea><label class="file-label">选择 CSV 文件 <input type="file" id="csv-file" accept=".csv,text/csv" ${busy ? 'disabled' : ''}></label>` : isText ? task.fields.map(([name, label]) => `<label class="csv-label" for="text-${name}">${escapeHTML(label)}</label><textarea id="text-${name}" name="${name}" rows="4" maxlength="${name === 'question' ? 300 : task.id === 'speech' ? 500 : 4000}" ${busy ? 'disabled' : ''} required>${escapeHTML(fields[name] || '')}</textarea>`).join('') :
        `<label class="file-label">${title} <input type="file" id="media-file" accept="${isImage ? '.png,.jpg,.jpeg,image/png,image/jpeg' : '.wav,audio/wav'}" ${busy ? 'disabled' : ''}></label>${media ? `<div class="media-input">${previewMarkup(media, '当前输入')}<p class="hint">${escapeHTML(media.name)}</p></div>` : ''}`}
        ${task.predict_parameters.length ? `<div class="fields">${parameterInputs(task.predict_parameters, predictParameters, busy)}</div>` : ''}
        <div class="actions"><button type="submit" class="primary" ${busy || !selected.has_weights || (!isText && !media) ? 'disabled' : ''}>${busy === 'predict' ? '正在运行…' : '运行模型'}</button><button type="button" data-action="example" ${busy ? 'disabled' : ''}>${busy === 'example' ? '正在加载…' : '加载示例'}</button>${isBatch ? `<button type="button" data-action="template" ${busy ? 'disabled' : ''}>下载 CSV 模板</button>` : ''}</div>
      </form>${notice(error, 'alert')}${notice(info)}</section>${resultPanel(result, task, false)}`;
}

function parameterInputs(definitions, values, busy) {
  return definitions.map(parameter => `<label>${escapeHTML(parameter.label)}<input type="number" name="${parameter.key}" min="${parameter.minimum}" max="${parameter.maximum}" step="${parameter.step}" value="${escapeHTML(values[parameter.key])}" required ${busy ? 'disabled' : ''}>${parameter.description ? `<small class="hint">${escapeHTML(parameter.description)}</small>` : ''}</label>`).join('');
}

function numericSettings(values) {
  return Object.fromEntries(Object.entries(values).map(([key, value]) => [key, Number(value)]));
}

function dataPage(task) {
  const current = state.get(task.id);
  const {parameters, model, busy, generated, generationError} = current;
  const options = Object.entries(catalog.groups).map(([group, label]) => {
    const groupTasks = catalog.tasks.filter(item => item.group === group);
    return groupTasks.length ? `<optgroup label="${escapeHTML(label)}">${groupTasks.map(item => `<option value="${item.id}" ${task.id === item.id ? 'selected' : ''}>${escapeHTML(item.title)}</option>`).join('')}</optgroup>` : '';
  }).join('');
  return `<header class="topline"><a href="#home">任务目录</a><span>数据生成</span></header>
    <div class="page-heading"><span class="task-icon dimensionality" aria-hidden="true">▤</span><div><h1>生成合成数据</h1><p>选择任务，生成一份可预览、下载或直接运行的示例数据。</p></div></div>
    <section class="panel"><h2>生成设置</h2><label class="task-picker" for="data-task">任务类型<select id="data-task">${options}</select></label>
    ${modelPicker(task, model, busy)}<form id="generate-form"><div class="fields">${parameterInputs(task.parameters, parameters, busy)}</div>
    <p class="hint">相同设置和随机种子可以复现数据。</p><button class="primary" type="submit" ${busy ? 'disabled' : ''}>${busy === 'generate' ? '正在生成…' : '生成数据'}</button></form>${notice(generationError, 'alert')}</section>${resultPanel(generated, task, true)}`;
}

function toCSV(columns, rows) {
  const cell = value => value === null ? '' : typeof value !== 'string' ? String(value) : `"${value.replaceAll('"', '""')}"`;
  return `${columns.join(',')}\n${rows.map(row => row.map(cell).join(',')).join('\n')}\n`;
}

function download(filename, csv) {
  downloadFile(filename, new Blob(['\ufeff', csv], {type: 'text/csv;charset=utf-8'}));
}

function downloadFile(filename, blob) {
  const url = URL.createObjectURL(blob);
  const link = document.createElement('a');
  link.href = url;
  link.download = filename;
  link.click();
  setTimeout(() => URL.revokeObjectURL(url), 1000);
}

function downloadBase64(filename, type, data) {
  const bytes = Uint8Array.from(atob(data), char => char.charCodeAt(0));
  downloadFile(filename, new Blob([bytes], {type}));
}

function selectSample(current, sample) {
  current.media = sample.data ? sample : null;
  current.fields = {...sample.fields};
  if (sample.settings) Object.assign(current.predictParameters, sample.settings);
  current.tab = 'manual';
  current.result = null;
  current.error = '';
  current.info = '已加载示例。点击「运行模型」查看结果。';
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
    current.errorsOnly = false;
  }
  render();
  try {
    if (action === 'predict') {
      let input;
      if (task.input_type === 'text') input = task.batch_limit && tab === 'csv' ? {csv} : {...current.fields, ...numericSettings(current.predictParameters)};
      else if (['image', 'audio'].includes(task.input_type)) {
        input = {[task.input_type]: current.media?.data || ''};
        if (task.input_type === 'image' && current.media?.target) {
          input.reference = current.media.target;
          input.reference_mask = current.media.reference_mask;
        }
      }
      else input = {csv: tab === 'csv' ? csv : toCSV(task.columns, [task.columns.map(column => manual[column])])};
      current.result = await request(task, 'predict', {model, ...input});
    } else if (action === 'example') {
      const generated = await request(task, 'generate', {model, parameters: task.dataset_type === 'samples' ? {samples: 1} : {}});
      if (generated.samples && task.batch_limit && tab === 'csv') {
        current.csv = toCSV(['text'], generated.samples.map(sample => [sample.fields.text]));
        current.info = '已加载示例。点击「运行模型」查看结果。';
      } else if (generated.samples) selectSample(current, generated.samples[0]);
      else {
        const n = Math.min(12, generated.rows.length);
        const rows = task.id === 'anomaly' ? [...generated.rows.slice(0, 8), ...generated.rows.slice(-4)] :
          Array.from({length: n}, (_, i) => generated.rows[Math.floor(i * generated.rows.length / n)]);
        current.csv = toCSV(generated.columns, rows);
        current.tab = 'csv';
        current.info = `已加载 ${rows.length} 条示例记录。点击「运行模型」查看结果。`;
      }
    } else {
      current.generated = await request(task, 'generate', {model, parameters: numericSettings(parameters)});
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
  else if (target.id?.startsWith('text-')) current.fields[target.name] = target.value;
  else if (Object.hasOwn(current.predictParameters, target.name)) current.predictParameters[target.name] = target.value;
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
  if (target.id === 'media-file' && target.files.length) {
    const file = target.files[0];
    current.media = null;
    current.error = '';
    current.info = '';
    current.result = null;
    current.busy = 'upload';
    render();
    try {
      if (file.size > Math.floor(catalog.max_bytes * .70)) throw new Error('文件太大，请选择不超过 1.4 MiB 的文件。');
      const data = await new Promise((resolve, reject) => {
        const reader = new FileReader();
        reader.onload = () => resolve(reader.result.split(',')[1]);
        reader.onerror = () => reject(new Error('无法读取文件。'));
        reader.readAsDataURL(file);
      });
      const type = task.input_type === 'audio' ? 'audio/wav' : (/\.jpe?g$/i.test(file.name) ? 'image/jpeg' : 'image/png');
      current.media = {data, type, name: file.name};
    } catch (error) {
      current.error = error.message;
    } finally {
      current.busy = '';
      if (route().task === task) render();
    }
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
  const filterButton = event.target.closest('button[data-filter]');
  if (filterButton) {
    filter = filterButton.dataset.filter;
    render();
    document.querySelector(`button[data-filter="${filter}"]`)?.focus();
    return;
  }
  const button = event.target.closest('button[data-action]');
  if (!button) return;
  const {page, task} = route();
  const current = state.get(task.id);
  const action = button.dataset.action;
  if (action === 'example') run(task, 'example');
  if (action === 'template') download(`${task.id}-template.csv`, toCSV(task.batch_limit ? ['text'] : task.columns, []));
  if (action === 'download') {
    const result = page === 'data' ? current.generated : current.result;
    const safeRows = result.rows.map(row => row.map(value => typeof value === 'string' && /^\s*[=+\-@]/.test(value) ? `'${value}` : value));
    download(`${task.id}-${page === 'data' ? 'data' : 'result'}.csv`, toCSV(result.columns, safeRows));
  }
  if (action === 'download-preview') {
    const {type, data} = current.result.preview;
    downloadBase64(`${task.id}-result.${type === 'audio/wav' ? 'wav' : 'png'}`, type, data);
  }
  if (action === 'download-dataset') downloadBase64(`${task.id}-dataset.zip`, 'application/zip', current.generated.archive);
  if (action === 'sample') {
    selectSample(current, current.generated.samples[Number(button.dataset.index)]);
    location.hash = task.id;
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
    const index = current.result.columns.indexOf(scoreColumn[task.result_type]);
    current.result.rows.sort((a, b) => b[index] - a[index]);
    render();
    document.querySelector('[data-action="sort"]')?.focus();
  }
  if (action === 'errors-only') {
    current.errorsOnly = !current.errorsOnly;
    render();
    document.querySelector('[data-action="errors-only"]')?.focus();
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
      state.set(task.id, {model: task.models[0].id, tab: 'manual', csv: '', manual: {}, fields: {},
        parameters: Object.fromEntries(task.parameters.map(parameter => [parameter.key, parameter.default])),
        predictParameters: Object.fromEntries(task.predict_parameters.map(parameter => [parameter.key, parameter.default]))});
    }
    taskNav.innerHTML = Object.entries(catalog.groups).map(([group, label]) => {
      const tasks = catalog.tasks.filter(task => task.group === group);
      return tasks.length ? `<p class="nav-group-label">${escapeHTML(label)}</p>${tasks.map(task => `<a href="#${task.id}"><span aria-hidden="true">${icons[task.id] || '◇'}</span>${escapeHTML(task.title)}</a>`).join('')}` : '';
    }).join('');
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
