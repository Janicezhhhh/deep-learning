# 模型工坊

在本机使用仓库中的模型和合成数据生成器。数值任务覆盖二分类、多分类、多标签分类、回归、聚类、降维、异常检测、单步及多步序列预测、多变量序列预测、时间序列异常检测、推荐评分和 Top-K 推荐。

图像任务支持手写数字识别、目标检测、语义分割和实例分割。可上传 PNG 或 JPEG，也可以从「数据生成」预览样本并下载包含原图和标注的 ZIP。

文本任务支持英文情感分类、命名实体识别、英译中、摘要、阅读理解问答和短文本生成。音频任务支持英文指令词分类、英文语音识别和英文语音合成。

## 在 Windows 上启动

在仓库根目录运行以下命令。工作台使用 Python 3.10 或更高版本，已在 Python 3.12 下验证。

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r dl_workbench/requirements.txt
.\.venv\Scripts\python.exe -m dl_workbench
```

打开 <http://127.0.0.1:8765>。端口被占用时，可在启动命令末尾加 `--port 8766`。模型默认用 CPU 推理，服务只监听本机地址。

### 准备文本和语音模型

这些模型需要首次联网下载。已下载的模型在本机推理，用户输入在本机处理。可按需指定一个或多个模型。省略模型名会下载全部，建议预留 3 GB 磁盘空间。

```powershell
.\.venv\Scripts\python.exe -m pip install -r dl_workbench/requirements-media.txt
.\.venv\Scripts\python.exe -m dl_workbench.download_models
# 仅安装语音识别及生成语音样本需要的模型
.\.venv\Scripts\python.exe -m dl_workbench.download_models transcription speech
```

下载后刷新任务目录。模型保存在 `.workbench_models`，可在模型卡查看适用范围和许可。

| 下载名称 | 任务 | 模型卡 |
| --- | --- | --- |
| `sentiment` | 英文评论情感分类 | [DistilBERT SST-2](https://huggingface.co/distilbert/distilbert-base-uncased-finetuned-sst-2-english) |
| `entities` | 英文实体识别 | [DistilBERT NER](https://huggingface.co/dslim/distilbert-NER) |
| `translation` | 英译中 | [MarianMT](https://huggingface.co/Helsinki-NLP/opus-mt-en-zh) |
| `summary` | 英文摘要 | [FLAN-T5 Small](https://huggingface.co/google/flan-t5-small) |
| `question` | 英文段落问答 | [DistilBERT SQuAD](https://huggingface.co/distilbert/distilbert-base-cased-distilled-squad) |
| `generation` | 英文短文本生成 | [SmolLM2](https://huggingface.co/HuggingFaceTB/SmolLM2-135M-Instruct) |
| `keywords` | 英文指令词分类 | [Wav2Vec2 SUPERB](https://huggingface.co/superb/wav2vec2-base-superb-ks) |
| `transcription` | 英文语音识别 | [Whisper Tiny English](https://huggingface.co/openai/whisper-tiny.en) |
| `speech` | 英文语音合成及生成音频样本 | [MMS-TTS English](https://huggingface.co/facebook/mms-tts-eng) |

## 使用方式

1. 从任务目录选一个任务。可以手填一条、加载示例，或下载 CSV 模板后上传或粘贴数据。
2. 点击「运行模型」，查看结果表格。聚类和降维附有散点图，异常检测结果可按重建误差排序。下载 CSV 可获得完整结果。
3. 在「数据生成」中选择任务、模型和生成参数。生成后可以下载 CSV，也可以点「在任务中使用」把数据载入任务页面。

CSV 使用 UTF-8，支持带 BOM。输入列名以模板为准，列顺序不限。其他列可用于保存生成标签，推理仅取模型所需列，结果 CSV 包含模型所需输入列和输出列。每次最多处理 2,000 行，单个 API 请求最多 2 MiB。

文本分类另外支持包含 `text` 列的批量 CSV，每批最多 64 条，结果以输入记录序号对应原文。图片和音频上传单个文件，页面上限约为 1.4 MiB。音频接受未压缩的 PCM WAV，最长 30 秒，指令词分类最长 3 秒。生成语音样本会附带参考文本，生成的文本样本会根据任务附带标签、参考答案或参考译文。

数值任务的权重基于合成数据，适合演示和验证流程。推荐评分接受已有的用户编号 `0–49` 和物品编号 `0–29`。Top-K 根据同一评分模型对 30 个候选编号排序，接受 `k=1–10`，每条请求对应一至十条结果。同分时按物品编号排序。异常检测给出重建误差，时间序列异常检测给出实际观测与预测之间的绝对误差。

可用 `python -m classic_tabular.train` 复现多分类、多标签分类、多步和多变量序列模型。其测试数据使用独立随机种子生成。

手写数字模型结合 MNIST 和生成的数字笔画训练。在 [MNIST 数据集](https://github.com/cvdfoundation/mnist)下载四个 `.gz` 文件后，可运行 `python -m classic_vision.train --mnist <目录> --shapes` 复现图像权重。检测和分割使用生成的几何图形，可识别深色背景中分开的圆形、方形和三角形。检测框按输入图片像素输出，右、下边界采用不含端点的坐标。实例由 CNN 掩膜的连通域区分，适用场景是同类物体彼此分开。

## 页面与接口

| 页面 | 路由 | 用途 |
| --- | --- | --- |
| 任务目录 | `/#home` | 按任务目标浏览，按输入类型筛选 |
| 任务工作区 | `/#<task-id>` | 选择模型、准备输入、运行、查看并导出结果 |
| 数据生成 | `/#data/<task-id>` | 调整生成参数、预览、导出或送入任务 |

| HTTP 接口 | 请求数据 | 返回数据 |
| --- | --- | --- |
| `GET /api/v1/tasks` | 无 | 功能分组、输入类型、结果类型、列设置、模型选项和生成参数 |
| `POST /api/v1/tasks/{task-id}/generate` | `{"model":"...","parameters":{"seed":42}}` | 表格返回 `columns`/`rows`，图像、文本或音频返回 `samples` 和 Base64 ZIP `archive` |
| `POST /api/v1/tasks/{task-id}/predict` | 表格传 `csv`，图像或音频传 Base64 `image`/`audio`，文本使用目录的 `fields`，`model` 可选 | 返回 `columns`/`rows`，检测、分割或语音合成另附 `preview` |

调用时省略 `model` 则使用该任务的第一个模型，省略生成参数则采用任务默认值。参数错误返回 HTTP 400 和 `{"error":"..."}`，模型文件或依赖缺失返回 HTTP 503。

## 架构与扩展

- `dl_workbench/tasks.py` 是任务目录的唯一来源。任务声明输入和输出列、模型清单、生成参数、推理适配函数和生成函数。已有模型类、生成逻辑及权重从各子项目复用。权重按需加载并在当前进程复用。
- `dl_workbench/__main__.py` 提供本地 HTTP 服务、静态资源与版本化 API。`dl_workbench/web/` 使用任务目录返回的字段构建导航、模型选择、表单和结果表。
- 新增已有任务的模型时，把权重及模型定义加入相应子项目，并在任务目录登记模型版本。共享输入输出的模型沿用该任务的页面。新增表格任务时，登记新的任务、输入输出列和适配函数，通用页面自动出现。
- `input_type` 选择数据输入组件，`result_type` 决定如何展示结果，`dataset_type` 控制生成数据的预览与下载方式。图像适配器和文本音频适配器分别位于 `dl_workbench/vision_tasks.py`、`dl_workbench/language_tasks.py`。可选模型使用 `model_assets.py` 的清单准备，适配器以本地文件模式加载。
- 扩展到用户数据训练时，可增加独立训练任务、训练状态和模型版本登记，训练完成后再把新权重登记为可用模型。
