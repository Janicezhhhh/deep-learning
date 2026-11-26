import csv
import io
import json
import wave
from functools import lru_cache
from random import Random
from threading import Lock

from .datasets import MediaDataset
from .media import InputError, binary_input, encode, table_csv
from .model_assets import available, model_directory


RUNTIME_LOCK = Lock()
SPEECH_SPEED_RANGE = (50, 180)
SPEECH_NOISE_RANGE = (0, 100)
MODELS = {
    "sentiment": ("AutoModelForSequenceClassification", "AutoTokenizer"),
    "entities": ("AutoModelForTokenClassification", "AutoTokenizer"),
    "translation": ("AutoModelForSeq2SeqLM", "AutoTokenizer"),
    "summary": ("AutoModelForSeq2SeqLM", "AutoTokenizer"),
    "question": ("AutoModelForQuestionAnswering", "AutoTokenizer"),
    "generation": ("AutoModelForCausalLM", "AutoTokenizer"),
    "keywords": ("AutoModelForAudioClassification", "AutoFeatureExtractor"),
    "transcription": ("AutoModelForSpeechSeq2Seq", "AutoProcessor"),
    "speech": ("VitsModel", "AutoTokenizer"),
}


class ModelUnavailable(FileNotFoundError):
    pass


@lru_cache(maxsize=3)
def runtime(key):
    if not available(key):
        raise ModelUnavailable(f"模型 {key} 的本地文件尚未准备。请按 README 下载该模型后重试。")
    import transformers

    model_class, processor_class = MODELS[key]
    options = {"local_files_only": True, "trust_remote_code": False}
    path = model_directory(key)
    processor = getattr(transformers, processor_class).from_pretrained(path, **options)
    model = getattr(transformers, model_class).from_pretrained(path, **options).eval()
    if key == "entities":
        return transformers.pipeline("token-classification", model=model, tokenizer=processor, device=-1), None
    return model, processor


def table(columns, rows):
    return {"columns": columns, "rows": rows}


def text_field(payload, name, label, maximum=4000):
    value = payload.get(name)
    if not isinstance(value, str) or not value.strip():
        raise InputError(f"请填写{label}。")
    value = value.strip()
    if len(value) > maximum:
        raise InputError(f"{label}最多 {maximum} 个字符。")
    return value


def sentiment_csv(value):
    if not isinstance(value, str) or not value.strip():
        raise InputError("请上传或粘贴包含 text 列的 CSV。")
    reader = csv.reader(io.StringIO(value.removeprefix("\ufeff")), strict=True)
    try:
        header = [value.strip() for value in next(reader, [])]
        if header.count("text") != 1 or len(header) != len(set(header)):
            raise InputError("CSV 需要一个名为 text 的列，列名不可重复。")
        index = header.index("text")
        rows = []
        for row in reader:
            if not row:
                continue
            if len(row) != len(header):
                raise InputError(f"CSV 第 {reader.line_num} 行的列数与表头不同。")
            rows.append(text_field({"text": row[index]}, "text", f"CSV 第 {reader.line_num} 行的文本"))
            if len(rows) > 64:
                raise InputError("文本分类每批最多 64 行。")
    except csv.Error as error:
        raise InputError(f"CSV 第 {reader.line_num} 行格式有误。") from error
    if not rows:
        raise InputError("CSV 至少需要一条文本。")
    return rows


def sentiments(model, tokenizer, texts):
    for index in range(0, len(texts), 8):
        inputs = tokenizer(texts[index:index + 8], return_tensors="pt", padding=True, truncation=True, max_length=512)
        for scores in model(**inputs).logits.softmax(dim=-1).tolist():
            yield sorted(((model.config.id2label[i], score) for i, score in enumerate(scores)),
                         key=lambda row: row[1], reverse=True)


def answer_span(model, tokenizer, question, context):
    import torch

    inputs = tokenizer(question, context, truncation="only_second", max_length=384, stride=96, padding=True,
                       return_overflowing_tokens=True, return_offsets_mapping=True, return_tensors="pt")
    offsets = inputs.pop("offset_mapping").tolist()
    inputs.pop("overflow_to_sample_mapping")
    output = model(**inputs)
    best = (-1., 0, 0)
    for index, spans in enumerate(offsets):
        context_tokens = torch.tensor([sequence == 1 for sequence in inputs.sequence_ids(index)])
        if not context_tokens.any():
            continue
        start_probability = output.start_logits[index].masked_fill(~context_tokens, -torch.inf).softmax(dim=0)
        end_probability = output.end_logits[index].masked_fill(~context_tokens, -torch.inf).softmax(dim=0)
        joint = torch.outer(start_probability, end_probability).triu().tril(diagonal=31)
        width = len(spans)
        start, end = divmod(int(joint.argmax()), width)
        score = float(joint[start, end])
        if score > best[0]:
            best = (score, spans[start][0], spans[end][1])
    score, start, end = best
    return [context[start:end], start, end, max(0., score)]


def text_predict(key, payload):
    import torch

    if key == "sentiment" and "csv" in payload:
        texts = sentiment_csv(payload["csv"])
        with RUNTIME_LOCK, torch.inference_mode():
            model, tokenizer = runtime(key)
            rows = [[index, *scores[0], dict(scores)["POSITIVE"], dict(scores)["NEGATIVE"]]
                    for index, scores in enumerate(sentiments(model, tokenizer, texts), 1)]
        return table(("query", "class_name", "probability", "positive_probability", "negative_probability"), rows)
    if key == "question":
        question = text_field(payload, "question", "问题", 300)
        text = text_field(payload, "context", "参考段落")
    else:
        text = text_field(payload, "text", "文本", 500 if key == "speech" else 4000)
    with RUNTIME_LOCK, torch.inference_mode():
        model, processor = runtime(key)
        if key == "sentiment":
            return table(("class_name", "probability"), next(sentiments(model, processor, [text])))
        if key == "entities":
            entities = model(text, aggregation_strategy="simple")
            return table(("entity", "entity_type", "start", "end", "probability"),
                         [[text[item["start"]:item["end"]], item["entity_group"], int(item["start"]),
                           int(item["end"]), float(item["score"])] for item in entities])
        if key == "question":
            return table(("answer", "start", "end", "probability"), [answer_span(model, processor, question, text)])
        if key == "speech":
            speed = audio_setting(payload, "speed_percent", 100, SPEECH_SPEED_RANGE, "语速")
            noise = audio_setting(payload, "noise_percent", 0, SPEECH_NOISE_RANGE, "背景噪声")
            data = speech_bytes(text, model, processor, speed_percent=speed, noise_percent=noise)
            return {**table((), []), "preview": {"type": "audio/wav", "data": encode(data)}}
        if key == "summary":
            text = "Summarize the following text in one or two sentences: " + text
        if key == "generation" and processor.chat_template:
            inputs = processor.apply_chat_template([{"role": "user", "content": text}], tokenize=True,
                                                   add_generation_prompt=True, return_dict=True, return_tensors="pt",
                                                   truncation=True, max_length=1024)
        else:
            inputs = processor(text, return_tensors="pt", truncation=True, max_length=512)
        tokens = model.generate(**inputs, do_sample=False, max_new_tokens=120 if key == "generation" else 96)
        if not model.config.is_encoder_decoder:
            tokens = tokens[:, inputs["input_ids"].shape[-1]:]
        content = processor.batch_decode(tokens, skip_special_tokens=True)[0].strip()
        column = {"summary": "summary", "translation": "translation", "generation": "generated_text"}[key]
        return table((column,), [[content]])


def decode_wav(value):
    import numpy as np

    contents = binary_input(value, "WAV 音频")
    try:
        with wave.open(io.BytesIO(contents), "rb") as audio:
            rate = audio.getframerate()
            count = audio.getnframes()
            channels = audio.getnchannels()
            width = audio.getsampwidth()
            if count < 1 or not 8000 <= rate <= 96000 or count > 30 * rate or channels not in (1, 2) or width not in (1, 2, 4):
                raise InputError("请选择最长 30 秒、8–96 kHz、单声道或立体声的 8/16/32 位 PCM WAV。")
            raw = audio.readframes(count)
            if len(raw) != count * channels * width:
                raise InputError("WAV 音频不完整。")
    except (wave.Error, EOFError) as error:
        raise InputError("无法读取 WAV。请导入未压缩的 PCM WAV 音频。") from error
    types = {1: "u1", 2: "<i2", 4: "<i4"}
    samples = np.frombuffer(raw, dtype=types[width]).astype(np.float32).reshape(-1, channels).mean(axis=1)
    if width == 1:
        samples -= 128
    samples /= 2 ** (8 * width - 1)
    if rate != 16000:
        samples = np.interp(np.arange(max(1, round(len(samples) * 16000 / rate))) * (rate / 16000),
                            np.arange(len(samples)), samples).astype(np.float32)
    return samples


def audio_predict(key, payload):
    import numpy as np
    import torch

    samples = decode_wav(payload.get("audio"))
    with RUNTIME_LOCK, torch.inference_mode():
        model, processor = runtime(key)
        if key == "transcription":
            inputs = processor(samples, sampling_rate=16000, return_tensors="pt", return_attention_mask=True)
            tokens = model.generate(**inputs, max_new_tokens=200)
            text = processor.batch_decode(tokens, skip_special_tokens=True)[0].strip()
            return table(("transcript",), [[text]])
        if len(samples) > 48000:
            raise InputError("关键词分类适用于最多 3 秒的英文单词音频。")
        if len(samples) < 16000:
            samples = np.pad(samples, (0, 16000 - len(samples)))
        inputs = processor(samples, sampling_rate=16000, return_tensors="pt")
        probabilities = model(**inputs).logits.softmax(dim=-1)[0].tolist()
        rows = sorted(((model.config.id2label[i], score) for i, score in enumerate(probabilities)),
                      key=lambda row: row[1], reverse=True)
        return table(("class_name", "probability"), rows)


def audio_setting(payload, key, default, bounds, label):
    value = payload.get(key, default)
    if type(value) is not int or not bounds[0] <= value <= bounds[1]:
        raise InputError(f"{label}必须是 {bounds[0]}–{bounds[1]} 之间的整数。")
    return value


def speech_bytes(text, model, processor, *, speed_percent=100, noise_percent=0, noise_seed=0):
    import numpy as np

    inputs = processor(text, return_tensors="pt")
    # The caller holds RUNTIME_LOCK while using the shared synthesis model.
    original_rate = model.speaking_rate
    try:
        model.speaking_rate = original_rate * speed_percent / 100
        waveform = model(**inputs).waveform[0].float().numpy()
    finally:
        model.speaking_rate = original_rate
    if noise_percent and waveform.size:
        rng = np.random.default_rng(noise_seed)
        noise = rng.standard_normal(waveform.size, dtype=np.float32)
        rms = np.sqrt(np.mean(waveform ** 2))
        waveform = waveform + noise * (noise_percent / 100) * rms / np.sqrt(np.mean(noise ** 2))
        waveform /= max(1, np.abs(waveform).max())
    pcm = (np.clip(waveform, -1, 1) * 32767).astype("<i2")
    output = io.BytesIO()
    with wave.open(output, "wb") as audio:
        audio.setnchannels(1)
        audio.setsampwidth(2)
        audio.setframerate(model.config.sampling_rate)
        audio.writeframes(pcm.tobytes())
    return output.getvalue()


PEOPLE = ("Amelia Chen", "Morgan Lee", "Riley Patel", "Jamie Wilson")
CITIES = ("London", "Toronto", "Seattle", "Sydney")
SUBJECTS = ("photography", "gardening", "painting", "music")
KEYWORDS = ("yes", "no", "up", "down", "left", "right", "on", "off", "stop", "go")


def sample_text(key, rng):
    person, city, subject = rng.choice(PEOPLE), rng.choice(CITIES), rng.choice(SUBJECTS)
    if key == "sentiment":
        positive = rng.randrange(2) == 0
        opinion = rng.choice(("I loved every minute of", "I was delighted by")) if positive else rng.choice(("I was bored by", "I strongly disliked"))
        return {"text": f"{opinion} the {subject} class."}, {"label": "POSITIVE" if positive else "NEGATIVE"}
    if key == "entities":
        return {"text": f"{person} joined the United Nations and moved to {city}."}, {
            "entities": [[person, "PER"], ["United Nations", "ORG"], [city, "LOC"]]}
    if key == "translation":
        noun, chinese = rng.choice((("museum", "博物馆"), ("library", "图书馆"), ("airport", "机场")))
        return {"text": f"Where is the nearest {noun}?"}, {"reference": f"最近的{chinese}在哪里？"}
    if key == "summary":
        day = rng.choice(("Monday", "Tuesday", "Wednesday", "Thursday"))
        article = f"{person} announced a free {subject} class in {city}. The first session begins on {day}. The class is open to beginners, and all materials will be provided."
        return {"text": article}, {"reference": f"{person} will offer a free {subject} class in {city}, starting on {day}."}
    if key == "question":
        return {"context": f"{person} moved to {city} last year. In her spare time, she enjoys {subject}.",
                "question": f"Where did {person} move?"}, {"reference": city}
    if key == "generation":
        return {"text": f"Write three sentences about a {subject} club meeting for the first time in {city}."}, {}
    if key == "keywords":
        word = rng.choice(KEYWORDS)
        return {"text": word}, {"label": word}
    return {"text": f"Hello, my name is {person}. I live in {city} and enjoy {subject}."}, {}


def sample_dataset(key):
    def generate(module, options):
        rng = Random(options["seed"])
        samples = []
        if key in ("transcription", "keywords", "speech"):
            import torch

            with RUNTIME_LOCK, torch.inference_mode():
                model, processor = runtime("speech")
                with torch.random.fork_rng(devices=[]):
                    torch.random.default_generator.manual_seed(options["seed"])
                    settings = {name: options[name] for name in ("speed_percent", "noise_percent")}
                    for index in range(options["samples"]):
                        fields, target = sample_text(key, rng)
                        data = speech_bytes(fields["text"], model, processor, **settings,
                                            noise_seed=options["seed"] + index)
                        samples.append(({"name": f"sample-{index + 1:02}.wav", "type": "audio/wav",
                                         "data": encode(data), "target": {**target, "transcript": fields["text"]},
                                         **({"fields": fields, "settings": settings} if key == "speech" else {})}, data))
            return MediaDataset([sample for sample, _ in samples], [(sample["name"], data) for sample, data in samples],
                                [{"audio": sample["name"], **sample["target"]} for sample, _ in samples])
        for index in range(options["samples"]):
            fields, target = sample_text(key, rng)
            samples.append({"name": f"样本 {index + 1}", "fields": fields, "target": target})
        columns = list(samples[0]["fields"])
        rows = [[*(sample["fields"][column] for column in columns), json.dumps(sample["target"], ensure_ascii=False)]
                for sample in samples]
        data = table_csv([*columns, "reference"], rows)
        return MediaDataset(samples, [("samples.csv", data)], samples)
    return generate
