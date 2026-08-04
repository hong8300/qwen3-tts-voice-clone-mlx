"""Qwen3-TTS ボイスクローン専用 Gradio UI (Apple Silicon / MLX)."""

from __future__ import annotations

import json
import os
import re
import time
import unicodedata
from datetime import datetime
from html import escape
from pathlib import Path

import gradio as gr
import mlx.core as mx
import numpy as np
import soundfile as sf

from theme import CSS, HEADER_HTML, build_theme, step_html

MODEL_ID = os.environ.get("QWEN3_TTS_MODEL", "mlx-community/Qwen3-TTS-12Hz-1.7B-Base-8bit")
STT_MODEL_ID = os.environ.get(
    "QWEN3_TTS_STT_MODEL", "mlx-community/whisper-large-v3-turbo-asr-fp16"
)

BASE_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = BASE_DIR / "outputs"
VOICE_DIR = BASE_DIR / "voices"
OUTPUT_DIR.mkdir(exist_ok=True)
VOICE_DIR.mkdir(exist_ok=True)

LANGUAGES = [
    ("自動判定", "auto"),
    ("日本語", "japanese"),
    ("英語", "english"),
    ("中国語", "chinese"),
    ("韓国語", "korean"),
    ("ドイツ語", "german"),
    ("フランス語", "french"),
    ("イタリア語", "italian"),
    ("スペイン語", "spanish"),
    ("ポルトガル語", "portuguese"),
    ("ロシア語", "russian"),
]

_model = None
_stt = None


# --------------------------------------------------------------------------
# モデル読み込み
# --------------------------------------------------------------------------
def _register_qwen3_tts_config() -> None:
    """transformers に無い model_type `qwen3_tts` を登録する。

    mlx-audio は post_load_hook で AutoTokenizer.from_pretrained() を呼ぶ。その中で
    AutoConfig が qwen3_tts を解決できず基底クラス PreTrainedConfig に落ちるため
    「You are using a model of type `qwen3_tts` to instantiate a model of type ``」
    という警告が出る。トークナイザ自体は tokenizer_config.json の Qwen2Tokenizer が
    使われるので実害は無いが、型だけ登録しておけば解決が成功して警告も出ない。
    transformers が正式対応したら登録しない（本物の設定クラスを隠さないため）。
    呼ぶのは mlx_audio を import した後。mlx_audio/__init__.py が
    TRANSFORMERS_NO_ADVISORY_WARNINGS を立てているので、先に transformers を
    import すると別の助言警告（PyTorch was not found）が漏れる。
    """
    from transformers import AutoConfig, PretrainedConfig
    from transformers.models.auto.configuration_auto import CONFIG_MAPPING_NAMES

    if "qwen3_tts" in CONFIG_MAPPING_NAMES:
        return

    class Qwen3TTSConfig(PretrainedConfig):
        model_type = "qwen3_tts"

    AutoConfig.register("qwen3_tts", Qwen3TTSConfig, exist_ok=True)


def get_model():
    global _model
    if _model is None:
        print(f"[load] TTS model: {MODEL_ID}")
        from mlx_audio.tts.utils import load_model

        _register_qwen3_tts_config()
        t0 = time.time()
        _model = load_model(MODEL_ID)
        print(f"[load] done in {time.time() - t0:.1f}s")
    return _model


def get_stt():
    global _stt
    if _stt is None:
        print(f"[load] STT model: {STT_MODEL_ID}")
        from mlx_audio.stt import load as load_stt_model

        _stt = load_stt_model(STT_MODEL_ID)
    return _stt


# --------------------------------------------------------------------------
# テキスト分割
# --------------------------------------------------------------------------
_SENT_END = re.compile(r"(?<=[。．！？!?\.])\s*")


def split_text(text: str, max_chars: int) -> list[str]:
    """長文を文単位で max_chars 以下のかたまりに分割する。"""
    chunks: list[str] = []
    for para in [p.strip() for p in text.split("\n") if p.strip()]:
        if len(para) <= max_chars:
            chunks.append(para)
            continue
        buf = ""
        for sent in [s for s in _SENT_END.split(para) if s]:
            # 1文が長すぎる場合は読点で分割
            pieces = [sent]
            if len(sent) > max_chars:
                pieces = [p for p in re.split(r"(?<=[、,])\s*", sent) if p]
            for piece in pieces:
                if buf and len(buf) + len(piece) > max_chars:
                    chunks.append(buf.strip())
                    buf = ""
                buf += piece
        if buf.strip():
            chunks.append(buf.strip())
    return chunks or ([text.strip()] if text.strip() else [])


# --------------------------------------------------------------------------
# 再生速度（WSOLA によるピッチ保持のテンポ変更）
# --------------------------------------------------------------------------
def change_speed(audio: np.ndarray, speed: float, sample_rate: int) -> np.ndarray:
    """速度を変える。speed<1 で遅く、>1 で速くなる。ピッチは変わらない。

    Qwen3-TTS 側に速度指定が無いため生成後に伸縮する。単純なリサンプリングだと
    声の高さまで変わってしまうので、波形の周期に合わせて重ね合わせる WSOLA を使う。
    """
    if abs(speed - 1.0) < 1e-3 or len(audio) == 0:
        return audio.astype(np.float32)

    frame = int(sample_rate * 0.040)  # 40ms の分析窓
    hop_out = frame // 2  # 出力側ホップ（50% オーバーラップ）
    hop_in = max(1, int(round(hop_out * speed)))  # 入力側ホップ
    search = int(sample_rate * 0.010)  # 前フレームとの整合を探す幅（±10ms）
    overlap = frame - hop_out
    window = np.hanning(frame).astype(np.float32)

    x = np.concatenate(
        [audio.astype(np.float32), np.zeros(frame + search + hop_in, np.float32)]
    )
    out_len = int(len(audio) / speed) + 2 * frame
    out = np.zeros(out_len, np.float32)
    norm = np.zeros(out_len, np.float32)

    pos_in = pos_out = 0
    tail = None
    while pos_in + frame + search < len(x) and pos_out + frame < out_len:
        if tail is None:
            best = pos_in
        else:
            # 直前フレームの後半と最も似た位置を探し、位相のずれによる濁りを防ぐ
            lo = max(0, pos_in - search)
            hi = min(len(x) - frame, pos_in + search)
            if hi <= lo:
                best = min(pos_in, len(x) - frame)
            else:
                seg = x[lo : hi + overlap]
                best = lo + int(np.argmax(np.correlate(seg, tail, mode="valid")))
        out[pos_out : pos_out + frame] += x[best : best + frame] * window
        norm[pos_out : pos_out + frame] += window
        tail = x[best + hop_out : best + hop_out + overlap]
        pos_out += hop_out
        pos_in += hop_in

    end = pos_out + frame
    norm[:end][norm[:end] < 1e-6] = 1.0
    return (out[:end] / norm[:end]).astype(np.float32)


# --------------------------------------------------------------------------
# 文字起こし
# --------------------------------------------------------------------------
def transcribe(ref_audio: str | None) -> str:
    if not ref_audio:
        raise gr.Error("先に参照音声をアップロードしてください。")
    gr.Info("Whisper で文字起こし中… (初回はモデルを取得します)")
    text = get_stt().generate(ref_audio).text.strip()
    if not text:
        raise gr.Error("文字起こしに失敗しました。手動で入力してください。")
    return text


# --------------------------------------------------------------------------
# 音声生成
# --------------------------------------------------------------------------
def synthesize(
    ref_audio: str | None,
    ref_text: str,
    text: str,
    lang: str,
    temperature: float,
    top_k: int,
    top_p: float,
    repetition_penalty: float,
    max_tokens: int,
    chunk_chars: int,
    pause_sec: float,
    seed: int,
    speed: float = 1.0,
    progress=gr.Progress(),
):
    if not ref_audio:
        raise gr.Error("参照音声をアップロードしてください。")
    text = (text or "").strip()
    if not text:
        raise gr.Error("読み上げるテキストを入力してください。")

    ref_text = (ref_text or "").strip() or None
    if ref_text is None:
        gr.Warning(
            "参照テキストが空のため話者埋め込みのみのモードで生成します"
            "（クローン精度は下がります）。"
        )

    model = get_model()
    if seed >= 0:
        mx.random.seed(int(seed))

    chunks = split_text(text, int(chunk_chars))
    audio_parts: list[np.ndarray] = []
    sample_rate = 24000
    total_tokens = 0
    t0 = time.time()

    for i, chunk in enumerate(chunks):
        progress((i, len(chunks)), desc=f"生成中 {i + 1}/{len(chunks)}")
        for result in model.generate(
            text=chunk,
            ref_audio=ref_audio,
            ref_text=ref_text,
            lang_code=lang,
            temperature=float(temperature),
            top_k=int(top_k),
            top_p=float(top_p),
            repetition_penalty=float(repetition_penalty),
            max_tokens=int(max_tokens),
            split_pattern=None,
            verbose=True,
        ):
            audio_parts.append(np.asarray(result.audio, dtype=np.float32).reshape(-1))
            sample_rate = result.sample_rate
            total_tokens += result.token_count
        if pause_sec > 0 and i < len(chunks) - 1:
            audio_parts.append(np.zeros(int(sample_rate * pause_sec), dtype=np.float32))

    if not audio_parts:
        raise gr.Error("音声が生成されませんでした。設定を変えて再試行してください。")

    audio = np.concatenate(audio_parts)
    elapsed = time.time() - t0
    raw_duration = len(audio) / sample_rate

    extra = [
        (f"{elapsed:.1f}s", "生成時間"),
        (f"{elapsed / raw_duration:.2f}", "RTF"),
        (str(len(chunks)), "チャンク"),
        (str(total_tokens), "トークン"),
    ]
    return (*_finalize(audio, sample_rate, speed, extra), (sample_rate, audio))


def _finalize(
    audio: np.ndarray,
    sample_rate: int,
    speed: float,
    extra_metrics: list[tuple[str, str]] | None = None,
    head: str = "生成が完了しました",
):
    """速度を適用して WAV に保存し、(音声, 保存パス, 結果カード HTML) を返す。"""
    out = change_speed(audio, float(speed), sample_rate)
    duration = len(out) / sample_rate

    suffix = "" if abs(speed - 1.0) < 1e-3 else f"_x{speed:.2f}"
    out_path = OUTPUT_DIR / f"{datetime.now():%Y%m%d-%H%M%S}{suffix}.wav"
    sf.write(out_path, out, sample_rate)

    metrics = [(f"{duration:.1f}s", "音声の長さ"), (f"{speed:.2f}x", "再生速度")]
    metrics += extra_metrics or []
    cells = "".join(f"<div><b>{v}</b><span>{k}</span></div>" for v, k in metrics)
    info = (
        f'<div class="result-card"><div class="result-head">{head}</div>'
        f'<div class="result-metrics">{cells}</div>'
        f'<div class="result-path">{escape(str(out_path))}</div></div>'
    )
    return (sample_rate, out), str(out_path), info


def apply_speed(raw, speed: float):
    """生成済みの音声に速度だけ掛け直す（TTS は再実行しない）。"""
    if not raw:
        raise gr.Error("先に音声を生成してください。")
    sample_rate, audio = raw
    return _finalize(audio, sample_rate, speed, head="速度を変更しました")


# --------------------------------------------------------------------------
# 音声ライブラリ
# --------------------------------------------------------------------------
def _slug(name: str) -> str:
    name = unicodedata.normalize("NFKC", name).strip()
    return re.sub(r'[/\\:*?"<>|\s]+', "_", name)[:60]


def list_voices() -> list[str]:
    return sorted(p.stem for p in VOICE_DIR.glob("*.json"))


def save_voice(name: str, ref_audio: str | None, ref_text: str):
    if not (name or "").strip():
        raise gr.Error("保存する名前を入力してください。")
    if not ref_audio:
        raise gr.Error("参照音声がありません。")
    slug = _slug(name)
    audio, sr = sf.read(ref_audio, dtype="float32", always_2d=False)
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    sf.write(VOICE_DIR / f"{slug}.wav", audio, sr)
    (VOICE_DIR / f"{slug}.json").write_text(
        json.dumps({"name": name.strip(), "ref_text": ref_text or ""}, ensure_ascii=False),
        encoding="utf-8",
    )
    gr.Info(f"「{name}」を保存しました。")
    return gr.update(choices=list_voices(), value=slug)


def load_voice(slug: str):
    if not slug:
        raise gr.Error("読み込む音声を選択してください。")
    meta = json.loads((VOICE_DIR / f"{slug}.json").read_text(encoding="utf-8"))
    return str(VOICE_DIR / f"{slug}.wav"), meta.get("ref_text", ""), meta.get("name", slug)


def delete_voice(slug: str):
    if not slug:
        raise gr.Error("削除する音声を選択してください。")
    for suffix in (".wav", ".json"):
        (VOICE_DIR / f"{slug}{suffix}").unlink(missing_ok=True)
    gr.Info(f"「{slug}」を削除しました。")
    return gr.update(choices=list_voices(), value=None)


# --------------------------------------------------------------------------
# UI
# --------------------------------------------------------------------------
with gr.Blocks(title="Voice Clone Studio — Qwen3-TTS") as demo:
    gr.HTML(HEADER_HTML)

    with gr.Row(equal_height=False):
        with gr.Column(scale=1, elem_classes="studio-panel"):
            gr.HTML(
                step_html(
                    "1",
                    "参照音声（クローン元）",
                    "3〜10 秒程度、雑音・BGM のないクリアな音声が最適です。",
                )
            )
            ref_audio = gr.Audio(
                label="参照音声をアップロード / 録音",
                sources=["upload", "microphone"],
                type="filepath",
            )
            gr.HTML(
                step_html(
                    "2",
                    "参照テキスト",
                    "参照音声で実際に話している内容。精度に直結します。",
                )
            )
            ref_text = gr.Textbox(
                label="参照テキスト",
                lines=3,
                placeholder="例：こんにちは、今日はいい天気ですね。",
                info="難読な固有名詞はひらがなで書くと読みが安定します。",
            )
            stt_btn = gr.Button(
                "自動文字起こし（Whisper）", size="sm", elem_classes="ic-captions"
            )

            with gr.Accordion("音声ライブラリ（保存して再利用）", open=False):
                voice_name = gr.Textbox(label="名前", placeholder="例：ナレーターA")
                save_btn = gr.Button(
                    "現在の参照音声を保存", size="sm", elem_classes="ic-save"
                )
                voice_select = gr.Dropdown(
                    label="保存済みの音声", choices=list_voices(), value=None
                )
                with gr.Row():
                    load_btn = gr.Button("読み込み", size="sm", elem_classes="ic-folder")
                    del_btn = gr.Button(
                        "削除", size="sm", variant="stop", elem_classes="ic-trash"
                    )

        with gr.Column(scale=1, elem_classes="studio-panel"):
            gr.HTML(
                step_html(
                    "3",
                    "読み上げるテキスト",
                    "長文は自動で分割して生成し、つなげて出力します。",
                )
            )
            text = gr.Textbox(
                label="テキスト",
                lines=10,
                placeholder="ここに読み上げたい文章を入力します。",
            )
            lang = gr.Dropdown(
                label="言語", choices=LANGUAGES, value="auto", filterable=False
            )
            speed = gr.Slider(
                0.5,
                2.0,
                value=1.0,
                step=0.05,
                label="再生速度",
                info="0.9 でゆっくり、1.2 で速く。声の高さは変わりません。",
            )
            run_btn = gr.Button(
                "生成する", variant="primary", size="lg", elem_classes=["cta", "ic-play"]
            )

            out_audio = gr.Audio(
                label="生成された音声", type="numpy", autoplay=False, elem_classes="slim"
            )
            speed_btn = gr.Button(
                "速度だけ変えて出力し直す", size="sm", elem_classes="ic-speed"
            )
            out_file = gr.File(label="ダウンロード", elem_classes="slim")
            out_info = gr.HTML()
            raw_audio_state = gr.State()

    with gr.Accordion("詳細設定", open=False):
        with gr.Row():
            temperature = gr.Slider(0.1, 1.5, value=0.9, step=0.05, label="temperature")
            top_k = gr.Slider(1, 100, value=50, step=1, label="top_k")
            top_p = gr.Slider(0.1, 1.0, value=1.0, step=0.05, label="top_p")
        with gr.Row():
            repetition_penalty = gr.Slider(
                1.0, 2.0, value=1.5, step=0.05, label="repetition_penalty"
            )
            max_tokens = gr.Slider(
                256, 4096, value=2048, step=128, label="max_tokens（1チャンクあたり）"
            )
            seed = gr.Number(value=-1, precision=0, label="シード（-1でランダム）")
        with gr.Row():
            chunk_chars = gr.Slider(
                50, 500, value=150, step=10, label="分割文字数（長文の1チャンク上限）"
            )
            pause_sec = gr.Slider(
                0.0, 1.0, value=0.2, step=0.05, label="チャンク間の無音（秒）"
            )

    stt_btn.click(transcribe, inputs=ref_audio, outputs=ref_text)
    save_btn.click(save_voice, inputs=[voice_name, ref_audio, ref_text], outputs=voice_select)
    load_btn.click(load_voice, inputs=voice_select, outputs=[ref_audio, ref_text, voice_name])
    del_btn.click(delete_voice, inputs=voice_select, outputs=voice_select)
    run_btn.click(
        synthesize,
        inputs=[
            ref_audio, ref_text, text, lang, temperature, top_k, top_p,
            repetition_penalty, max_tokens, chunk_chars, pause_sec, seed, speed,
        ],
        outputs=[out_audio, out_file, out_info, raw_audio_state],
    )
    speed_btn.click(
        apply_speed,
        inputs=[raw_audio_state, speed],
        outputs=[out_audio, out_file, out_info],
    )


LAUNCH_KWARGS = dict(theme=build_theme(), css=CSS)


if __name__ == "__main__":
    if os.environ.get("QWEN3_TTS_PRELOAD", "1") == "1":
        get_model()
    host = os.environ.get("QWEN3_TTS_HOST", "127.0.0.1")
    port = int(os.environ.get("QWEN3_TTS_PORT", "7860"))
    print(f"[serve] http://{host}:{port}")
    # quiet=True は共有リンクの案内を消すためだが、URL 表示も一緒に消えるので自前で出す
    demo.launch(
        **LAUNCH_KWARGS,
        server_name=host,
        server_port=port,
        inbrowser=True,
        quiet=True,
    )
