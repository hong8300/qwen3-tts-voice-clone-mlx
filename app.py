"""Qwen3-TTS ボイスクローン専用 Gradio UI (Apple Silicon / MLX)."""

from __future__ import annotations

import json
import os
import re
import time
import unicodedata
from datetime import datetime
from pathlib import Path

import gradio as gr
import mlx.core as mx
import numpy as np
import soundfile as sf

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
def get_model():
    global _model
    if _model is None:
        print(f"[load] TTS model: {MODEL_ID}")
        from mlx_audio.tts.utils import load_model

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
    duration = len(audio) / sample_rate

    out_path = OUTPUT_DIR / f"{datetime.now():%Y%m%d-%H%M%S}.wav"
    sf.write(out_path, audio, sample_rate)

    info = (
        f"✅ {duration:.1f} 秒の音声を生成 / 所要 {elapsed:.1f} 秒 "
        f"(RTF {elapsed / duration:.2f}, {len(chunks)} チャンク, {total_tokens} トークン)\n"
        f"保存先: {out_path}"
    )
    return (sample_rate, audio), str(out_path), info


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
with gr.Blocks(title="Qwen3-TTS Voice Clone") as demo:
    gr.Markdown(
        f"""# 🎙️ Qwen3-TTS ボイスクローン
`{MODEL_ID}` / Apple Silicon (MLX)

**使い方** — ①参照音声をアップロード（3〜10秒程度のクリアな音声）→ ②その音声の内容を
参照テキストに入力（「自動文字起こし」でも可）→ ③読み上げたいテキストを入力 → ④生成。
"""
    )

    with gr.Row():
        with gr.Column(scale=1):
            gr.Markdown("### 1. 参照音声（クローン元）")
            ref_audio = gr.Audio(
                label="参照音声をアップロード / 録音",
                sources=["upload", "microphone"],
                type="filepath",
            )
            ref_text = gr.Textbox(
                label="参照テキスト（参照音声で実際に話している内容）",
                lines=3,
                placeholder="例：こんにちは、今日はいい天気ですね。",
                info="精度に直結します。正確に入力してください。",
            )
            stt_btn = gr.Button("🎧 自動文字起こし（Whisper）", size="sm")

            with gr.Accordion("音声ライブラリ（保存して再利用）", open=False):
                voice_name = gr.Textbox(label="名前", placeholder="例：田中さん")
                save_btn = gr.Button("💾 現在の参照音声を保存", size="sm")
                voice_select = gr.Dropdown(
                    label="保存済みの音声", choices=list_voices(), value=None
                )
                with gr.Row():
                    load_btn = gr.Button("📂 読み込み", size="sm")
                    del_btn = gr.Button("🗑️ 削除", size="sm", variant="stop")

        with gr.Column(scale=1):
            gr.Markdown("### 2. 読み上げるテキスト")
            text = gr.Textbox(
                label="テキスト",
                lines=10,
                placeholder="ここに読み上げたい文章を入力します。長文は自動で分割されます。",
            )
            lang = gr.Dropdown(
                label="言語", choices=LANGUAGES, value="auto", filterable=False
            )
            run_btn = gr.Button("🔊 生成", variant="primary", size="lg")

            out_audio = gr.Audio(label="生成された音声", type="numpy", autoplay=False)
            out_file = gr.File(label="ダウンロード")
            out_info = gr.Markdown()

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
            repetition_penalty, max_tokens, chunk_chars, pause_sec, seed,
        ],
        outputs=[out_audio, out_file, out_info],
    )


if __name__ == "__main__":
    if os.environ.get("QWEN3_TTS_PRELOAD", "1") == "1":
        get_model()
    demo.launch(
        theme=gr.themes.Soft(),
        server_name=os.environ.get("QWEN3_TTS_HOST", "127.0.0.1"),
        server_port=int(os.environ.get("QWEN3_TTS_PORT", "7860")),
        inbrowser=True,
    )
