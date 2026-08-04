# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## リポジトリの内容

Qwen3-TTS（MLX 版）でボイスクローンを行うローカル Gradio アプリ。`app.py`（UI と生成処理）と `theme.py`（見た目）の 2 ファイル構成。Apple Silicon 専用。git 管理下、公開リポジトリ。

## 実行方法

```bash
# 環境構築
uv venv --python 3.12 .venv
VIRTUAL_ENV=.venv uv pip install -r requirements.txt

# 起動（http://127.0.0.1:7860）
./run.sh

# UI を立てずに生成だけ確認する
.venv/bin/python -c "
import app
r = app.synthesize(ref_audio='ref.wav', ref_text='参照音声の書き起こし',
                   text='テストです。', lang='japanese', temperature=0.9,
                   top_k=50, top_p=1.0, repetition_penalty=1.5,
                   max_tokens=2048, chunk_chars=150, pause_sec=0.2, seed=0)
print(r[1])"
```

自動テストは無い。UI の見た目を変えたときは、ヘッドレス Chrome でスクリーンショットを撮って確認する。

```bash
"/Applications/Google Chrome.app/Contents/MacOS/Google Chrome" --headless=new --disable-gpu \
  --screenshot=/tmp/ui.png --window-size=1380,1250 --virtual-time-budget=9000 http://127.0.0.1:7860
```

依存:
- Apple Silicon（MLX のため Intel Mac では動かない）
- Python 3.12 / mlx-audio / gradio 6
- モデル 3.1GB は初回実行時に Hugging Face から自動取得され `~/.cache/huggingface/hub/` に入る

## コードの構造

- `app.py` — UI 定義と処理の両方。`_model` / `_stt` はモジュールグローバルの遅延シングルトン（`get_model()` / `get_stt()` 経由でのみ触る）
- `theme.py` — 配色トークン、`build_theme()`、`CSS`、`HEADER_HTML`、`step_html()`。機能側からは表示にしか使われないので、まるごと差し替えても生成処理は壊れない
- `model.generate()` はジェネレータで `GenerationResult` を yield する。音声は `result.audio`（`mx.array`）なので `np.asarray(...)` で変換してから連結する
- `synthesize()` の戻り値は `(sample_rate, audio), 保存パス, 結果カードの HTML, 等倍の生データ` の 4 つ組。3 番目は `gr.HTML` に流し込む前提の文字列で、`theme.py` の `.result-card` 系 CSS に依存している。4 番目は `gr.State` に入り、`apply_speed()` が TTS を再実行せず速度だけ掛け直すために使う
- 保存とカード生成は `_finalize()` に集約してある。速度の適用・ファイル名の接尾辞・メトリクス表示はすべてここを通る

## 規約・地雷

**mlx-audio の挙動**

- `ref_text` が `None` だと ICL モードに入らない。mlx-audio 側の判定は `ref_audio is not None and ref_text is not None and speech_tokenizer.has_encoder`。外れると話者埋め込みのみのモードに落ちてクローン精度が明確に下がる。UI が警告を出しているのはこのため
- ICL モードは**テキストを分割しない**。長文をそのまま渡すと 1 回の生成で処理される。`app.py` 側で `split_text()` してチャンクごとに `generate()` を呼び、無音を挟んで結合しているのはこの穴を埋めるため。`split_pattern=None` を渡しているのは mlx-audio 側の二重分割を避けるため
- ICL モードでは `repetition_penalty` に `max(値, 1.5)` が適用される。UI で 1.0 にしても 1.5 として扱われる
- 参照音声と参照テキストの組は mlx-audio 内部でキャッシュされる。参照テキストを書き換えると再エンコードが走る
- `generate(speed=...)` は Qwen3-TTS では**効かない**（mlx-audio の docstring に "not directly supported yet"、`_generate_icl` にも渡っていない）。話速は `change_speed()` の WSOLA で生成後に伸縮している。ここをリサンプリングに置き換えると声の高さまで変わるので不可
- RTF は速度適用**前**の長さで計算する。速度を掛けた後の長さで割ると生成コストの指標にならない

**Gradio 6 固有**

- `panel` は予約クラス名で、`elem_classes="panel"` を指定しても DOM から除去される。独自クラスは `studio-` 接頭辞をつける
- `css` / `js` / `theme` は `Blocks()` ではなく `launch()` に渡す。`app.LAUNCH_KWARGS` にまとめてあるので、別ポートで起動するときもこれを展開して渡す
- `gr.themes.Base.set()` は存在しないキーワードを渡すと `TypeError`。`*_dark` が無い変数もあるため、`build_theme()` はシグネチャと突き合わせてから渡している
- 起動ログの `To create a public link, ...` は警告ではなく `share=False` のときの案内。`launch(quiet=True)` で消せるが URL 表示も消えるので、`app.py` 側で `[serve] http://...` を自前で出している

**transformers**

- transformers は `model_type: qwen3_tts` を知らない。mlx-audio が `AutoTokenizer.from_pretrained()` を呼ぶと `AutoConfig` の解決に失敗して基底クラスに落ち、型不一致の警告が出る。トークナイザは `tokenizer_config.json` の `Qwen2Tokenizer` が使われるので実害は無いが、`get_model()` の `_register_qwen3_tts_config()` で型だけ登録して黙らせている。transformers が正式対応したら登録をスキップする
- `transformers` を `mlx_audio` より先に import しない。`mlx_audio/__init__.py` が `TRANSFORMERS_NO_ADVISORY_WARNINGS` を立てているので、順序を逆にすると `PyTorch was not found` の助言警告が漏れる

**日本語テキスト**

- 読みの制御は表記でしか行えない（アクセント辞書の仕組みは無い）。難読な固有名詞は読めないので、テキスト側をひらがなに置き換える。常用漢字・数字の読み分けは問題ない
- 全文をひらがなにするのは避ける。語の切れ目が失われて分節を誤ることがある

**その他**

- `outputs/` と `voices/` は音声＝個人情報になりうるため `.gitignore` 済み。コミットしない
- 公開リポジトリのため、このリポジトリのローカル git 設定で `user.email` を GitHub の noreply アドレスにしてある。グローバル設定に戻さない
- コメント・UI 文言・ドキュメントは日本語で統一する
