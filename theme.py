"""ダークスタジオ配色のテーマと CSS（見た目のみ・機能には影響しない）。"""

from __future__ import annotations

import inspect
from urllib.parse import quote

import gradio as gr

# --------------------------------------------------------------------------
# 配色トークン
# --------------------------------------------------------------------------
BG = "#0F0F23"  # 背景（深い紺）
PANEL = "#16162F"  # パネル
PANEL_HI = "#1C1C3B"  # パネル（明）
LINE = "#2A2A4D"  # 罫線
TEXT = "#F8FAFC"  # 本文
MUTED = "#A9ADCE"  # 補助テキスト（BG 比 7.4:1）
ACCENT = "#F97316"  # アクセント（オレンジ）
ACCENT_HI = "#FB923C"
ACCENT_INK = "#1F1103"  # オレンジ上の文字（コントラスト確保のため濃色）
INDIGO = "#4F46E5"

FONTS = [
    "-apple-system",
    "BlinkMacSystemFont",
    "Hiragino Sans",
    "Noto Sans JP",
    "system-ui",
    "sans-serif",
]
FONTS_MONO = ["SF Mono", "SFMono-Regular", "Menlo", "monospace"]


# --------------------------------------------------------------------------
# アイコン（絵文字ではなく SVG を mask で描画し、文字色に追従させる）
# --------------------------------------------------------------------------
_ICON_PATHS = {
    "mic": (
        "<path d='M12 2a3 3 0 0 0-3 3v7a3 3 0 0 0 6 0V5a3 3 0 0 0-3-3Z'/>"
        "<path d='M19 10v2a7 7 0 0 1-14 0v-2'/><path d='M12 19v3'/>"
    ),
    "captions": (
        "<rect x='3' y='5' width='18' height='14' rx='2'/>"
        "<path d='M7 15h4'/><path d='M15 15h2'/><path d='M7 11h10'/>"
    ),
    "save": (
        "<path d='M19 21H5a2 2 0 0 1-2-2V5a2 2 0 0 1 2-2h11l5 5v11a2 2 0 0 1-2 2Z'/>"
        "<path d='M17 21v-8H7v8'/><path d='M7 3v5h8'/>"
    ),
    "folder": (
        "<path d='M4 20a2 2 0 0 1-2-2V5a2 2 0 0 1 2-2h4l2 3h8a2 2 0 0 1 2 2v2'/>"
        "<path d='m6 20 2.4-7.2A2 2 0 0 1 10.3 11H21a1 1 0 0 1 .95 1.32L19.6 19.4A2 2 0"
        " 0 1 17.7 20Z'/>"
    ),
    "trash": (
        "<path d='M3 6h18'/><path d='M8 6V4a1 1 0 0 1 1-1h6a1 1 0 0 1 1 1v2'/>"
        "<path d='M19 6v14a2 2 0 0 1-2 2H7a2 2 0 0 1-2-2V6'/>"
        "<path d='M10 11v6'/><path d='M14 11v6'/>"
    ),
    "play": "<path d='M6 4.5v15l13-7.5Z'/>",
    "speed": "<path d='m12 14 4-4'/><path d='M3.34 19a10 10 0 1 1 17.32 0'/>",
    "check": (
        "<path d='M22 11.1V12a10 10 0 1 1-5.9-9.1'/><path d='m9 11 3 3 10-10'/>"
    ),
    "wave": (
        "<path d='M2 12h2'/><path d='M7 7v10'/><path d='M12 3v18'/>"
        "<path d='M17 8v8'/><path d='M22 11v2'/>"
    ),
}


def _icon_uri(body: str) -> str:
    svg = (
        "<svg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 24 24' fill='none' "
        "stroke='black' stroke-width='2' stroke-linecap='round' "
        f"stroke-linejoin='round'>{body}</svg>"
    )
    return "data:image/svg+xml;utf8," + quote(svg, safe="")


ICON_URIS = {name: _icon_uri(body) for name, body in _ICON_PATHS.items()}

_ICON_RULES = "\n".join(
    f'button.ic-{name}::before, .ic-{name} > button::before {{'
    f'-webkit-mask-image:url("{uri}");mask-image:url("{uri}");}}'
    for name, uri in ICON_URIS.items()
)


# --------------------------------------------------------------------------
# テーマ（ライト／ダーク両方に同じ暗色を入れ、OS 設定に関係なく同じ見た目にする）
# --------------------------------------------------------------------------
def build_theme() -> gr.themes.Base:
    # ライト用の変数名に加え、対応する *_dark が存在する場合のみ同じ値を入れる
    valid = set(inspect.signature(gr.themes.Base.set).parameters)
    kwargs: dict[str, str] = {}
    for key, value in {
        "body_background_fill": BG,
        "body_text_color": TEXT,
        "body_text_color_subdued": MUTED,
        "background_fill_primary": PANEL,
        "background_fill_secondary": BG,
        "block_background_fill": PANEL,
        "block_border_color": LINE,
        "block_label_background_fill": "transparent",
        "block_label_text_color": MUTED,
        "block_title_text_color": TEXT,
        "block_info_text_color": MUTED,
        "border_color_primary": LINE,
        "border_color_accent": INDIGO,
        "panel_background_fill": PANEL,
        "panel_border_color": LINE,
        "input_background_fill": "#12122A",
        "input_background_fill_focus": "#151534",
        "input_border_color": LINE,
        "input_border_color_focus": ACCENT,
        "input_placeholder_color": "#6F739A",
        "button_primary_background_fill": f"linear-gradient(135deg,{ACCENT},{ACCENT_HI})",
        "button_primary_background_fill_hover": f"linear-gradient(135deg,{ACCENT_HI},{ACCENT})",
        "button_primary_text_color": ACCENT_INK,
        "button_primary_text_color_hover": ACCENT_INK,
        "button_primary_border_color": ACCENT,
        "button_secondary_background_fill": PANEL_HI,
        "button_secondary_background_fill_hover": "#24244B",
        "button_secondary_text_color": TEXT,
        "button_secondary_text_color_hover": TEXT,
        "button_secondary_border_color": LINE,
        "button_cancel_background_fill": "#3A1620",
        "button_cancel_text_color": "#FCA5A5",
        "button_cancel_border_color": "#5B2230",
        "checkbox_background_color": "#12122A",
        "checkbox_border_color": LINE,
        "slider_color": ACCENT,
        "link_text_color": ACCENT_HI,
        "link_text_color_hover": ACCENT,
        "table_border_color": LINE,
        "table_even_background_fill": PANEL,
        "table_odd_background_fill": PANEL_HI,
        "color_accent_soft": PANEL_HI,
        "shadow_drop": "0 1px 2px rgba(0,0,0,.35)",
        "shadow_drop_lg": "0 12px 32px rgba(0,0,0,.45)",
    }.items():
        for name in (key, f"{key}_dark"):
            if name in valid:
                kwargs[name] = value

    return gr.themes.Base(
        font=FONTS,
        font_mono=FONTS_MONO,
        radius_size=gr.themes.sizes.radius_lg,
        spacing_size=gr.themes.sizes.spacing_lg,
        text_size=gr.themes.sizes.text_md,
    ).set(**kwargs)


# --------------------------------------------------------------------------
# CSS
# --------------------------------------------------------------------------
CSS = f"""
.gradio-container {{
  --studio-bg: {BG};
  --studio-panel: {PANEL};
  --studio-line: {LINE};
  --studio-text: {TEXT};
  --studio-muted: {MUTED};
  --studio-accent: {ACCENT};
  --studio-accent-hi: {ACCENT_HI};
  --studio-indigo: {INDIGO};
  max-width: 1180px !important;
  margin: 0 auto !important;
  padding: 20px 18px 56px !important;
}}

body, gradio-app {{
  background:
    radial-gradient(900px 420px at 12% -8%, rgba(79,70,229,.16), transparent 60%),
    radial-gradient(760px 380px at 92% 4%, rgba(249,115,22,.10), transparent 62%),
    {BG} !important;
}}

/* ---------- ヘッダー ---------- */
.studio-header {{
  position: relative;
  overflow: hidden;
  border: 1px solid var(--studio-line);
  border-radius: 20px;
  padding: 22px 26px;
  margin-bottom: 22px;
  background:
    linear-gradient(135deg, rgba(79,70,229,.20), rgba(15,15,35,0) 55%),
    linear-gradient(180deg, {PANEL_HI}, {PANEL});
  box-shadow: 0 16px 40px rgba(0,0,0,.42), inset 0 1px 0 rgba(255,255,255,.06);
}}
.studio-header::after {{
  content: "";
  position: absolute; inset: 0;
  background: radial-gradient(420px 140px at 82% 0%, rgba(249,115,22,.16), transparent 70%);
  pointer-events: none;
}}
.studio-top {{ display:flex; align-items:center; justify-content:space-between; gap:16px; flex-wrap:wrap; }}
.studio-brand {{ display:flex; align-items:center; gap:14px; }}
.studio-mark {{
  width:44px; height:44px; border-radius:14px; flex:none;
  display:grid; place-items:center;
  background: linear-gradient(135deg, var(--studio-indigo), #312E81);
  box-shadow: 0 6px 18px rgba(79,70,229,.42), inset 0 1px 0 rgba(255,255,255,.18);
}}
.studio-mark svg {{ width:22px; height:22px; stroke:#EEF0FF; }}
.studio-title {{
  margin:0; font-size:1.55rem; font-weight:700; letter-spacing:-.02em; line-height:1.2;
  background: linear-gradient(92deg, #FFFFFF 10%, #C3C7F2 90%);
  -webkit-background-clip: text; background-clip: text; color: transparent;
}}
.studio-sub {{ margin:3px 0 0; font-size:.83rem; color:var(--studio-muted); letter-spacing:.02em; }}
.studio-chip {{
  display:inline-flex; align-items:center; gap:8px;
  padding:7px 13px; border-radius:999px; white-space:nowrap;
  border:1px solid rgba(249,115,22,.34);
  background: rgba(249,115,22,.10);
  color:#FDBA74; font-size:.76rem; font-weight:600; letter-spacing:.04em;
}}
.studio-dot {{
  width:7px; height:7px; border-radius:50%; background:var(--studio-accent);
  box-shadow:0 0 0 3px rgba(249,115,22,.18); animation: studioPulse 2.6s ease-in-out infinite;
}}
@keyframes studioPulse {{ 0%,100%{{opacity:1}} 50%{{opacity:.42}} }}
.studio-steps {{
  display:flex; gap:8px; flex-wrap:wrap; margin-top:16px;
  font-size:.79rem; color:var(--studio-muted);
}}
.studio-steps span {{
  padding:5px 11px; border-radius:8px;
  border:1px solid var(--studio-line); background:rgba(255,255,255,.03);
}}

/* ---------- パネル ---------- */
.studio-panel {{
  border:1px solid var(--studio-line) !important;
  border-radius:18px !important;
  background: linear-gradient(180deg, {PANEL_HI}, {PANEL}) !important;
  padding:18px !important;
  box-shadow: 0 10px 28px rgba(0,0,0,.30), inset 0 1px 0 rgba(255,255,255,.05);
}}
.step-head {{ display:flex; align-items:center; gap:11px; margin-bottom:4px; }}
.step-num {{
  width:26px; height:26px; flex:none; border-radius:9px;
  display:grid; place-items:center; font-size:.78rem; font-weight:700; color:#fff;
  background: linear-gradient(135deg, var(--studio-indigo), #312E81);
  box-shadow: 0 3px 10px rgba(79,70,229,.40);
}}
.step-title {{ font-size:.98rem; font-weight:650; color:var(--studio-text); letter-spacing:.01em; }}
.step-hint {{ margin:6px 0 2px; font-size:.79rem; color:var(--studio-muted); line-height:1.65; }}

/* ---------- ボタン ---------- */
.gradio-container button {{ cursor:pointer; transition: filter .2s ease, background-color .2s ease, border-color .2s ease; }}
button.ic-mic, button.ic-captions, button.ic-save, button.ic-folder,
button.ic-trash, button.ic-play, button.ic-speed, .ic-mic > button, .ic-captions > button,
.ic-save > button, .ic-folder > button, .ic-trash > button, .ic-play > button,
.ic-speed > button {{
  display:inline-flex !important; align-items:center; justify-content:center; gap:8px;
}}
button.ic-mic::before, button.ic-captions::before, button.ic-save::before,
button.ic-folder::before, button.ic-trash::before, button.ic-play::before,
button.ic-speed::before,
.ic-mic > button::before, .ic-captions > button::before, .ic-save > button::before,
.ic-folder > button::before, .ic-trash > button::before, .ic-play > button::before,
.ic-speed > button::before {{
  content:""; width:16px; height:16px; flex:none;
  background-color: currentColor;
  -webkit-mask-repeat:no-repeat; mask-repeat:no-repeat;
  -webkit-mask-position:center; mask-position:center;
  -webkit-mask-size:contain; mask-size:contain;
}}
{_ICON_RULES}
button.cta, .cta > button {{
  font-size:1rem !important; font-weight:700 !important; letter-spacing:.02em;
  padding:15px 22px !important; border-radius:14px !important;
  box-shadow: 0 10px 26px rgba(249,115,22,.30);
}}
button.cta:hover, .cta > button:hover {{ filter:brightness(1.06); }}
button.cta::before, .cta > button::before {{ width:15px; height:15px; }}

/* ---------- フォーム ---------- */
.gradio-container label span,
.gradio-container .block-title {{ font-weight:600 !important; letter-spacing:.01em; }}
.gradio-container textarea, .gradio-container input[type="text"] {{ line-height:1.75 !important; }}
.gradio-container :is(button, input, textarea, select, [tabindex]):focus-visible {{
  outline:2px solid var(--studio-accent) !important; outline-offset:2px !important;
}}
.gradio-container .accordion, .gradio-container details {{
  border:1px solid var(--studio-line) !important; border-radius:14px !important;
  background: rgba(255,255,255,.02) !important;
}}

/* 空状態のプレースホルダが縦に間延びしないよう抑える */
.slim .empty.large {{ min-height: 92px !important; }}
.slim .empty.small {{ min-height: 64px !important; }}

/* ---------- 生成結果 ---------- */
.result-card {{
  border:1px solid rgba(249,115,22,.28); border-radius:14px; padding:14px 16px;
  background: linear-gradient(180deg, rgba(249,115,22,.09), rgba(249,115,22,.03));
}}
.result-head {{ display:flex; align-items:center; gap:9px; font-weight:650; color:#FDBA74; font-size:.92rem; }}
.result-head::before {{
  content:""; width:17px; height:17px; flex:none; background-color:currentColor;
  -webkit-mask:url("{ICON_URIS["check"]}") center/contain no-repeat;
  mask:url("{ICON_URIS["check"]}") center/contain no-repeat;
}}
.result-metrics {{ display:flex; flex-wrap:wrap; gap:8px; margin-top:11px; }}
.result-metrics div {{
  border:1px solid var(--studio-line); border-radius:10px; padding:7px 11px;
  background: rgba(255,255,255,.03); min-width:82px;
}}
.result-metrics b {{ display:block; font-size:1.02rem; color:var(--studio-text); font-weight:650; }}
.result-metrics span {{ font-size:.71rem; color:var(--studio-muted); letter-spacing:.04em; }}
.result-path {{
  margin-top:11px; font-size:.74rem; color:var(--studio-muted);
  font-family:{", ".join(FONTS_MONO)}; word-break:break-all;
}}

footer, .built-with {{ display:none !important; }}

@media (max-width: 720px) {{
  .gradio-container {{ padding:14px 12px 40px !important; overflow-x:hidden; }}
  /* Gradio が列に入れる min-width:320px を外し、横スクロールを防ぐ */
  .gradio-container .column {{ min-width:0 !important; }}
  .studio-header {{ padding:18px; }}
  .studio-title {{ font-size:1.3rem; }}
  .studio-panel {{ padding:14px !important; }}
}}
@media (prefers-reduced-motion: reduce) {{
  .gradio-container *, .studio-dot {{ animation:none !important; transition:none !important; }}
}}
"""


HEADER_HTML = f"""
<div class="studio-header">
  <div class="studio-top">
    <div class="studio-brand">
      <div class="studio-mark">
        <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"
             stroke-linecap="round" stroke-linejoin="round" aria-hidden="true">
          <path d="M2 12h2"/><path d="M7 7v10"/><path d="M12 3v18"/>
          <path d="M17 8v8"/><path d="M22 11v2"/>
        </svg>
      </div>
      <div>
        <h1 class="studio-title">Voice Clone Studio</h1>
        <p class="studio-sub">Qwen3-TTS 12Hz 1.7B Base · 8bit</p>
      </div>
    </div>
    <div class="studio-chip"><span class="studio-dot"></span>MLX / Apple Silicon</div>
  </div>
  <div class="studio-steps">
    <span>1 — 参照音声をアップロード</span>
    <span>2 — 参照テキストを入力</span>
    <span>3 — 読み上げ文を入力</span>
    <span>4 — 生成</span>
  </div>
</div>
"""


def step_html(num: str, title: str, hint: str = "") -> str:
    hint_html = f'<p class="step-hint">{hint}</p>' if hint else ""
    return (
        f'<div class="step-head"><div class="step-num">{num}</div>'
        f'<div class="step-title">{title}</div></div>{hint_html}'
    )
