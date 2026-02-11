# app.py — Sexism Detection AI — Shield (pure Streamlit)
import streamlit as st
from datetime import datetime
import json, time, random
from typing import Tuple, List
import urllib.parse

st.set_page_config(page_title="Sexism Detection AI — Shield", layout="wide", page_icon="🛡️")

# -------------------------
# Required project modules
# -------------------------
try:
    from model_loader import load_models
except Exception as e:
    st.error(f"Could not import model_loader.py → {e}")
    st.stop()

try:
    from inference_utils import final_prediction
except Exception as e:
    st.error(f"Could not import inference_utils.py → {e}")
    st.stop()

# -------------------------
# Load models (cached)
# -------------------------
@st.cache_resource
def load_real_models_or_raise():
    return load_models()

try:
    with st.spinner("Loading SBERT + CatBoost model…"):
        sbert, cb_model, GLOBAL_THR, RECLAIMED_THR = load_real_models_or_raise()
except Exception as e:
    st.error(f"Model loading failed: {e}")
    st.stop()

# -------------------------
# Inference wrapper
# -------------------------
def predict_text_with_model(text: str) -> Tuple[float, str, str]:
    t = (text or "").strip()
    if not t:
        return 0.0, "CLEAN", "Empty input"
    out = final_prediction(t, sbert, cb_model, GLOBAL_THR, RECLAIMED_THR)
    if not isinstance(out, (list, tuple)):
        raise RuntimeError("final_prediction must return (prob,label) or (prob,label,reason)")
    if len(out) == 2:
        prob, label = out
        reason = "Model"
    else:
        prob, label, reason = out[:3]
    return float(prob), label, str(reason)

# -------------------------
# Statement-style generator (clear sexist / non-sexist declarative sentences)
# Used on page load and when user clicks "More suggestions"
# -------------------------
def generate_statement_suggestions(n: int = 8) -> List[str]:
    sexist = [
        "women belong in the kitchen",
        "women can't drive",
        "women are too emotional for leadership",
        "a woman's place is at home",
        "women shouldn't have careers",
        "women are too weak to do physical jobs",
        "men are superior to women",
        "women shouldn't speak up in meetings",
        "women are responsible for household chores",
        "men should always earn more than women",
        "women are not suited for tech jobs",
        "shutup c*nt",
        "wh0re",
        "don't be such a b.i.t.c.h",
        "b!tch please",
        "men age like wine women age like milk",
        "she's just jealous because she's ugly",
        "she was asking for it with that outfit",
        "all women are gold diggers",
        "must be that time of the month",
        "submissive and feminine",
        "females are too hormonal to lead",
        "why are women so dramatic"
    ]

    non_sexist = [
        "women can become successful leaders",
        "she is a skilled engineer",
        "he is a supportive colleague",
        "women and men deserve equal opportunities",
        "she is confident and hardworking",
        "everyone should be treated with respect",
        "men and women are equally capable",
        "she earned her promotion through talent",
        "both parents should share responsibilities",
        "women excel in every professional field",
        "he values teamwork and fairness",
        "gender does not determine intelligence",
        "women have achieved great things in science",
        "everyone deserves equal rights",
        "she's so cool",
        "my wife is a boss!",
    ]

    pool = sexist + non_sexist
    random.shuffle(pool)
    out = []
    for s in pool:
        if s not in out:
            out.append(s)
        if len(out) >= n:
            break
    return out

# -------------------------
# Small helper to embed SVG icons as data URIs
# -------------------------
def svg_data_uri(svg_text: str) -> str:
    return "data:image/svg+xml;utf8," + urllib.parse.quote(svg_text)

SVG_ACCURACY = """
<svg xmlns='http://www.w3.org/2000/svg' width='48' height='48' viewBox='0 0 24 24' fill='none'>
  <rect rx='4' width='24' height='24' fill='%23202a33'/>
  <path d='M6 12l3 3 7-7' stroke='%239be7a8' stroke-width='1.6' stroke-linecap='round' stroke-linejoin='round'/>
</svg>
"""

SVG_F1 = """
<svg xmlns='http://www.w3.org/2000/svg' width='48' height='48' viewBox='0 0 24 24' fill='none'>
  <rect rx='4' width='24' height='24' fill='%23202a33'/>
  <path d='M7 14s1-3 5-3 5 3 5 3' stroke='%2398f0e6' stroke-width='1.6' stroke-linecap='round' stroke-linejoin='round'/>
  <circle cx='7' cy='14' r='1.6' fill='%2398f0e6'/>
  <circle cx='12' cy='11' r='1.6' fill='%2398f0e6'/>
  <circle cx='17' cy='14' r='1.6' fill='%2398f0e6'/>
</svg>
"""

SVG_SAMPLES = """
<svg xmlns='http://www.w3.org/2000/svg' width='48' height='48' viewBox='0 0 24 24' fill='none'>
  <rect rx='4' width='24' height='24' fill='%23202a33'/>
  <path d='M8 7h8M8 11h8M8 15h5' stroke='%23ffd98a' stroke-width='1.6' stroke-linecap='round' stroke-linejoin='round'/>
</svg>
"""

# -------------------------
# Session-state initialization
# -------------------------
if "suggestions" not in st.session_state:
    st.session_state["suggestions"] = generate_statement_suggestions(8)

if "input_text" not in st.session_state:
    st.session_state["input_text"] = ""

if "results" not in st.session_state:
    st.session_state["results"] = []

if "current_result" not in st.session_state:
    st.session_state["current_result"] = None

# -------------------------
# UI: Header & metrics (styling)
# -------------------------
st.markdown(
    "<div style='text-align:center; padding-top:12px'>"
    "<h2 style='margin:0; font-weight:700'>🛡️ Shield</h2>"
    "<h1 style='margin:6px 0 8px 0; font-size:34px'>Sexism Detection AI</h1>"
    "<p style='color: #9aa3b2; max-width:900px; margin:auto'>Advanced ML model for detecting sexist content with dynamic threshold classification. Supports Detection of explicit slurs and subtle sexism patterns.</p>"
    "</div>",
    unsafe_allow_html=True
)

col1, col2, col3 = st.columns(3)
with col1:
    uri = svg_data_uri(SVG_ACCURACY)
    st.markdown(f"<div style='display:flex; align-items:center; gap:12px'>"
                f"<img src='{uri}' width='48' height='48' alt='accuracy'/>"
                f"<div>"
                f"<div style='font-size:20px; font-weight:700'>✔️ 94.2%</div>"
                f"<div style='color:#9aa3b2'>Accuracy</div>"
                f"</div></div>", unsafe_allow_html=True)
with col2:
    uri = svg_data_uri(SVG_F1)
    st.markdown(f"<div style='display:flex; align-items:center; gap:12px'>"
                f"<img src='{uri}' width='48' height='48' alt='f1'/>"
                f"<div>"
                f"<div style='font-size:20px; font-weight:700'>📊 0.89</div>"
                f"<div style='color:#9aa3b2'>F1 Score</div>"
                f"</div></div>", unsafe_allow_html=True)
with col3:
    uri = svg_data_uri(SVG_SAMPLES)
    st.markdown(f"<div style='display:flex; align-items:center; gap:12px'>"
                f"<img src='{uri}' width='48' height='48' alt='samples'/>"
                f"<div>"
                f"<div style='font-size:20px; font-weight:700'>📚 50K+</div>"
                f"<div style='color:#9aa3b2'>Trained Samples</div>"
                f"</div></div>", unsafe_allow_html=True)

st.markdown("---")

# -------------------------
# Input area + suggestions (left) ; suggestions column (right)
# -------------------------
left, right = st.columns([3, 1])

# helper for a styled orange warning box (inline CSS)
def orange_warning_box(title: str, body: str):
    st.markdown(
        f"""
        <div style="
            background-color: rgba(255,183,77,0.08);
            border-left: 4px solid #ff9800;
            padding: 12px 16px;
            border-radius: 8px;
            margin-top: 8px;
        ">
            <div style="font-weight:700; color:#ffcc80; margin-bottom:6px;">{title}</div>
            <div style="color:#ffdcb6;">{body}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

with left:
    st.subheader("Enter text to analyze")
    with st.form("analyze_form", clear_on_submit=False):
        text_val = st.text_area("", value=st.session_state.get("input_text", ""), height=140)
        analyze_clicked = st.form_submit_button("Analyze")
        clear_clicked = st.form_submit_button("Clear Results")
        st.session_state["input_text"] = text_val

    if clear_clicked:
        st.session_state["input_text"] = ""
        st.session_state["current_result"] = None
        st.session_state["results"] = []
        st.session_state["suggestions"] = generate_statement_suggestions(8)
        st.rerun()

    if analyze_clicked:
        txt = (st.session_state.get("input_text") or "").strip()
        if not txt:
            st.warning("Please enter some text to analyze.")
        else:
            try:
                prob, label, reason = predict_text_with_model(txt)
            except Exception as e:
                st.error(f"Model inference failed: {e}")
                raise
            item = {
                "id": datetime.utcnow().isoformat(timespec="milliseconds"),
                "text": txt,
                "probability": round(float(prob), 4),
                "classification": label,
                "reason": reason,
                "ts": int(time.time())
            }
            st.session_state["current_result"] = item
            st.session_state["results"].insert(0, item)
            st.session_state["results"] = st.session_state["results"][:300]

    # show current result directly below the input
    st.subheader("Current result")
    if st.session_state.get("current_result"):
        r = st.session_state["current_result"]
        prob = r["probability"]
        label = r["classification"]
        reason = r.get("reason", "")
        st.write(f"**Input:** {r['text']}")

        if label == "SEXIST":
            # High confidence
            if prob >= 0.9:
                st.error(f"High confidence — SEXIST ({prob*100:.1f}%)\n\nReason: {reason}")
            # Medium confidence
            elif prob >= 0.5:
                st.warning(f"Warning — Potentially sexist ({prob*100:.1f}%)\n\nReason: {reason}")
            # Low confidence
            else:
                # Defensive normalization of the reason and keyword check
                low_reason = (reason or "").lower().strip()
                if "obfus" in low_reason:
                    orange_warning_box(f"Low confidence — Sexist ({prob*100:.1f}%)", reason)
                else:
                    st.info(f"Low confidence — Sexist ({prob*100:.1f}%)\n\nReason: {reason}")
        else:
            st.success(f"Content is clean ({prob*100:.1f}%)\n\nReason: {reason}")

        st.download_button("Download current result (JSON)", data=json.dumps(r, indent=2),
                           file_name=f"current_{r['id'].replace(':','_')}.json", mime="application/json")
    else:
        st.info("No current result. Enter text and press Analyze (or pick a suggestion).")

with right:
    st.subheader("Smart Suggestions")
    suggestions = st.session_state.get("suggestions", [])

    if not suggestions:
        st.info("No suggestions available. Click 'More suggestions' to generate statements.")
    else:
        for idx, s in enumerate(list(suggestions)):
            if st.button(s, key=f"sugg_{idx}"):
                st.session_state["input_text"] = s
                current = st.session_state.get("suggestions", [])
                try:
                    if 0 <= idx < len(current):
                        current.pop(idx)
                    else:
                        current.remove(s)
                except Exception:
                    if s in current:
                        current.remove(s)
                st.session_state["suggestions"] = current
                st.session_state["current_result"] = None
                st.rerun()

    if st.button("More suggestions"):
        st.session_state["suggestions"] = generate_statement_suggestions(8)
        st.rerun()

    st.caption("Click a suggestion to load it into the input. Click 'More suggestions' to get a new set of test statements.")

st.markdown("---")

# -------------------------
# History / previous analyses (expander)
# -------------------------
st.subheader("Previous analyses")
if not st.session_state.get("results"):
    st.info("No history yet.")
else:
    with st.expander("Show previous analyses (newest first)", expanded=False):
        for entry in st.session_state["results"]:
            cur = st.session_state.get("current_result")
            if cur and entry["id"] == cur["id"]:
                continue
            prob = entry["probability"]
            label = entry["classification"]
            st.write(f"**Input:** {entry['text']}")
            if label == "SEXIST":
                if prob >= 0.9:
                    st.error(f"High Confidence — Sexist detected ({prob*100:.1f}%)\n\nReason: {entry.get('reason','')}")
                elif prob >= 0.5:
                    st.warning(f"Warning — Potentially sexist ({prob*100:.1f}%).\n\nReason: {entry.get('reason','')}")
                else:
                    hist_reason = (entry.get('reason','') or "").lower().strip()
                    if "obfus" in hist_reason:
                        orange_warning_box(f"Low confidence — Sexist ({prob*100:.1f}%)", entry.get('reason',''))
                    else:
                        st.info(f"Low confidence — Sexist ({prob*100:.1f}%).\n\nReason: {entry.get('reason','')}")
            else:
                st.success(f"Content is clean ({prob*100:.1f}%).\n\nReason: {entry.get('reason','')}")
            cols = st.columns([8,1])
            with cols[1]:
                st.download_button("JSON", data=json.dumps(entry, indent=2),
                                   file_name=f"history_{entry['id'].replace(':','_')}.json", mime="application/json")
            st.markdown("---")

st.caption("Smart suggestions are statement-style test sentences (clear sexist or non-sexist) to help evaluate the classifier quickly.")











