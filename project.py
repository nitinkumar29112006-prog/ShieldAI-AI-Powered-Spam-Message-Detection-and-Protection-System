from pathlib import Path
import joblib
import streamlit as st

ROOT = Path(__file__).parent
MODEL_PATH = ROOT / "models" / "spam_pipeline.joblib"

st.set_page_config(
    page_title="ShieldAI | Message Security",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ---------- Visual design ----------
st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&family=Space+Grotesk:wght@500;600;700&display=swap');

    :root {
        --bg: #0b1020;
        --panel: #121a2d;
        --panel-2: #172238;
        --line: rgba(148, 163, 184, .18);
        --muted: #9aa9c2;
        --text: #edf3ff;
        --blue: #65a7ff;
        --teal: #4de0c1;
    }
    html, body, [class*="css"] {
        font-family: 'DM Sans', sans-serif;
    }
    .stApp {
        background:
            radial-gradient(ellipse at 12% 0%, rgba(38, 91, 176, .20), transparent 38%),
            radial-gradient(ellipse at 92% 8%, rgba(40, 188, 170, .10), transparent 28%),
            var(--bg);
        color: var(--text);
    }
    .block-container {
        max-width: 1440px;
        padding-top: 2rem;
        padding-bottom: 3rem;
    }
    h1, h2, h3, .brand-title { font-family: 'Space Grotesk', sans-serif !important; }
    .hero {
        border: 1px solid var(--line);
        border-radius: 24px;
        padding: 28px 30px;
        background: linear-gradient(125deg, rgba(22, 37, 65, .96), rgba(14, 25, 44, .92));
        box-shadow: 0 18px 55px rgba(0,0,0,.18);
        margin-bottom: 22px;
        position: relative;
        overflow: hidden;
    }
    .hero:after {
        content: "";
        position: absolute;
        width: 220px; height: 220px;
        right: -70px; top: -100px;
        border-radius: 50%;
        background: rgba(77, 224, 193, .08);
        border: 1px solid rgba(77, 224, 193, .16);
    }
    .eyebrow {
        color: var(--teal); font-size: .76rem; font-weight: 700;
        letter-spacing: .15em; text-transform: uppercase;
    }
    .brand-title {
        font-size: clamp(2rem, 3.4vw, 3.15rem);
        font-weight: 700; letter-spacing: -.045em; line-height: 1.1;
        margin: 10px 0 12px; color: #f5f8ff;
    }
    .hero-copy { color: #b8c7df; font-size: 1.02rem; max-width: 700px; }
    .pill {
        display: inline-block; border: 1px solid rgba(77,224,193,.3);
        background: rgba(77,224,193,.08); color: #7bf0d6;
        padding: 6px 11px; border-radius: 99px; font-size: .78rem;
        font-weight: 600; margin-bottom: 7px;
    }
    .section-label {
        color: #b7c6df; text-transform: uppercase; letter-spacing: .12em;
        font-size: .74rem; font-weight: 700; margin: 4px 0 12px;
    }
    .panel {
        background: linear-gradient(160deg, rgba(20,31,52,.96), rgba(15,24,42,.96));
        border: 1px solid var(--line); border-radius: 20px;
        padding: 22px; height: 100%;
    }
    .panel h3 { margin-top: 0; color: #f2f6ff; font-size: 1.22rem; }
    .subtle { color: var(--muted); font-size: .91rem; line-height: 1.55; }
    .mini-card {
        background: rgba(20,31,52,.85); border: 1px solid var(--line);
        border-radius: 16px; padding: 16px 18px; min-height: 95px;
    }
    .mini-label { color: var(--muted); font-size: .78rem; margin-bottom: 8px; }
    .mini-value { color: #f3f7ff; font-family: 'Space Grotesk',sans-serif;
        font-size: 1.45rem; font-weight: 700; }
    .result {
        border-radius: 16px; padding: 18px 20px; border: 1px solid;
        margin: 16px 0 12px;
    }
    .result-spam { background: rgba(248, 87, 107, .10); border-color: rgba(248,87,107,.36); }
    .result-ham { background: rgba(77, 224, 193, .08); border-color: rgba(77,224,193,.3); }
    .result-uncertain { background: rgba(255, 190, 92, .09); border-color: rgba(255,190,92,.35); }
    .result-title { font-weight: 700; font-size: 1.12rem; margin-bottom: 4px; }
    .stTextArea textarea {
        background: #0d1628 !important; color: #eff5ff !important;
        border: 1px solid rgba(148,163,184,.24) !important;
        border-radius: 14px !important; line-height: 1.6 !important;
    }
    .stTextArea textarea:focus {
        border-color: var(--blue) !important;
        box-shadow: 0 0 0 1px var(--blue) !important;
    }
    .stButton > button[kind="primary"] {
        background: linear-gradient(90deg, #367df5, #4c9cff);
        color: white; border: 0; border-radius: 12px; min-height: 3rem;
        font-weight: 700; box-shadow: 0 8px 22px rgba(54,125,245,.20);
    }
    .stButton > button[kind="primary"]:hover {
        border: 0; filter: brightness(1.08); transform: translateY(-1px);
    }
    div[data-testid="stMetric"] {
        background: rgba(13,22,40,.72); border: 1px solid var(--line);
        padding: 14px 16px; border-radius: 14px;
    }
    div[data-testid="stMetricLabel"] { color: var(--muted); }
    .footer {
        color: #7f8da5; font-size: .78rem; text-align: center;
        padding-top: 22px; margin-top: 26px; border-top: 1px solid var(--line);
    }
    [data-testid="stAlert"] { border-radius: 12px; }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_resource
def load_artifact():
    if not MODEL_PATH.exists():
        return None
    return joblib.load(MODEL_PATH)


artifact = load_artifact()

# Supports both upgraded artifact dictionary and earlier direct sklearn pipeline.
pipeline = None
threshold = 0.5
min_words = 3
model_name = "Existing trained classifier"

if isinstance(artifact, dict) and "pipeline" in artifact:
    pipeline = artifact["pipeline"]
    threshold = float(artifact.get("spam_threshold", 0.5))
    min_words = int(artifact.get("minimum_words", 3))
    model_name = artifact.get("model_name", "Trained classifier")
elif artifact is not None:
    pipeline = artifact

# ---------- Header ----------
st.markdown(
    """
    <div class="hero">
      <div class="pill">●&nbsp; ML-POWERED MESSAGE SCREENING</div>
      <div class="brand-title">ShieldAI <span style="color:#65a7ff">Message Security</span></div>
      <div class="hero-copy">
        A clean, fast first-pass screen for suspicious SMS messages.
        Review the model's signal before you decide what to do next.
      </div>
    </div>
    """,
    unsafe_allow_html=True,
)

# Quick info cards
a, b, c = st.columns(3)
with a:
    st.markdown(f'<div class="mini-card"><div class="mini-label">MODEL</div><div class="mini-value">{model_name}</div></div>', unsafe_allow_html=True)
with b:
    st.markdown(f'<div class="mini-card"><div class="mini-label">DECISION THRESHOLD</div><div class="mini-value">{threshold:.3f}</div></div>', unsafe_allow_html=True)
with c:
    st.markdown('<div class="mini-card"><div class="mini-label">INPUT TYPE</div><div class="mini-value">SMS / Short text</div></div>', unsafe_allow_html=True)

st.write("")
st.markdown('<div class="section-label">MESSAGE ANALYSIS WORKSPACE</div>', unsafe_allow_html=True)

if pipeline is None:
    st.error("The trained model was not found.")
    st.code("python train_model.py\npython -m streamlit run app.py")
    st.stop()

left, right = st.columns([1.35, 0.85], gap="large")

with left:
    st.markdown('<div class="panel"><h3>Inspect a message</h3><p class="subtle">Paste the complete SMS below. Avoid including real passwords, OTPs, or other private information.</p></div>', unsafe_allow_html=True)
    message = st.text_area(
        "Message content",
        height=190,
        placeholder="Example: Your account needs verification. Review this message carefully...",
        label_visibility="collapsed",
        key="message_input",
    )
    word_count = len(message.split())
    st.caption(f"{len(message)} characters  ·  {word_count} words")
    analyze = st.button("Analyze message  →", type="primary", use_container_width=True)

with right:
    st.markdown(
        """
        <div class="panel">
          <h3>How the screening works</h3>
          <p class="subtle">01 &nbsp; Text is transformed into TF-IDF features.</p>
          <p class="subtle">02 &nbsp; The trained classifier estimates spam likelihood.</p>
          <p class="subtle">03 &nbsp; The result is shown with caution indicators.</p>
          <hr style="border-color:rgba(148,163,184,.16); margin:18px 0">
          <div class="section-label">SAFETY NOTE</div>
          <p class="subtle">A message can be dangerous even if it is not flagged. Never trust a link or sender solely because of this result.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

if analyze:
    cleaned = " ".join(message.split())
    if not cleaned:
        st.warning("Enter a message before analyzing it.")
    elif len(cleaned.split()) < min_words:
        st.markdown(
            '<div class="result result-uncertain"><div class="result-title">⚠️ Not enough context</div><div class="subtle">Paste the complete message. Very short text is difficult to classify reliably.</div></div>',
            unsafe_allow_html=True,
        )
    else:
        spam_score = None
        if hasattr(pipeline, "predict_proba"):
            classes = [str(label).lower() for label in pipeline.classes_]
            probabilities = pipeline.predict_proba([cleaned])[0]
            if "spam" in classes:
                spam_score = float(probabilities[classes.index("spam")])
                prediction = "spam" if spam_score >= threshold else "ham"
            else:
                prediction = str(pipeline.predict([cleaned])[0]).lower()
        else:
            prediction = str(pipeline.predict([cleaned])[0]).lower()

        uncertain = spam_score is not None and abs(spam_score - threshold) < 0.08
        if uncertain:
            st.markdown(
                '<div class="result result-uncertain"><div class="result-title">⚠️ Uncertain result</div><div class="subtle">The score is close to the decision threshold. Verify the sender and message independently.</div></div>',
                unsafe_allow_html=True,
            )
        elif prediction == "spam":
            st.markdown(
                '<div class="result result-spam"><div class="result-title">🚨 Likely spam</div><div class="subtle">The model flagged this message. This is a warning, not proof of fraud.</div></div>',
                unsafe_allow_html=True,
            )
        else:
            st.markdown(
                '<div class="result result-ham"><div class="result-title">✓ Not flagged as spam</div><div class="subtle">The model did not flag this text. It is not a guarantee that the message is legitimate.</div></div>',
                unsafe_allow_html=True,
            )

        if spam_score is not None:
            st.markdown("#### Model score")
            score_col, note_col = st.columns([1, 2])
            with score_col:
                st.metric("Estimated spam score", f"{spam_score:.1%}")
            with note_col:
                st.caption("This is a model score and may not be a calibrated probability.")
                st.progress(min(1.0, max(0.0, spam_score)))

        indicators = [
            "urgent", "winner", "won", "claim", "prize", "free", "click",
            "verify", "password", "account", "cash", "offer", "limited time",
            "payment details", "invoice", "sign in", "gift card", "suspended",
            "restore access", "otp",
        ]
        found = [term for term in indicators if term in cleaned.lower()]
        with st.expander("Review text indicators"):
            if found:
                st.write("Terms to review:", ", ".join(f"`{term}`" for term in found))
            else:
                st.write("No terms from the simple keyword hint list were found.")
            st.caption("Keyword hints are not the model's explanation. Legitimate messages can contain these terms.")

        with st.expander("View analyzed message"):
            st.write(cleaned)

st.markdown(
    """
    <div class="footer">
      SHIELDAI · Academic machine-learning prototype · TF-IDF + supervised classifier<br>
      Trained on SMS data; performance on full-length emails or new scam patterns is not established.
    </div>
    """,
    unsafe_allow_html=True,
)
