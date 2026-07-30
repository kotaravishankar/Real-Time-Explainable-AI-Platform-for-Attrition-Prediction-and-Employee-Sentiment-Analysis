"""
Real-Time Explainable Deep Learning Platform
Employee Attrition Prediction  +  Sentiment Analysis
Gradio app for Hugging Face Spaces.

Attrition : FT-Transformer / Tabular ResNet  (PyTorch)
Sentiment : BiLSTM / CNN-BiGRU-Attention     (TensorFlow / Keras)
"""

import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

import re
import pickle
import joblib
import numpy as np
import gradio as gr
import spaces

import torch
import torch.nn as nn

import tensorflow as tf
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.layers import (
    Layer, Input, Embedding, SpatialDropout1D, Bidirectional, LSTM, GRU,
    Conv1D, GlobalMaxPooling1D, Dense, BatchNormalization, Dropout,
)
import tensorflow.keras.backend as K

import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# --------------------------------------------------------------------------- #
# 0. NLTK resources (downloaded on first launch)
# --------------------------------------------------------------------------- #
for pkg in ["stopwords", "wordnet", "omw-1.4"]:
    try:
        nltk.data.find(f"corpora/{pkg}")
    except LookupError:
        nltk.download(pkg, quiet=True)

DEVICE = torch.device("cpu")

# =========================================================================== #
#  PART A — ATTRITION (PyTorch)
# =========================================================================== #

FEATURE_ORDER = [
    "city", "city_development_index", "gender", "relevent_experience",
    "enrolled_university", "education_level", "major_discipline",
    "experience", "company_size", "company_type", "last_new_job",
    "training_hours",
]
CATEGORICAL = [
    "city", "gender", "relevent_experience", "enrolled_university",
    "education_level", "major_discipline", "experience",
    "company_size", "company_type", "last_new_job",
]

label_encoders = joblib.load("attrition_label_encoders.pkl")
attr_scaler    = joblib.load("attrition_scaler.pkl")


# ---- model definitions (must match training exactly) ---------------------- #
class FeatureTokenizer(nn.Module):
    def __init__(self, num_features, d_token):
        super().__init__()
        self.weight = nn.Parameter(torch.empty(num_features, d_token))
        self.bias   = nn.Parameter(torch.empty(num_features, d_token))

    def forward(self, x):
        return x.unsqueeze(-1) * self.weight + self.bias


class TransformerBlock(nn.Module):
    def __init__(self, d_token, n_heads, dropout=0.2, ff_factor=4):
        super().__init__()
        self.attention = nn.MultiheadAttention(
            d_token, n_heads, dropout=dropout, batch_first=True)
        self.norm1 = nn.LayerNorm(d_token)
        self.norm2 = nn.LayerNorm(d_token)
        self.ffn = nn.Sequential(
            nn.Linear(d_token, d_token * ff_factor), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(d_token * ff_factor, d_token), nn.Dropout(dropout),
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        a, w = self.attention(x, x, x, need_weights=True, average_attn_weights=True)
        x = self.norm1(x + self.dropout(a))
        x = self.norm2(x + self.ffn(x))
        return x, w


class AttentionPooling(nn.Module):
    def __init__(self, d_token):
        super().__init__()
        self.attention_vector = nn.Linear(d_token, 1)

    def forward(self, x):
        w = torch.softmax(self.attention_vector(x), dim=1)
        return torch.sum(w * x, dim=1), w


class AdvancedFTTransformer(nn.Module):
    def __init__(self, num_features, d_token=64, n_heads=8, n_layers=4, dropout=0.3):
        super().__init__()
        self.tokenizer = FeatureTokenizer(num_features, d_token)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, d_token))
        self.transformer_blocks = nn.ModuleList(
            [TransformerBlock(d_token, n_heads, dropout) for _ in range(n_layers)])
        self.pooling = AttentionPooling(d_token)
        self.classifier = nn.Sequential(
            nn.Linear(d_token, 256), nn.BatchNorm1d(256), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(256, 128),     nn.BatchNorm1d(128), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(128, 64),      nn.BatchNorm1d(64),  nn.GELU(), nn.Dropout(dropout),
            nn.Linear(64, 1),
        )

    def forward(self, x):
        b = x.shape[0]
        tokens = self.tokenizer(x)
        cls = self.cls_token.expand(b, -1, -1)
        x = torch.cat([cls, tokens], dim=1)
        for blk in self.transformer_blocks:
            x, _ = blk(x)
        pooled, _ = self.pooling(x)
        return self.classifier(pooled)


class ResNetBlock(nn.Module):
    def __init__(self, d_main, d_hidden, dropout):
        super().__init__()
        self.norm    = nn.BatchNorm1d(d_main)
        self.linear0 = nn.Linear(d_main, d_hidden)
        self.linear1 = nn.Linear(d_hidden, d_main)
        self.act     = nn.GELU()
        self.drop0   = nn.Dropout(dropout)
        self.drop1   = nn.Dropout(dropout)

    def forward(self, x):
        z = self.norm(x)
        z = self.drop0(self.act(self.linear0(z)))
        z = self.drop1(self.linear1(z))
        return x + z


class TabularResNet(nn.Module):
    def __init__(self, num_features, d_main=128, d_hidden=256, n_blocks=4, dropout=0.3):
        super().__init__()
        self.input_proj = nn.Linear(num_features, d_main)
        self.blocks = nn.ModuleList(
            [ResNetBlock(d_main, d_hidden, dropout) for _ in range(n_blocks)])
        self.head = nn.Sequential(
            nn.BatchNorm1d(d_main), nn.GELU(), nn.Linear(d_main, 1))

    def forward(self, x):
        x = self.input_proj(x)
        for blk in self.blocks:
            x = blk(x)
        return self.head(x)


def _load_torch(path, cls):
    ck = torch.load(path, map_location=DEVICE, weights_only=False)
    model = cls(**ck["model_config"])
    model.load_state_dict(ck["model_state_dict"], strict=True)
    model.eval().to(DEVICE)
    return model


ft_model     = _load_torch("ft_transformer_attrition.pth", AdvancedFTTransformer)
resnet_model = _load_torch("tabular_resnet_attrition.pth",  TabularResNet)
ATTR_MODELS  = {"FT-Transformer": ft_model, "Tabular ResNet": resnet_model}


PRETTY = {
    "city": "City",
    "city_development_index": "City Development Index",
    "gender": "Gender",
    "relevent_experience": "Relevant Experience",
    "enrolled_university": "University Enrolment",
    "education_level": "Education Level",
    "major_discipline": "Major Discipline",
    "experience": "Years of Experience",
    "company_size": "Company Size",
    "company_type": "Company Type",
    "last_new_job": "Years Since Last Job Change",
    "training_hours": "Training Hours",
}


def _risk_band(p):
    if p >= 0.70:
        return "HIGH RISK", "#b91c1c"
    if p >= 0.50:
        return "ELEVATED RISK", "#c2410c"
    if p >= 0.30:
        return "MODERATE RISK", "#a16207"
    return "LOW RISK", "#15803d"


def _card(prob, base, model_name):
    """Headline verdict card."""
    stay = prob < 0.5
    verdict = "LIKELY TO STAY WITH THE COMPANY" if stay else "LIKELY TO LEAVE THE COMPANY"
    sub = ("below" if stay else "above") + " the current decision threshold"
    conf = (1 - prob) if stay else prob
    band, band_col = _risk_band(prob)
    delta = (prob - base) * 100
    accent = "#15803d" if stay else "#b91c1c"
    bg = "#f0fdf4" if stay else "#fef2f2"

    def stat(v, lab, col="#111827"):
        return (f"<div style='flex:1;min-width:110px'>"
                f"<div style='font-size:1.45rem;font-weight:700;color:{col}'>{v}</div>"
                f"<div style='font-size:.66rem;letter-spacing:.06em;color:#6b7280'>{lab}</div></div>")

    return f"""
<div style="border:1px solid {accent}33;border-left:5px solid {accent};background:{bg};
            border-radius:10px;padding:14px 16px;font-family:system-ui,sans-serif">
  <div style="font-weight:700;color:{accent};letter-spacing:.02em">{verdict}</div>
  <div style="font-size:.82rem;color:#4b5563;margin:2px 0 12px">
      The model places this employee {sub}.</div>
  <div style="display:flex;gap:14px;flex-wrap:wrap">
    {stat(f"{conf:.1%}", "CONFIDENCE")}
    {stat(f"{prob:.1%}", "PROBABILITY OF LEAVING")}
    {stat(f"{delta:+.1f} pts", "VS AVERAGE EMPLOYEE")}
    {stat(band, "RISK RATING", band_col)}
  </div>
  <div style="font-size:.68rem;color:#6b7280;margin-top:10px">
      Decision threshold 50.0%. Average employee in the training data scores
      {base:.1%}. Model: {model_name}.</div>
</div>"""


def _tables(rows):
    """Two side-by-side tables: features pushing each way."""
    leave = [r for r in rows if r[2] > 0]
    stay  = [r for r in rows if r[2] < 0]
    top = max([abs(r[2]) for r in rows] + [1e-9])

    def block(items, title, col, bg):
        if not items:
            body = ("<tr><td colspan='3' style='padding:10px;color:#9ca3af;"
                    "font-size:.78rem'>No features push this way.</td></tr>")
        else:
            body = ""
            for name, val, c in items:
                w = min(abs(c) / top * 100, 100)
                body += f"""<tr style="border-top:1px solid #e5e7eb">
                  <td style="padding:6px 8px;font-size:.78rem">{name}</td>
                  <td style="padding:6px 8px;font-size:.78rem;color:#4b5563">{val}</td>
                  <td style="padding:6px 8px;width:120px">
                    <div style="display:flex;align-items:center;gap:6px">
                      <div style="height:7px;width:{w}%;background:{col};border-radius:4px"></div>
                      <span style="font-size:.66rem;color:#6b7280">{abs(c)*100:.2f}</span>
                    </div></td></tr>"""
        return f"""
        <div style="flex:1;min-width:260px;border:1px solid {col}33;border-radius:9px;overflow:hidden">
          <div style="background:{bg};color:{col};font-weight:700;font-size:.74rem;
                      padding:7px 9px;letter-spacing:.04em">{title}</div>
          <table style="width:100%;border-collapse:collapse;background:#fff">
            <tr style="background:#f9fafb">
              <th style="text-align:left;padding:5px 8px;font-size:.62rem;color:#6b7280">FEATURE</th>
              <th style="text-align:left;padding:5px 8px;font-size:.62rem;color:#6b7280">VALUE</th>
              <th style="text-align:left;padding:5px 8px;font-size:.62rem;color:#6b7280">INFLUENCE</th>
            </tr>{body}
          </table></div>"""

    return f"""
<div style="font-family:system-ui,sans-serif">
  <div style="display:flex;gap:12px;flex-wrap:wrap">
    {block(leave, "PUSHING TOWARDS LEAVING", "#b91c1c", "#fee2e2")}
    {block(stay,  "PUSHING TOWARDS STAYING", "#15803d", "#dcfce7")}
  </div>
  <div style="font-size:.66rem;color:#9ca3af;margin-top:8px">
    Influence is measured by masking one feature at a time (resetting it to the training
    average) and recording how far the model's output moves. Larger bars mean the feature
    moved the decision further. Values are relative to this employee only.</div>
</div>"""


@spaces.GPU(duration=30)
def predict_attrition(model_name, city, city_dev, gender, rel_exp, enrolled,
                      edu, major, exp, comp_size, comp_type, last_job, train_hours):
    raw = {
        "city": city, "city_development_index": float(city_dev), "gender": gender,
        "relevent_experience": rel_exp, "enrolled_university": enrolled,
        "education_level": edu, "major_discipline": major, "experience": exp,
        "company_size": comp_size, "company_type": comp_type,
        "last_new_job": last_job, "training_hours": float(train_hours),
    }
    row = []
    for feat in FEATURE_ORDER:
        val = raw[feat]
        if feat in CATEGORICAL:
            val = int(label_encoders[feat].transform([str(val)])[0])
        row.append(float(val))

    x = np.array(row, dtype=np.float32).reshape(1, -1)
    xs = attr_scaler.transform(x).astype(np.float32)      # (1, 12) standardised

    n = len(FEATURE_ORDER)
    # row 0 = employee, rows 1..n = employee with feature i masked to the mean (0 when
    # standardised), row n+1 = the "average employee" (every feature at the mean).
    batch = np.repeat(xs, n + 2, axis=0)
    for i in range(n):
        batch[i + 1, i] = 0.0
    batch[n + 1, :] = 0.0

    # CUDA is only touched inside this decorated function, never at import time
    # (ZeroGPU forbids initialising CUDA in the main process).
    model = ATTR_MODELS[model_name]
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(dev)
    t = torch.tensor(batch, dtype=torch.float32, device=dev)

    with torch.no_grad():
        probs = torch.sigmoid(model(t)).squeeze(-1).cpu().numpy()

    prob = float(probs[0])
    base = float(probs[n + 1])

    rows = []
    for i, feat in enumerate(FEATURE_ORDER):
        contrib = prob - float(probs[i + 1])   # >0 pushes towards leaving
        shown = raw[feat]
        if isinstance(shown, float):
            shown = f"{shown:g}"
        rows.append((PRETTY[feat], str(shown), contrib))
    rows.sort(key=lambda r: abs(r[2]), reverse=True)

    return ({"Attrition (leave)": prob, "Retention (stay)": 1 - prob},
            _card(prob, base, model_name),
            _tables(rows))


# =========================================================================== #
#  PART B — SENTIMENT (TensorFlow / Keras)
# =========================================================================== #

MAX_LEN = 80

with open("sentiment_tokenizer.pkl", "rb") as f:
    sent_tokenizer = pickle.load(f)
with open("sentiment_label_map.pkl", "rb") as f:
    label_map = pickle.load(f)          # {0:'Negative', 1:'Positive'}

_lem  = WordNetLemmatizer()
_sw   = set(stopwords.words("english"))
_URL  = re.compile(r"http\S+|www\.\S+")
_EMO  = re.compile("[\U0001F000-\U0001FFFF\U00002600-\U000027BF]", flags=re.UNICODE)


def clean_text(t):
    t = str(t).lower()
    t = _URL.sub(" ", t)
    t = _EMO.sub(" ", t)
    t = re.sub(r"[^a-z\s]", " ", t)
    return " ".join(_lem.lemmatize(w) for w in t.split()
                    if w not in _sw and len(w) > 1)


class AttentionPool(Layer):
    """Additive attention pooling used by the CNN-BiGRU model."""
    def build(self, input_shape):
        d = int(input_shape[-1])
        self.W = self.add_weight(name="att_W", shape=(d, d),
                                 initializer="glorot_uniform", trainable=True)
        self.b = self.add_weight(name="att_b", shape=(d,),
                                 initializer="zeros", trainable=True)
        self.u = self.add_weight(name="att_u", shape=(d, 1),
                                 initializer="glorot_uniform", trainable=True)
        super().build(input_shape)

    def call(self, x):
        score = K.tanh(K.dot(x, self.W) + self.b)
        score = K.dot(score, self.u)
        weights = K.softmax(score, axis=1)
        return K.sum(weights * x, axis=1)


# The .keras files were saved with Keras 3.13.2. Rebuilding the architectures here
# and loading weights only keeps the app independent of the Keras version installed
# on the Space (full load_model would fail on config deserialisation).
VOCAB, EMBED_DIM = 20000, 128


def build_bilstm():
    m = Sequential([
        Embedding(input_dim=VOCAB, output_dim=EMBED_DIM),
        SpatialDropout1D(0.3),
        Bidirectional(LSTM(64, return_sequences=True)),
        Bidirectional(LSTM(32, return_sequences=True)),
        GlobalMaxPooling1D(),
        Dense(64, activation="relu"),
        BatchNormalization(),
        Dropout(0.4),
        Dense(2, activation="softmax"),
    ])
    m.build(input_shape=(None, MAX_LEN))
    return m


def build_bigru():
    inp = Input(shape=(MAX_LEN,))
    x = Embedding(input_dim=VOCAB, output_dim=EMBED_DIM)(inp)
    x = SpatialDropout1D(0.3)(x)
    x = Conv1D(128, 3, padding="same", activation="relu")(x)
    x = Conv1D(64, 3, padding="same", activation="relu")(x)
    x = Bidirectional(GRU(64, return_sequences=True))(x)
    x = Bidirectional(GRU(32, return_sequences=True))(x)
    x = AttentionPool()(x)
    x = Dense(64, activation="relu")(x)
    x = BatchNormalization()(x)
    x = Dropout(0.4)(x)
    out = Dense(2, activation="softmax")(x)
    return Model(inp, out, name="CNN_BiGRU_Attention")


bilstm_model = build_bilstm()
bilstm_model.load_weights("bilstm_sentiment_model.keras")

bigru_model = build_bigru()
bigru_model.load_weights("cnn_bigru_attention_sentiment.keras")

SENT_MODELS = {"BiLSTM": bilstm_model, "CNN-BiGRU-Attention": bigru_model}


def _sent_card(pos, base, model_name, n_tok):
    """Headline verdict card for sentiment."""
    positive = pos >= 0.5
    verdict = "POSITIVE REVIEW" if positive else "NEGATIVE REVIEW"
    conf = pos if positive else (1 - pos)
    accent = "#15803d" if positive else "#b91c1c"
    bg = "#f0fdf4" if positive else "#fef2f2"
    delta = (pos - base) * 100
    if conf >= 0.90:
        strength, scol = "STRONG", accent
    elif conf >= 0.70:
        strength, scol = "MODERATE", "#a16207"
    else:
        strength, scol = "BORDERLINE", "#6b7280"

    def stat(v, lab, col="#111827"):
        return (f"<div style='flex:1;min-width:110px'>"
                f"<div style='font-size:1.45rem;font-weight:700;color:{col}'>{v}</div>"
                f"<div style='font-size:.66rem;letter-spacing:.06em;color:#6b7280'>{lab}</div></div>")

    return f"""
<div style="border:1px solid {accent}33;border-left:5px solid {accent};background:{bg};
            border-radius:10px;padding:14px 16px;font-family:system-ui,sans-serif">
  <div style="font-weight:700;color:{accent};letter-spacing:.02em">{verdict}</div>
  <div style="font-size:.82rem;color:#4b5563;margin:2px 0 12px">
      The model reads this review as {'positive' if positive else 'negative'} overall.</div>
  <div style="display:flex;gap:14px;flex-wrap:wrap">
    {stat(f"{conf:.1%}", "CONFIDENCE")}
    {stat(f"{pos:.1%}", "POSITIVE PROBABILITY")}
    {stat(f"{delta:+.1f} pts", "VS EMPTY REVIEW")}
    {stat(strength, "STRENGTH", scol)}
  </div>
  <div style="font-size:.68rem;color:#6b7280;margin-top:10px">
      Decision threshold 50.0%. {n_tok} words analysed (cap {MAX_LEN}).
      An empty review scores {base:.1%}. Model: {model_name}.</div>
</div>"""


def _word_tables(rows):
    """Top words pushing each way."""
    posw = [r for r in rows if r[1] > 0][:8]
    negw = [r for r in rows if r[1] < 0][:8]
    top = max([abs(r[1]) for r in rows] + [1e-9])

    def block(items, title, col, bg):
        if not items:
            body = ("<tr><td colspan='2' style='padding:10px;color:#9ca3af;"
                    "font-size:.78rem'>No words push this way.</td></tr>")
        else:
            body = ""
            for w, c in items:
                width = min(abs(c) / top * 100, 100)
                body += f"""<tr style="border-top:1px solid #e5e7eb">
                  <td style="padding:6px 8px;font-size:.78rem">{w}</td>
                  <td style="padding:6px 8px;width:130px">
                    <div style="display:flex;align-items:center;gap:6px">
                      <div style="height:7px;width:{width}%;background:{col};border-radius:4px"></div>
                      <span style="font-size:.66rem;color:#6b7280">{abs(c)*100:.2f}</span>
                    </div></td></tr>"""
        return f"""
        <div style="flex:1;min-width:250px;border:1px solid {col}33;border-radius:9px;overflow:hidden">
          <div style="background:{bg};color:{col};font-weight:700;font-size:.74rem;
                      padding:7px 9px;letter-spacing:.04em">{title}</div>
          <table style="width:100%;border-collapse:collapse;background:#fff">
            <tr style="background:#f9fafb">
              <th style="text-align:left;padding:5px 8px;font-size:.62rem;color:#6b7280">WORD</th>
              <th style="text-align:left;padding:5px 8px;font-size:.62rem;color:#6b7280">INFLUENCE</th>
            </tr>{body}
          </table></div>"""

    return f"""
<div style="font-family:system-ui,sans-serif">
  <div style="display:flex;gap:12px;flex-wrap:wrap">
    {block(posw, "PUSHING TOWARDS POSITIVE", "#15803d", "#dcfce7")}
    {block(negw, "PUSHING TOWARDS NEGATIVE", "#b91c1c", "#fee2e2")}
  </div>
  <div style="font-size:.66rem;color:#9ca3af;margin-top:8px">
    Influence is measured by masking one word at a time and recording how far the model's
    output moves. Larger bars mean the word moved the decision further. Stopwords are
    removed and words are lemmatised before the model sees them.</div>
</div>"""


def predict_sentiment(model_name, text):
    if not text or not text.strip():
        return {}, "", [], ""

    cleaned = clean_text(text)
    seq = sent_tokenizer.texts_to_sequences([cleaned])[0][:MAX_LEN]
    if not seq:
        return {}, "", [("No recognisable words after cleaning.", None)], ""

    words = cleaned.split()[:len(seq)]
    n = len(seq)
    model = SENT_MODELS[model_name]

    # row 0 = review, rows 1..n = review with word i masked, row n+1 = empty review
    batch = np.zeros((n + 2, MAX_LEN), dtype="int32")
    batch[0, :n] = seq
    for i in range(n):
        batch[i + 1, :n] = seq
        batch[i + 1, i] = 0            # 0 is the padding index
    # row n+1 stays all zeros

    probs = model.predict(batch, verbose=0)
    pos_idx = [i for i, v in label_map.items() if str(v).lower().startswith("pos")]
    pos_idx = pos_idx[0] if pos_idx else 1

    pos = float(probs[0][pos_idx])
    base = float(probs[n + 1][pos_idx])

    rows = [(words[i], pos - float(probs[i + 1][pos_idx])) for i in range(n)]
    ranked = sorted(rows, key=lambda r: abs(r[1]), reverse=True)

    # continuous highlight: + green (positive), - red (negative)
    top = max([abs(c) for _, c in rows] + [1e-9])
    highlighted = [(w, round(c / top, 3)) for w, c in rows]

    scores = {label_map[i]: float(probs[0][i]) for i in range(probs.shape[1])}
    return (scores,
            _sent_card(pos, base, model_name, n),
            highlighted,
            _word_tables(ranked))


# =========================================================================== #
#  UI
# =========================================================================== #

def _choices(col):
    return list(label_encoders[col].classes_)

with gr.Blocks(title="Explainable Attrition & Sentiment Platform") as demo:
    gr.Markdown(
        "# Real-Time Explainable Deep Learning Platform\n"
        "Employee **Attrition Prediction** and **Sentiment Analysis**."
    )

    with gr.Tab("Attrition Prediction"):
        attr_model = gr.Radio(list(ATTR_MODELS), value="FT-Transformer", label="Model")
        with gr.Row():
            with gr.Column():
                city    = gr.Dropdown(_choices("city"), value="city_103", label="City")
                city_dev = gr.Slider(0.0, 1.0, value=0.92, step=0.001,
                                     label="City development index")
                gender  = gr.Dropdown(_choices("gender"), value="Male", label="Gender")
                rel_exp = gr.Dropdown(_choices("relevent_experience"),
                                      value="Has relevent experience",
                                      label="Relevant experience")
                enrolled = gr.Dropdown(_choices("enrolled_university"),
                                       value="no_enrollment", label="Enrolled university")
                edu     = gr.Dropdown(_choices("education_level"),
                                      value="Graduate", label="Education level")
            with gr.Column():
                major   = gr.Dropdown(_choices("major_discipline"),
                                      value="STEM", label="Major discipline")
                exp     = gr.Dropdown(_choices("experience"), value="10", label="Experience (yrs)")
                comp_sz = gr.Dropdown(_choices("company_size"),
                                      value="50-99", label="Company size")
                comp_ty = gr.Dropdown(_choices("company_type"),
                                      value="Pvt Ltd", label="Company type")
                last_jb = gr.Dropdown(_choices("last_new_job"),
                                      value="1", label="Years since last new job")
                train_h = gr.Slider(0, 400, value=40, step=1, label="Training hours")

        attr_btn = gr.Button("Predict attrition", variant="primary")
        attr_card = gr.HTML()
        with gr.Accordion("Prediction probabilities", open=False):
            attr_out = gr.Label(label="Probability", show_label=False)
        attr_expl = gr.HTML()
        attr_btn.click(
            predict_attrition,
            [attr_model, city, city_dev, gender, rel_exp, enrolled, edu, major,
             exp, comp_sz, comp_ty, last_jb, train_h],
            [attr_out, attr_card, attr_expl],
        )

    with gr.Tab("Sentiment Analysis"):
        sent_model = gr.Radio(list(SENT_MODELS), value="BiLSTM", label="Model")
        sent_in = gr.Textbox(lines=4, label="Review text",
                             placeholder="Great place to work, supportive managers...")
        sent_btn = gr.Button("Analyse sentiment", variant="primary")
        sent_card = gr.HTML()
        with gr.Accordion("Sentiment scores", open=False):
            sent_out = gr.Label(label="Sentiment scores", show_label=False)
        sent_hl = gr.HighlightedText(
            label="Word influence (green pushes positive, red pushes negative)",
            combine_adjacent=False, show_legend=False, color_map=None)
        sent_expl = gr.HTML()
        sent_btn.click(predict_sentiment, [sent_model, sent_in],
                       [sent_out, sent_card, sent_hl, sent_expl])

if __name__ == "__main__":
    demo.launch()
