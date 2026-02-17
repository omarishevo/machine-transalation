"""
Machine Translation with Transformer - Streamlit App
Deep Learning Project: English ↔ French Translation
"""

import streamlit as st
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math
import time
import json
import random
from collections import Counter

# ──────────────────────────────────────────────
# PAGE CONFIG
# ──────────────────────────────────────────────
st.set_page_config(
    page_title="Transformer Machine Translation",
    page_icon="🌐",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ──────────────────────────────────────────────
# CUSTOM CSS
# ──────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Mono:wght@400;700&family=Sora:wght@300;400;600;700&display=swap');

:root {
    --bg-dark:     #0a0e1a;
    --bg-card:     #111827;
    --bg-input:    #1a2235;
    --accent-cyan: #00e5ff;
    --accent-lime: #b2ff59;
    --accent-rose: #ff4d6d;
    --text-main:   #e2e8f0;
    --text-muted:  #8892a4;
    --border:      #1e2d45;
}

html, body, [class*="css"] {
    font-family: 'Sora', sans-serif;
    background-color: var(--bg-dark);
    color: var(--text-main);
}

/* Hide streamlit chrome */
#MainMenu, footer, header { visibility: hidden; }
.block-container { padding-top: 1rem; max-width: 1200px; }

/* ── HERO BANNER ── */
.hero {
    background: linear-gradient(135deg, #0d1b2e 0%, #0a0e1a 50%, #0d1421 100%);
    border: 1px solid var(--border);
    border-radius: 16px;
    padding: 2.5rem 3rem;
    margin-bottom: 2rem;
    position: relative;
    overflow: hidden;
}
.hero::before {
    content: '';
    position: absolute;
    top: -60px; right: -60px;
    width: 240px; height: 240px;
    background: radial-gradient(circle, rgba(0,229,255,0.12) 0%, transparent 70%);
    border-radius: 50%;
}
.hero::after {
    content: '';
    position: absolute;
    bottom: -40px; left: 40px;
    width: 180px; height: 180px;
    background: radial-gradient(circle, rgba(178,255,89,0.08) 0%, transparent 70%);
    border-radius: 50%;
}
.hero-title {
    font-family: 'Space Mono', monospace;
    font-size: 2.2rem;
    font-weight: 700;
    color: var(--accent-cyan);
    letter-spacing: -1px;
    margin: 0 0 0.4rem 0;
}
.hero-sub {
    font-size: 1rem;
    color: var(--text-muted);
    font-weight: 300;
    letter-spacing: 0.5px;
}
.badge {
    display: inline-block;
    background: rgba(0,229,255,0.1);
    border: 1px solid rgba(0,229,255,0.3);
    color: var(--accent-cyan);
    font-family: 'Space Mono', monospace;
    font-size: 0.65rem;
    padding: 3px 10px;
    border-radius: 20px;
    margin-right: 6px;
    letter-spacing: 1px;
}

/* ── CARDS ── */
.card {
    background: var(--bg-card);
    border: 1px solid var(--border);
    border-radius: 12px;
    padding: 1.5rem;
    margin-bottom: 1rem;
}
.card-title {
    font-family: 'Space Mono', monospace;
    font-size: 0.75rem;
    color: var(--accent-lime);
    letter-spacing: 2px;
    text-transform: uppercase;
    margin-bottom: 1rem;
    display: flex;
    align-items: center;
    gap: 8px;
}

/* ── TRANSLATION OUTPUT ── */
.translation-box {
    background: linear-gradient(135deg, #0d1b2e, #111827);
    border: 1px solid rgba(0,229,255,0.25);
    border-radius: 12px;
    padding: 1.8rem;
    margin-top: 1rem;
    position: relative;
}
.translation-label {
    font-family: 'Space Mono', monospace;
    font-size: 0.65rem;
    letter-spacing: 2px;
    color: var(--accent-cyan);
    text-transform: uppercase;
    margin-bottom: 0.6rem;
}
.translation-text {
    font-size: 1.4rem;
    font-weight: 600;
    color: var(--text-main);
    line-height: 1.5;
}

/* ── METRIC CARDS ── */
.metric-row { display: flex; gap: 12px; margin-top: 1rem; }
.metric-card {
    flex: 1;
    background: var(--bg-input);
    border: 1px solid var(--border);
    border-radius: 10px;
    padding: 1rem;
    text-align: center;
}
.metric-val {
    font-family: 'Space Mono', monospace;
    font-size: 1.6rem;
    font-weight: 700;
    color: var(--accent-lime);
}
.metric-lbl {
    font-size: 0.7rem;
    color: var(--text-muted);
    text-transform: uppercase;
    letter-spacing: 1px;
    margin-top: 4px;
}

/* ── ATTENTION HEATMAP ── */
.attn-grid { font-family: 'Space Mono', monospace; font-size: 0.7rem; }
.attn-header { color: var(--accent-cyan); margin-bottom: 6px; }

/* ── ARCHITECTURE DIAGRAM ── */
.arch-layer {
    display: flex;
    align-items: center;
    justify-content: center;
    border-radius: 8px;
    padding: 10px 20px;
    margin: 6px 0;
    font-family: 'Space Mono', monospace;
    font-size: 0.75rem;
    text-align: center;
    transition: all 0.3s;
}

/* ── SIDEBAR STYLING ── */
[data-testid="stSidebar"] {
    background-color: var(--bg-card) !important;
    border-right: 1px solid var(--border);
}
[data-testid="stSidebar"] .stSelectbox label,
[data-testid="stSidebar"] .stSlider label,
[data-testid="stSidebar"] .stNumberInput label {
    color: var(--text-muted) !important;
    font-size: 0.8rem !important;
    font-family: 'Space Mono', monospace !important;
    letter-spacing: 1px;
}

/* Buttons */
.stButton > button {
    background: linear-gradient(135deg, #00e5ff 0%, #0097a7 100%);
    color: #0a0e1a;
    font-family: 'Space Mono', monospace;
    font-weight: 700;
    font-size: 0.85rem;
    letter-spacing: 1px;
    border: none;
    border-radius: 8px;
    padding: 0.65rem 1.8rem;
    transition: all 0.25s;
}
.stButton > button:hover { transform: translateY(-2px); box-shadow: 0 6px 20px rgba(0,229,255,0.3); }

/* Text areas */
.stTextArea textarea {
    background: var(--bg-input) !important;
    border: 1px solid var(--border) !important;
    color: var(--text-main) !important;
    font-family: 'Sora', sans-serif !important;
    font-size: 1rem !important;
    border-radius: 8px !important;
}
.stTextArea textarea:focus { border-color: var(--accent-cyan) !important; box-shadow: 0 0 0 2px rgba(0,229,255,0.15) !important; }

/* Tabs */
.stTabs [data-baseweb="tab-list"] { background: var(--bg-card); border-radius: 10px; padding: 4px; gap: 4px; }
.stTabs [data-baseweb="tab"] { font-family: 'Space Mono', monospace; font-size: 0.75rem; letter-spacing: 1px; color: var(--text-muted); border-radius: 7px; padding: 8px 18px; }
.stTabs [aria-selected="true"] { background: rgba(0,229,255,0.15) !important; color: var(--accent-cyan) !important; }

/* Progress & info boxes */
.stProgress > div > div { background: var(--accent-cyan) !important; }
.stInfo { background: rgba(0,229,255,0.07) !important; border: 1px solid rgba(0,229,255,0.2) !important; color: var(--text-main) !important; }
.stSuccess { background: rgba(178,255,89,0.07) !important; border: 1px solid rgba(178,255,89,0.2) !important; color: var(--text-main) !important; }
.stWarning { background: rgba(255,193,7,0.07) !important; border: 1px solid rgba(255,193,7,0.2) !important; color: var(--text-main) !important; }

/* Selectbox */
.stSelectbox > div > div { background: var(--bg-input) !important; border-color: var(--border) !important; color: var(--text-main) !important; }

/* Divider */
hr { border-color: var(--border) !important; margin: 1.5rem 0 !important; }

.token-pill {
    display: inline-block;
    background: rgba(0,229,255,0.08);
    border: 1px solid rgba(0,229,255,0.2);
    color: var(--accent-cyan);
    font-family: 'Space Mono', monospace;
    font-size: 0.7rem;
    padding: 2px 8px;
    border-radius: 4px;
    margin: 2px;
}

.step-box {
    background: var(--bg-input);
    border-left: 3px solid var(--accent-cyan);
    padding: 10px 14px;
    border-radius: 0 8px 8px 0;
    margin: 6px 0;
    font-size: 0.85rem;
    color: var(--text-muted);
}
</style>
""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════
#  TRANSFORMER MODEL COMPONENTS
# ══════════════════════════════════════════════════════

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000, dropout=0.1):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)


class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_heads, dropout=0.1):
        super().__init__()
        assert d_model % num_heads == 0
        self.d_k = d_model // num_heads
        self.num_heads = num_heads
        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.W_o = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)
        self.attention_weights = None

    def scaled_dot_product_attention(self, Q, K, V, mask=None):
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
        attn = F.softmax(scores, dim=-1)
        self.attention_weights = attn.detach()
        attn = self.dropout(attn)
        return torch.matmul(attn, V), attn

    def split_heads(self, x, batch_size):
        x = x.view(batch_size, -1, self.num_heads, self.d_k)
        return x.transpose(1, 2)

    def forward(self, query, key, value, mask=None):
        batch_size = query.size(0)
        Q = self.split_heads(self.W_q(query), batch_size)
        K = self.split_heads(self.W_k(key), batch_size)
        V = self.split_heads(self.W_v(value), batch_size)
        x, attn = self.scaled_dot_product_attention(Q, K, V, mask)
        x = x.transpose(1, 2).contiguous().view(batch_size, -1, self.num_heads * self.d_k)
        return self.W_o(x)


class FeedForward(nn.Module):
    def __init__(self, d_model, d_ff, dropout=0.1):
        super().__init__()
        self.linear1 = nn.Linear(d_model, d_ff)
        self.linear2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        return self.linear2(self.dropout(F.relu(self.linear1(x))))


class EncoderLayer(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, dropout=0.1):
        super().__init__()
        self.self_attn = MultiHeadAttention(d_model, num_heads, dropout)
        self.feed_forward = FeedForward(d_model, d_ff, dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask=None):
        attn_out = self.self_attn(x, x, x, mask)
        x = self.norm1(x + self.dropout(attn_out))
        ff_out = self.feed_forward(x)
        x = self.norm2(x + self.dropout(ff_out))
        return x


class DecoderLayer(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, dropout=0.1):
        super().__init__()
        self.self_attn   = MultiHeadAttention(d_model, num_heads, dropout)
        self.cross_attn  = MultiHeadAttention(d_model, num_heads, dropout)
        self.feed_forward = FeedForward(d_model, d_ff, dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, enc_out, src_mask=None, tgt_mask=None):
        attn1 = self.self_attn(x, x, x, tgt_mask)
        x = self.norm1(x + self.dropout(attn1))
        attn2 = self.cross_attn(x, enc_out, enc_out, src_mask)
        x = self.norm2(x + self.dropout(attn2))
        ff_out = self.feed_forward(x)
        x = self.norm3(x + self.dropout(ff_out))
        return x


class TransformerTranslator(nn.Module):
    def __init__(self, src_vocab, tgt_vocab, d_model=256, num_heads=8,
                 num_enc_layers=3, num_dec_layers=3, d_ff=512, dropout=0.1, max_len=100):
        super().__init__()
        self.d_model = d_model
        self.src_embed = nn.Embedding(src_vocab, d_model)
        self.tgt_embed = nn.Embedding(tgt_vocab, d_model)
        self.pos_enc   = PositionalEncoding(d_model, max_len, dropout)
        self.encoder_layers = nn.ModuleList([EncoderLayer(d_model, num_heads, d_ff, dropout) for _ in range(num_enc_layers)])
        self.decoder_layers = nn.ModuleList([DecoderLayer(d_model, num_heads, d_ff, dropout) for _ in range(num_dec_layers)])
        self.fc_out  = nn.Linear(d_model, tgt_vocab)
        self.dropout = nn.Dropout(dropout)
        self._init_weights()

    def _init_weights(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def encode(self, src, src_mask=None):
        x = self.dropout(self.pos_enc(self.src_embed(src) * math.sqrt(self.d_model)))
        for layer in self.encoder_layers:
            x = layer(x, src_mask)
        return x

    def decode(self, tgt, enc_out, src_mask=None, tgt_mask=None):
        x = self.dropout(self.pos_enc(self.tgt_embed(tgt) * math.sqrt(self.d_model)))
        for layer in self.decoder_layers:
            x = layer(x, enc_out, src_mask, tgt_mask)
        return x

    def forward(self, src, tgt, src_mask=None, tgt_mask=None):
        enc_out = self.encode(src, src_mask)
        dec_out = self.decode(tgt, enc_out, src_mask, tgt_mask)
        return self.fc_out(dec_out)

    def get_attention_weights(self):
        weights = {}
        for i, layer in enumerate(self.encoder_layers):
            w = layer.self_attn.attention_weights
            if w is not None:
                weights[f'enc_layer_{i+1}'] = w.cpu().numpy()
        for i, layer in enumerate(self.decoder_layers):
            w = layer.cross_attn.attention_weights
            if w is not None:
                weights[f'dec_cross_{i+1}'] = w.cpu().numpy()
        return weights


# ══════════════════════════════════════════════════════
#  VOCABULARY & TOKENISATION
# ══════════════════════════════════════════════════════

class Vocabulary:
    PAD, SOS, EOS, UNK = 0, 1, 2, 3

    def __init__(self, name):
        self.name = name
        self.word2idx = {'<PAD>': 0, '<SOS>': 1, '<EOS>': 2, '<UNK>': 3}
        self.idx2word = {v: k for k, v in self.word2idx.items()}
        self.word_freq = Counter()

    def add_sentence(self, sentence):
        for w in sentence.lower().split():
            self.word_freq[w] += 1
            if w not in self.word2idx:
                idx = len(self.word2idx)
                self.word2idx[w] = idx
                self.idx2word[idx] = w

    def encode(self, sentence, max_len=50):
        tokens = [self.SOS] + [self.word2idx.get(w, self.UNK) for w in sentence.lower().split()] + [self.EOS]
        tokens = tokens[:max_len]
        tokens += [self.PAD] * (max_len - len(tokens))
        return tokens

    def decode(self, tokens):
        words = []
        for t in tokens:
            w = self.idx2word.get(t, '<UNK>')
            if w in ('<PAD>', '<SOS>', '<EOS>'): continue
            words.append(w)
        return ' '.join(words)

    def __len__(self):
        return len(self.word2idx)


# ══════════════════════════════════════════════════════
#  SAMPLE BILINGUAL CORPUS (EN → FR)
# ══════════════════════════════════════════════════════

EN_FR_PAIRS = [
    ("hello", "bonjour"),
    ("good morning", "bonjour"),
    ("good evening", "bonsoir"),
    ("good night", "bonne nuit"),
    ("thank you", "merci"),
    ("thank you very much", "merci beaucoup"),
    ("you are welcome", "de rien"),
    ("please", "s'il vous plaît"),
    ("sorry", "désolé"),
    ("excuse me", "excusez-moi"),
    ("how are you", "comment allez-vous"),
    ("i am fine", "je vais bien"),
    ("what is your name", "quel est votre nom"),
    ("my name is john", "je m'appelle john"),
    ("where are you from", "d'où venez-vous"),
    ("i am from france", "je suis de france"),
    ("i love you", "je t'aime"),
    ("i miss you", "tu me manques"),
    ("see you later", "à bientôt"),
    ("goodbye", "au revoir"),
    ("the cat is black", "le chat est noir"),
    ("the dog is white", "le chien est blanc"),
    ("the sky is blue", "le ciel est bleu"),
    ("the apple is red", "la pomme est rouge"),
    ("the book is interesting", "le livre est intéressant"),
    ("i like to read books", "j'aime lire des livres"),
    ("she is very beautiful", "elle est très belle"),
    ("he is very smart", "il est très intelligent"),
    ("we are happy", "nous sommes heureux"),
    ("they are going to school", "ils vont à l'école"),
    ("i want to eat", "je veux manger"),
    ("i am hungry", "j'ai faim"),
    ("i am thirsty", "j'ai soif"),
    ("can i help you", "puis-je vous aider"),
    ("where is the bathroom", "où est la salle de bain"),
    ("i do not understand", "je ne comprends pas"),
    ("please speak slowly", "parlez lentement s'il vous plaît"),
    ("the weather is nice today", "il fait beau aujourd'hui"),
    ("it is raining", "il pleut"),
    ("i have a question", "j'ai une question"),
    ("this is important", "c'est important"),
    ("i work in a hospital", "je travaille dans un hôpital"),
    ("the train arrives at noon", "le train arrive à midi"),
    ("i need a doctor", "j'ai besoin d'un médecin"),
    ("how much does this cost", "combien ça coûte"),
    ("that is too expensive", "c'est trop cher"),
    ("where is the hotel", "où est l'hôtel"),
    ("i want to go to paris", "je veux aller à paris"),
    ("i speak a little french", "je parle un peu français"),
    ("english is my mother tongue", "l'anglais est ma langue maternelle"),
]


# ══════════════════════════════════════════════════════
#  SESSION-STATE HELPERS
# ══════════════════════════════════════════════════════

@st.cache_resource
def build_vocab_and_model(d_model, num_heads, num_layers, d_ff, dropout):
    src_vocab = Vocabulary("EN")
    tgt_vocab = Vocabulary("FR")
    for en, fr in EN_FR_PAIRS:
        src_vocab.add_sentence(en)
        tgt_vocab.add_sentence(fr)

    model = TransformerTranslator(
        src_vocab=len(src_vocab),
        tgt_vocab=len(tgt_vocab),
        d_model=d_model,
        num_heads=num_heads,
        num_enc_layers=num_layers,
        num_dec_layers=num_layers,
        d_ff=d_ff,
        dropout=dropout
    )
    return src_vocab, tgt_vocab, model


def make_causal_mask(size):
    mask = torch.tril(torch.ones(size, size)).unsqueeze(0).unsqueeze(0)
    return mask


def greedy_decode(model, src_tensor, src_vocab, tgt_vocab, max_len=30):
    model.eval()
    with torch.no_grad():
        enc_out = model.encode(src_tensor)
        tgt = torch.tensor([[tgt_vocab.SOS]])
        for _ in range(max_len):
            tgt_mask = make_causal_mask(tgt.size(1))
            dec_out = model.decode(tgt, enc_out, tgt_mask=tgt_mask)
            logits = model.fc_out(dec_out[:, -1, :])
            next_tok = logits.argmax(dim=-1).unsqueeze(0)
            tgt = torch.cat([tgt, next_tok], dim=1)
            if next_tok.item() == tgt_vocab.EOS:
                break
    return tgt.squeeze(0).tolist()


# ══════════════════════════════════════════════════════
#  TRAINING LOOP
# ══════════════════════════════════════════════════════

def train_model(model, src_vocab, tgt_vocab, epochs, lr, batch_size, progress_cb, log_cb):
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, betas=(0.9, 0.98), eps=1e-9)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5, factor=0.5)
    criterion = nn.CrossEntropyLoss(ignore_index=0)

    model.train()
    history = {'loss': [], 'acc': []}
    max_len = 30

    for epoch in range(epochs):
        total_loss, total_correct, total_tokens = 0.0, 0, 0
        random.shuffle(EN_FR_PAIRS)

        for i in range(0, len(EN_FR_PAIRS), batch_size):
            batch = EN_FR_PAIRS[i:i+batch_size]
            src_batch = torch.tensor([src_vocab.encode(en, max_len) for en, _ in batch])
            tgt_batch = torch.tensor([tgt_vocab.encode(fr, max_len) for _, fr in batch])

            tgt_in  = tgt_batch[:, :-1]
            tgt_out = tgt_batch[:, 1:]
            tgt_mask = make_causal_mask(tgt_in.size(1))

            optimizer.zero_grad()
            logits = model(src_batch, tgt_in, tgt_mask=tgt_mask)
            loss = criterion(logits.reshape(-1, len(tgt_vocab)), tgt_out.reshape(-1))
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            total_loss += loss.item()
            preds = logits.argmax(dim=-1)
            mask  = tgt_out != 0
            total_correct += (preds == tgt_out)[mask].sum().item()
            total_tokens  += mask.sum().item()

        avg_loss = total_loss / max(1, len(EN_FR_PAIRS) // batch_size)
        acc = total_correct / max(1, total_tokens)
        scheduler.step(avg_loss)
        history['loss'].append(avg_loss)
        history['acc'].append(acc)
        progress_cb((epoch + 1) / epochs)
        log_cb(epoch + 1, avg_loss, acc, optimizer.param_groups[0]['lr'])

    return history


# ══════════════════════════════════════════════════════
#  MAIN UI
# ══════════════════════════════════════════════════════

def main():
    # ── HERO ──────────────────────────────────────────
    st.markdown("""
    <div class="hero">
        <div style="margin-bottom:0.8rem;">
            <span class="badge">DEEP LEARNING</span>
            <span class="badge">NLP</span>
            <span class="badge">PYTORCH</span>
        </div>
        <div class="hero-title">🌐 Transformer Machine Translation</div>
        <div class="hero-sub">Build, train & explore a full Transformer architecture for English → French translation</div>
    </div>
    """, unsafe_allow_html=True)

    # ── SIDEBAR ────────────────────────────────────────
    with st.sidebar:
        st.markdown("### ⚙️ Model Configuration")
        st.markdown("---")

        d_model    = st.select_slider("Embedding Dimension (d_model)", options=[64, 128, 256, 512], value=128)
        num_heads  = st.selectbox("Attention Heads", [2, 4, 8], index=1)
        num_layers = st.slider("Encoder / Decoder Layers", 1, 6, 2)
        d_ff       = st.select_slider("FFN Hidden Size (d_ff)", options=[128, 256, 512, 1024], value=256)
        dropout    = st.slider("Dropout Rate", 0.0, 0.5, 0.1, 0.05)

        st.markdown("---")
        st.markdown("### 🏋️ Training Hyperparameters")

        epochs     = st.slider("Epochs", 5, 200, 50)
        lr         = st.select_slider("Learning Rate", options=[1e-4, 5e-4, 1e-3, 5e-3], value=1e-3)
        batch_size = st.selectbox("Batch Size", [4, 8, 16, 32], index=1)

        st.markdown("---")
        # Model summary
        src_v, tgt_v, model = build_vocab_and_model(d_model, num_heads, num_layers, d_ff, dropout)
        total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        st.markdown(f"""
        <div class="card">
            <div class="card-title">📊 Model Stats</div>
            <div style="font-size:0.8rem; color:#8892a4;">
                Parameters: <span style="color:#00e5ff; font-family:monospace;">{total_params:,}</span><br>
                Src Vocab: <span style="color:#b2ff59; font-family:monospace;">{len(src_v)}</span><br>
                Tgt Vocab: <span style="color:#b2ff59; font-family:monospace;">{len(tgt_v)}</span><br>
                Training pairs: <span style="color:#b2ff59; font-family:monospace;">{len(EN_FR_PAIRS)}</span>
            </div>
        </div>
        """, unsafe_allow_html=True)

    # Rebuild model if config changed
    src_vocab, tgt_vocab, model = build_vocab_and_model(d_model, num_heads, num_layers, d_ff, dropout)

    # ── TABS ───────────────────────────────────────────
    tab1, tab2, tab3, tab4 = st.tabs(["🏋️  Train", "🌐  Translate", "🔍  Attention", "📐  Architecture"])

    # ──────────────────────────────────────────────────
    # TAB 1 — TRAIN
    # ──────────────────────────────────────────────────
    with tab1:
        col_a, col_b = st.columns([2, 1])

        with col_a:
            st.markdown('<div class="card"><div class="card-title">🔥 Training Control</div>', unsafe_allow_html=True)
            st.markdown(f"Training on **{len(EN_FR_PAIRS)} EN→FR sentence pairs** for **{epochs} epochs**.")

            if st.button("▶  START TRAINING", use_container_width=True):
                progress_bar = st.progress(0.0)
                log_container = st.empty()
                chart_container = st.empty()
                status_placeholder = st.empty()

                training_log = []
                loss_history = []
                acc_history  = []

                def progress_cb(val):
                    progress_bar.progress(val)

                def log_cb(epoch, loss, acc, lr_now):
                    training_log.append(f"Epoch {epoch:>3} | Loss: {loss:.4f} | Acc: {acc*100:5.1f}% | LR: {lr_now:.6f}")
                    loss_history.append(loss)
                    acc_history.append(acc * 100)
                    # Show last 8 log lines
                    log_container.markdown(
                        '\n'.join([f'<div class="step-box">{l}</div>' for l in training_log[-8:]]),
                        unsafe_allow_html=True
                    )
                    if len(loss_history) > 1:
                        import pandas as pd
                        df = pd.DataFrame({'Loss': loss_history, 'Accuracy (%)': acc_history})
                        chart_container.line_chart(df, use_container_width=True)

                history = train_model(model, src_vocab, tgt_vocab, epochs, lr, batch_size,
                                      progress_cb, log_cb)

                st.session_state['trained']      = True
                st.session_state['loss_history'] = history['loss']
                st.session_state['acc_history']  = history['acc']
                st.session_state['model']        = model
                st.success(f"✅ Training complete! Final Loss: {history['loss'][-1]:.4f}  |  Accuracy: {history['acc'][-1]*100:.1f}%")

            st.markdown('</div>', unsafe_allow_html=True)

        with col_b:
            st.markdown('<div class="card"><div class="card-title">📈 Final Metrics</div>', unsafe_allow_html=True)
            if st.session_state.get('trained'):
                fl = st.session_state['loss_history'][-1]
                fa = st.session_state['acc_history'][-1] * 100
                st.markdown(f"""
                <div class="metric-row">
                    <div class="metric-card">
                        <div class="metric-val">{fl:.3f}</div>
                        <div class="metric-lbl">Final Loss</div>
                    </div>
                </div>
                <div class="metric-row">
                    <div class="metric-card">
                        <div class="metric-val">{fa:.1f}%</div>
                        <div class="metric-lbl">Token Acc</div>
                    </div>
                </div>
                """, unsafe_allow_html=True)
            else:
                st.info("Train the model to see metrics.")
            st.markdown('</div>', unsafe_allow_html=True)

            st.markdown('<div class="card"><div class="card-title">💡 Tips</div>', unsafe_allow_html=True)
            st.markdown("""
            <div style="font-size:0.78rem; color:#8892a4; line-height:1.7;">
            • More <b>epochs</b> → lower loss<br>
            • Larger <b>d_model</b> → more capacity<br>
            • More <b>heads</b> → richer attention<br>
            • Reduce LR if training diverges<br>
            • Small dataset → risk of overfitting
            </div>
            """, unsafe_allow_html=True)
            st.markdown('</div>', unsafe_allow_html=True)

    # ──────────────────────────────────────────────────
    # TAB 2 — TRANSLATE
    # ──────────────────────────────────────────────────
    with tab2:
        active_model = st.session_state.get('model', model)
        col1, col2 = st.columns(2)

        with col1:
            st.markdown('<div class="card"><div class="card-title">📝 Input (English)</div>', unsafe_allow_html=True)
            user_input = st.text_area(
                label="",
                placeholder="Type an English sentence…",
                height=120,
                label_visibility="collapsed"
            )

            if not st.session_state.get('trained'):
                st.warning("⚠️ Model is untrained — outputs will be random. Train first for meaningful translations.")

            translate_btn = st.button("🌐  TRANSLATE", use_container_width=True)
            st.markdown('</div>', unsafe_allow_html=True)

            # Example sentences
            st.markdown('<div class="card"><div class="card-title">📚 Try These</div>', unsafe_allow_html=True)
            examples = [e for e, _ in random.sample(EN_FR_PAIRS, min(6, len(EN_FR_PAIRS)))]
            for ex in examples:
                st.markdown(f'<span class="token-pill">{ex}</span>', unsafe_allow_html=True)
            st.markdown('</div>', unsafe_allow_html=True)

        with col2:
            st.markdown('<div class="card"><div class="card-title">🇫🇷 Output (French)</div>', unsafe_allow_html=True)

            if translate_btn and user_input.strip():
                with st.spinner("Translating…"):
                    src_tokens = src_vocab.encode(user_input.strip(), max_len=30)
                    src_tensor = torch.tensor([src_tokens])
                    out_tokens = greedy_decode(active_model, src_tensor, src_vocab, tgt_vocab, max_len=30)
                    translation = tgt_vocab.decode(out_tokens)
                    st.session_state['last_src']   = user_input.strip()
                    st.session_state['last_tgt']   = translation
                    st.session_state['last_src_t'] = src_tokens
                    st.session_state['last_tgt_t'] = out_tokens

                st.markdown(f"""
                <div class="translation-box">
                    <div class="translation-label">Translation</div>
                    <div class="translation-text">{translation if translation.strip() else '(empty — try training more epochs)'}</div>
                </div>
                """, unsafe_allow_html=True)

                # Reference if exact match exists
                for en, fr in EN_FR_PAIRS:
                    if en.lower() == user_input.strip().lower():
                        st.markdown(f"""
                        <div style="margin-top:1rem; padding:10px 14px; background:rgba(178,255,89,0.07);
                                    border:1px solid rgba(178,255,89,0.2); border-radius:8px; font-size:0.8rem;">
                            <span style="color:#b2ff59; font-family:monospace;">REFERENCE:</span>
                            <span style="color:#e2e8f0;"> {fr}</span>
                        </div>
                        """, unsafe_allow_html=True)
                        break
            elif translate_btn:
                st.warning("Please enter a sentence first.")
            else:
                st.markdown("""
                <div style="height:120px; display:flex; align-items:center; justify-content:center; color:#8892a4; font-size:0.9rem;">
                    Translation will appear here…
                </div>
                """, unsafe_allow_html=True)

            st.markdown('</div>', unsafe_allow_html=True)

    # ──────────────────────────────────────────────────
    # TAB 3 — ATTENTION VISUALIZER
    # ──────────────────────────────────────────────────
    with tab3:
        st.markdown('<div class="card"><div class="card-title">🔍 Attention Weight Visualizer</div>', unsafe_allow_html=True)

        active_model = st.session_state.get('model', model)

        attn_input = st.text_input(
            "English sentence to visualize:",
            value=st.session_state.get('last_src', "hello how are you"),
            placeholder="hello how are you"
        )

        if st.button("🔬  VISUALIZE ATTENTION", use_container_width=False):
            src_tokens = src_vocab.encode(attn_input, max_len=30)
            src_tensor = torch.tensor([src_tokens])
            src_words  = attn_input.lower().split()[:10]

            with torch.no_grad():
                active_model.eval()
                enc_out = active_model.encode(src_tensor)
                tgt_in  = torch.tensor([[tgt_vocab.SOS, tgt_vocab.EOS]])
                tgt_mask = make_causal_mask(tgt_in.size(1))
                active_model.decode(tgt_in, enc_out, tgt_mask=tgt_mask)

            weights = active_model.get_attention_weights()

            if weights:
                layer_names = list(weights.keys())
                sel_layer = st.selectbox("Select layer", layer_names)
                w = weights[sel_layer]  # (batch, heads, tgt_len, src_len)

                head_idx = st.slider("Head index", 0, w.shape[1]-1, 0)
                attn_matrix = w[0, head_idx]  # (tgt, src)

                # Display heatmap as a table
                import pandas as pd
                n_tgt, n_src = attn_matrix.shape
                row_labels = [f"out_{i}" for i in range(min(n_tgt, 5))]
                col_labels = ['<SOS>'] + src_words[:n_src-1]
                col_labels = col_labels[:n_src]

                df_attn = pd.DataFrame(
                    attn_matrix[:len(row_labels), :len(col_labels)],
                    index=row_labels,
                    columns=col_labels
                )
                st.dataframe(df_attn.style.background_gradient(cmap='Blues', axis=None).format("{:.3f}"))

                st.markdown("""
                <div style="font-size:0.78rem; color:#8892a4; margin-top:0.5rem;">
                    Darker = higher attention weight. Each row is a decoder position; each column is an encoder token.
                </div>
                """, unsafe_allow_html=True)
            else:
                st.info("No attention weights captured yet — run a translation first.")

        st.markdown('</div>', unsafe_allow_html=True)

        # Token breakdown
        st.markdown('<div class="card"><div class="card-title">🪙 Token Breakdown</div>', unsafe_allow_html=True)
        tok_input = st.text_input("Tokenize sentence:", value="hello how are you", key="tok_input")
        if tok_input:
            col_s, col_t = st.columns(2)
            with col_s:
                st.markdown("**Source tokens (EN)**")
                ids = src_vocab.encode(tok_input, max_len=20)
                for tok_id in ids:
                    word = src_vocab.idx2word.get(tok_id, '?')
                    if word not in ('<PAD>',):
                        st.markdown(f'`{tok_id}` → <span class="token-pill">{word}</span>', unsafe_allow_html=True)
            with col_t:
                st.markdown("**Top-5 FR vocab by frequency**")
                top5 = tgt_vocab.word_freq.most_common(5)
                for w, c in top5:
                    st.markdown(f'<span class="token-pill">{w}</span> ({c})', unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)

    # ──────────────────────────────────────────────────
    # TAB 4 — ARCHITECTURE
    # ──────────────────────────────────────────────────
    with tab4:
        col_arch, col_info = st.columns([2, 1])

        with col_arch:
            st.markdown('<div class="card"><div class="card-title">📐 Transformer Architecture</div>', unsafe_allow_html=True)

            layers_def = [
                ("#00e5ff", "⬆ OUTPUT PROBABILITIES  (Linear + Softmax)"),
                ("#0097a7", f"🟦 DECODER STACK  (×{num_layers} layers)"),
                ("#2d3748", "   └─ Feed-Forward Network"),
                ("#2d3748", "   └─ Cross-Attention  (Q from decoder, K/V from encoder)"),
                ("#2d3748", "   └─ Masked Self-Attention"),
                ("#1a2235", "   ↕  Residual connections + LayerNorm at each sub-layer"),
                ("#0d3b26", f"🟩 ENCODER STACK  (×{num_layers} layers)"),
                ("#1a3a1a", "   └─ Feed-Forward Network"),
                ("#1a3a1a", "   └─ Multi-Head Self-Attention  ({num_heads} heads)"),
                ("#1a2235", "   ↕  Residual connections + LayerNorm at each sub-layer"),
                ("#2d2510", "📍 POSITIONAL ENCODING  (sinusoidal, added to embeddings)"),
                ("#3d2a10", f"🔤 INPUT EMBEDDING  (vocab→ℝ^{d_model})"),
                ("#1e2d45", "📥 SOURCE TOKENS  (tokenised English)"),
            ]
            for color, label in layers_def:
                st.markdown(f"""
                <div class="arch-layer" style="background:{color}; border:1px solid rgba(255,255,255,0.06);">
                    {label}
                </div>
                """, unsafe_allow_html=True)
            st.markdown('</div>', unsafe_allow_html=True)

        with col_info:
            st.markdown('<div class="card"><div class="card-title">🔢 Dimensions</div>', unsafe_allow_html=True)
            params_table = {
                "d_model": d_model,
                "num_heads": num_heads,
                "d_k = d_v": d_model // num_heads,
                "d_ff": d_ff,
                "enc layers": num_layers,
                "dec layers": num_layers,
                "dropout": dropout,
                "src vocab": len(src_vocab),
                "tgt vocab": len(tgt_vocab),
                "total params": f"{sum(p.numel() for p in model.parameters()):,}",
            }
            for k, v in params_table.items():
                st.markdown(f"""
                <div style="display:flex; justify-content:space-between; padding:4px 0;
                            border-bottom:1px solid #1e2d45; font-size:0.8rem;">
                    <span style="color:#8892a4; font-family:monospace;">{k}</span>
                    <span style="color:#00e5ff; font-family:monospace; font-weight:700;">{v}</span>
                </div>
                """, unsafe_allow_html=True)
            st.markdown('</div>', unsafe_allow_html=True)

            st.markdown('<div class="card"><div class="card-title">📖 Key Equations</div>', unsafe_allow_html=True)
            st.markdown("""
            **Scaled Dot-Product Attention:**
            $$\\text{Attn}(Q,K,V) = \\text{softmax}\\!\\left(\\frac{QK^T}{\\sqrt{d_k}}\\right)V$$

            **Multi-Head Attention:**
            $$\\text{MHA}(Q,K,V) = \\text{Concat}(h_1,\\ldots,h_h)W^O$$

            **Position Encoding:**
            $$PE_{(pos,2i)} = \\sin\\!\\left(\\frac{pos}{10000^{2i/d}}\\right)$$
            """)
            st.markdown('</div>', unsafe_allow_html=True)


if __name__ == "__main__":
    main()
