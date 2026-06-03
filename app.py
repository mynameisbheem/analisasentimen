import streamlit as st
import pandas as pd
import numpy as np
import pickle
import re
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud
import nltk
from nltk.tokenize import word_tokenize
from Sastrawi.StopWordRemover.StopWordRemoverFactory import StopWordRemoverFactory
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory

# =============================================
# KONFIGURASI HALAMAN
# =============================================
st.set_page_config(
    page_title="Analisis Sentimen Honor of King",
    page_icon="🎮",
    layout="wide"
)

# =============================================
# CSS STYLING
# =============================================
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Orbitron:wght@700&family=Syne:wght@400;600&display=swap');

    html, body, [class*="css"] {
        font-family: 'Syne', sans-serif;
    }
    .main-header {
        font-family: 'Orbitron', sans-serif;
        font-size: 2.2rem;
        font-weight: 700;
        color: #F0C040;
        text-align: center;
        letter-spacing: 2px;
        padding: 1rem 0 0.2rem 0;
    }
    .sub-header {
        text-align: center;
        color: #aaa;
        font-size: 0.95rem;
        margin-bottom: 2rem;
    }
    .metric-card {
        background: #1a1a2e;
        border: 1px solid #F0C040;
        border-radius: 12px;
        padding: 1.2rem;
        text-align: center;
    }
    .metric-label {
        color: #aaa;
        font-size: 0.8rem;
        letter-spacing: 1px;
        text-transform: uppercase;
    }
    .metric-value {
        color: #F0C040;
        font-size: 2rem;
        font-weight: 700;
        font-family: 'Orbitron', sans-serif;
    }
    .positif-badge { background:#1a3a1a; color:#4CAF50; border-radius:8px; padding:2px 10px; font-weight:600; }
    .netral-badge  { background:#2a2a1a; color:#FFC107; border-radius:8px; padding:2px 10px; font-weight:600; }
    .negatif-badge { background:#3a1a1a; color:#F44336; border-radius:8px; padding:2px 10px; font-weight:600; }

    .stSelectbox label, .stFileUploader label { color: #F0C040 !important; font-weight: 600; }
    .stTabs [data-baseweb="tab"] { color: #aaa; font-family: 'Syne', sans-serif; }
    .stTabs [aria-selected="true"] { color: #F0C040 !important; border-bottom: 2px solid #F0C040; }
    .block-container { padding-top: 2rem; }
</style>
""", unsafe_allow_html=True)

# =============================================
# INISIALISASI TOOLS PREPROCESSING
# =============================================
@st.cache_resource
def load_preprocessing_tools():
    nltk.download('punkt', quiet=True)
    nltk.download('punkt_tab', quiet=True)
    factory_stem = StemmerFactory()
    stemmer = factory_stem.create_stemmer()
    factory_stop = StopWordRemoverFactory()
    stopword_remover = factory_stop.create_stop_word_remover()
    return stemmer, stopword_remover

@st.cache_resource
def load_model(path):
    with open(path, 'rb') as f:
        return pickle.load(f)

# Kamus Normalisasi (sama persis dengan Colab)
norm_dict = {
    'yg': 'yang', 'gk': 'tidak', 'gak': 'tidak', 'ga': 'tidak',
    'bgt': 'banget', 'dlm': 'dalam', 'sy': 'saya', 'aku': 'saya',
    'tdk': 'tidak', 'jgn': 'jangan', 'utk': 'untuk', 'sdh': 'sudah',
    'udh': 'sudah', 'kalo': 'kalau', 'kl': 'kalau', 'kpn': 'kapan',
    'dgn': 'dengan', 'krn': 'karena', 'tpy': 'tapi', 'tp': 'tapi',
    'mw': 'mau', 'tau': 'tahu', 'km': 'kamu', 'blm': 'belum'
}

def normalisasi_kata(text):
    words = text.split()
    normalized_words = [norm_dict.get(word, word) for word in words]
    return ' '.join(normalized_words)

def full_preprocessing(text, stemmer, stopword_remover):
    text = str(text).lower()
    text = re.sub(r'http\S+', '', text)
    text = re.sub(r'@\w+', '', text)
    text = re.sub(r'#\w+', '', text)
    text = re.sub(r'[^a-z\s]', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    text = normalisasi_kata(text)
    text = stopword_remover.remove(text)
    text = stemmer.stem(text)
    tokens = word_tokenize(text)
    return ' '.join(tokens)

def badge_sentimen(label):
    if label == 'Positif':
        return f'<span class="positif-badge">✅ Positif</span>'
    elif label == 'Netral':
        return f'<span class="netral-badge">⚪ Netral</span>'
    else:
        return f'<span class="negatif-badge">❌ Negatif</span>'

# =============================================
# HEADER
# =============================================
st.markdown('<div class="main-header">🎮 ANALISIS SENTIMEN HONOR OF KING</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-header">Klasifikasi Komentar TikTok • Naive Bayes • SVM • Random Forest</div>', unsafe_allow_html=True)
st.divider()

# =============================================
# SIDEBAR
# =============================================
with st.sidebar:
    st.markdown("### ⚙️ Pengaturan")
    model_choice = st.selectbox(
        "Pilih Model Klasifikasi",
        ["SVM (Linear)", "Naive Bayes", "Random Forest"]
    )

    st.markdown("---")
    st.markdown("### 📂 Upload File")
    uploaded_file = st.file_uploader("Upload file CSV", type=["csv"])

    st.markdown("---")
    st.markdown("### ℹ️ Info Model")
    info = {
        "SVM (Linear)":    ("66.83%", "Akurasi Tertinggi 🏆"),
        "Naive Bayes":     ("61.81%", "Paling Cepat ⚡"),
        "Random Forest":   ("62.81%", "Paling Stabil 🌲"),
    }
    acc, desc = info[model_choice]
    st.metric("Akurasi (Data Penelitian)", acc)
    st.caption(desc)

    st.markdown("---")
    st.caption("Skripsi: Bimo Wijaya K.P.P.S\nUniversitas Ahmad Dahlan • 2026")

# =============================================
# MAIN CONTENT
# =============================================
if uploaded_file is None:
    st.info("👈 Upload file CSV terlebih dahulu di sidebar untuk memulai analisis.")
    st.markdown("""
    **Format CSV yang dibutuhkan:**
    - File CSV boleh memiliki nama kolom **apapun**
    - Setelah upload, kamu bisa **pilih sendiri** kolom mana yang berisi komentar
    
    **Contoh:**
    | text | uid |
    |------|-----|
    | Game ini sangat seru dan menyenangkan! | 001 |
    | Server sering lag dan tidak stabil | 002 |
    | Biasa saja, tidak ada yang istimewa | 003 |
    """)
    st.stop()

# =============================================
# LOAD FILE & MODEL
# =============================================
try:
    df_input = pd.read_csv(uploaded_file)
except Exception as e:
    st.error(f"❌ Gagal membaca file CSV: {e}")
    st.stop()

# Tampilkan preview & pilih kolom
st.markdown("### 👀 Preview File CSV")
st.dataframe(df_input.head(3), use_container_width=True)

col_pilih, _ = st.columns([1, 2])
with col_pilih:
    kolom_teks = st.selectbox(
        "📌 Pilih kolom yang berisi komentar/teks:",
        options=df_input.columns.tolist(),
        index=0
    )

if df_input[kolom_teks].dropna().empty:
    st.error(f"❌ Kolom `{kolom_teks}` kosong. Pilih kolom lain.")
    st.stop()

st.success(f"✅ Kolom **`{kolom_teks}`** dipilih sebagai input teks. Klik tombol di bawah untuk mulai analisis.")

if not st.button("🚀 Mulai Analisis Sentimen"):
    st.stop()

# Load tools
stemmer, stopword_remover = load_preprocessing_tools()

# Load model & vectorizer
MODEL_PATHS = {
    "SVM (Linear)":  "svm_model_linear.pkl",
    "Naive Bayes":   "naive_bayes_model.pkl",
    "Random Forest": "random_forest_model.pkl",
}
TFIDF_PATH = "tfidf_model.pkl"

try:
    model = load_model(MODEL_PATHS[model_choice])
    tfidf  = load_model(TFIDF_PATH)
except FileNotFoundError as e:
    st.error(f"❌ File model tidak ditemukan: `{e.filename}`. Pastikan file .pkl ada di folder yang sama dengan app.py")
    st.stop()

# =============================================
# PROSES PREDIKSI
# =============================================
with st.spinner("⏳ Sedang memproses... (Preprocessing + Prediksi)"):
    df_result = df_input[[kolom_teks]].dropna().copy()
    df_result = df_result.rename(columns={kolom_teks: 'text'})
    df_result['text_bersih'] = df_result['text'].apply(
        lambda x: full_preprocessing(x, stemmer, stopword_remover)
    )
    X_tfidf = tfidf.transform(df_result['text_bersih'])
    df_result['Sentimen'] = model.predict(X_tfidf)

st.success(f"✅ Selesai! {len(df_result)} komentar berhasil dianalisis menggunakan **{model_choice}**")

# =============================================
# STATISTIK RINGKAS
# =============================================
counts = df_result['Sentimen'].value_counts()
total  = len(df_result)
positif = counts.get('Positif', 0)
netral  = counts.get('Netral', 0)
negatif = counts.get('Negatif', 0)

c1, c2, c3, c4 = st.columns(4)
with c1:
    st.markdown(f'<div class="metric-card"><div class="metric-label">Total Data</div><div class="metric-value">{total}</div></div>', unsafe_allow_html=True)
with c2:
    st.markdown(f'<div class="metric-card"><div class="metric-label">✅ Positif</div><div class="metric-value" style="color:#4CAF50">{positif}</div></div>', unsafe_allow_html=True)
with c3:
    st.markdown(f'<div class="metric-card"><div class="metric-label">⚪ Netral</div><div class="metric-value" style="color:#FFC107">{netral}</div></div>', unsafe_allow_html=True)
with c4:
    st.markdown(f'<div class="metric-card"><div class="metric-label">❌ Negatif</div><div class="metric-value" style="color:#F44336">{negatif}</div></div>', unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

# =============================================
# TABS OUTPUT
# =============================================
tab1, tab2, tab3 = st.tabs(["📋 Hasil Prediksi", "📊 Visualisasi", "☁️ Word Cloud"])

# --- TAB 1: TABEL HASIL ---
with tab1:
    st.markdown(f"### Hasil Klasifikasi ({model_choice})")

    df_display = df_result[['text', 'Sentimen']].copy()
    df_display['Sentimen_Badge'] = df_display['Sentimen'].apply(badge_sentimen)

    # Filter
    filter_col1, filter_col2 = st.columns([2, 1])
    with filter_col1:
        search = st.text_input("🔍 Cari teks komentar", "")
    with filter_col2:
        filter_label = st.selectbox("Filter Sentimen", ["Semua", "Positif", "Netral", "Negatif"])

    df_filtered = df_display.copy()
    if search:
        df_filtered = df_filtered[df_filtered['text'].str.contains(search, case=False, na=False)]
    if filter_label != "Semua":
        df_filtered = df_filtered[df_filtered['Sentimen'] == filter_label]

    st.markdown(df_filtered[['text', 'Sentimen_Badge']].rename(columns={'text': 'Komentar', 'Sentimen_Badge': 'Sentimen'}).to_html(escape=False, index=False), unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    csv_out = df_result[['text', 'text_bersih', 'Sentimen']].to_csv(index=False).encode('utf-8')
    st.download_button("⬇️ Download Hasil CSV", data=csv_out, file_name=f"hasil_sentimen_{model_choice.replace(' ','_')}.csv", mime='text/csv')

# --- TAB 2: VISUALISASI ---
with tab2:
    st.markdown("### Distribusi Sentimen")

    v1, v2 = st.columns(2)

    with v1:
        fig1, ax1 = plt.subplots(figsize=(5, 5), facecolor='#0e0e1a')
        ax1.set_facecolor('#0e0e1a')
        colors = ['#4CAF50', '#FFC107', '#F44336']
        labels_pie = [f"{k}\n{v}" for k, v in counts.items()]
        wedges, texts, autotexts = ax1.pie(
            counts.values,
            labels=labels_pie,
            autopct='%1.1f%%',
            colors=colors[:len(counts)],
            startangle=140,
            textprops={'color': 'white', 'fontsize': 11}
        )
        for at in autotexts:
            at.set_color('white')
            at.set_fontweight('bold')
        ax1.set_title('Pie Chart Sentimen', color='#F0C040', fontsize=13, fontweight='bold')
        st.pyplot(fig1)
        plt.close()

    with v2:
        fig2, ax2 = plt.subplots(figsize=(5, 5), facecolor='#0e0e1a')
        ax2.set_facecolor('#0e0e1a')
        bar_colors = {'Positif': '#4CAF50', 'Netral': '#FFC107', 'Negatif': '#F44336'}
        bar_c = [bar_colors.get(k, '#aaa') for k in counts.index]
        bars = ax2.bar(counts.index, counts.values, color=bar_c, edgecolor='none', width=0.5)
        for bar in bars:
            ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                     str(int(bar.get_height())), ha='center', va='bottom', color='white', fontweight='bold')
        ax2.set_title('Bar Chart Sentimen', color='#F0C040', fontsize=13, fontweight='bold')
        ax2.set_xlabel('Sentimen', color='#aaa')
        ax2.set_ylabel('Jumlah', color='#aaa')
        ax2.tick_params(colors='white')
        for spine in ax2.spines.values():
            spine.set_edgecolor('#333')
        ax2.set_facecolor('#0e0e1a')
        st.pyplot(fig2)
        plt.close()

# --- TAB 3: WORD CLOUD ---
with tab3:
    st.markdown("### Word Cloud per Sentimen")

    sentimen_list = ['Positif', 'Netral', 'Negatif']
    wc_cols = st.columns(3)
    wc_colors = {'Positif': 'Greens', 'Netral': 'YlOrBr', 'Negatif': 'Reds'}

    for i, sent in enumerate(sentimen_list):
        with wc_cols[i]:
            teks = ' '.join(df_result[df_result['Sentimen'] == sent]['text_bersih'])
            if teks.strip():
                wc = WordCloud(
                    width=500, height=350,
                    background_color='#0e0e1a',
                    colormap=wc_colors[sent],
                    min_font_size=10
                ).generate(teks)
                fig_wc, ax_wc = plt.subplots(figsize=(4, 3), facecolor='#0e0e1a')
                ax_wc.imshow(wc, interpolation='bilinear')
                ax_wc.axis('off')
                ax_wc.set_title(f'{sent}', color='white', fontsize=12, fontweight='bold')
                st.pyplot(fig_wc)
                plt.close()
            else:
                st.info(f"Tidak ada data sentimen {sent}")
