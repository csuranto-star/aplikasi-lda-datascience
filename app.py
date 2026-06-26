import streamlit as st
import pandas as pd
import numpy as np
import nltk
import re
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import LatentDirichletAllocation
import networkx as nx
import matplotlib.pyplot as plt
import plotly.express as px
from collections import Counter
import os

# 1. KONFIGURASI HALAMAN UTAMA DASHBOARD
st.set_page_config(
    page_title="Temporal Bibliometric & Topic Modeling Dashboard",
    page_icon="📊",
    layout="wide"
)

# 2. MEKANISME UNDUHAN RESOURCE NLTK SECARA AMAN DI SERVER CLOUD
@st.cache_resource
def load_nltk_safely():
    try:
        # Membuat folder nltk_data temporer di server jika belum ada
        nltk_data_dir = os.path.join(os.path.expanduser('~'), 'nltk_data')
        if not os.path.exists(nltk_data_dir):
            os.makedirs(nltk_data_dir)
        nltk.data.path.append(nltk_data_dir)
        
        # Mengunduh secara paksa paket dasar yang dibutuhkan pipeline NLP
        nltk.download('punkt', download_dir=nltk_data_dir, quiet=True)
        nltk.download('punkt_tab', download_dir=nltk_data_dir, quiet=True)
        nltk.download('stopwords', download_dir=nltk_data_dir, quiet=True)
    except Exception as e:
        pass

load_nltk_safely()

# 3. PIPELINE NLP: CLEANING, CASE FOLDING, & STOPWORDS
try:
    stop_words = set(stopwords.words('english'))
except:
    stop_words = set()

custom_stops = {'paper', 'research', 'study', 'using', 'based', 'results', 'analysis', 'system', 'method', 'approach', 'data', 'science'}
stop_words.update(custom_stops)

def clean_and_tokenize(text):
    if pd.isna(text) or text == "":
        return ""
    # Tahap a: Cleaning & Case Folding (Menggunakan Regex)
    text_cleaned = re.sub(r'[^a-zA-Z\s]', '', str(text)).lower()
    # Tahap b: Tokenization & Stopwords Removal
    tokens = text_cleaned.split()
    tokens_filtered = [w for w in tokens if w not in stop_words and len(w) > 3]
    return " ".join(tokens_filtered)

# --- UI HEADER ---
st.title("📊 Dasbor Analisis Temporal Riset Data Science")
st.markdown("Deteksi Evolusi dan Pergeseran Tren Kata Kunci Riset Data Science melalui Analisis Bibliometrik dan Topic Modeling (LDA).")
st.divider()

# --- SIDEBAR: INPUT DATASET ---
st.sidebar.header("📁 [Metode 1] Input Dataset")
uploaded_file = st.sidebar.file_uploader("Unggah Dataset CSV (Harzing's Publish or Perish):", type=["csv"])

if uploaded_file is not None:
    try:
        df = pd.read_csv(uploaded_file)
        # Normalisasi nama kolom secara otomatis
        mapping = {
            'title': 'Title', 'TITLE': 'Title',
            'abstract': 'Abstract', 'ABSTRACT': 'Abstract',
            'year': 'Year', 'YEAR': 'Year'
        }
        for k, v in mapping.items():
            if k in df.columns:
                df.rename(columns={k: v}, inplace=True)
        st.sidebar.success("Dataset berhasil diunggah!")
    except Exception as e:
        st.sidebar.error(f"Gagal membaca file CSV: {e}")
        st.stop()
else:
    st.sidebar.warning("Silakan unggah file CSV Semantic Scholar Anda untuk memulai.")
    st.stop()

# Validasi kolom minimum
if not {'Title', 'Abstract', 'Year'}.issubset(df.columns):
    st.error("Dataset harus memiliki kolom minimum: Title, Abstract, dan Year!")
    st.stop()

# --- SIDEBAR: TEMPORAL SET ---
st.sidebar.header("⏳ [Metode 3] Temporal Set")
df['Year'] = pd.to_numeric(df['Year'], errors='coerce').fillna(2020).astype(int)
min_year = int(df['Year'].min())
max_year = int(df['Year'].max())
if min_year == max_year:
    max_year += 1

split_year = st.sidebar.slider("Tentukan Tahun Pemisah Era:", min_value=min_year, max_value=max_year, value=int((min_year + max_year) / 2))
num_topics = st.sidebar.slider("Jumlah Topik Laten per Era (LDA)", min_value=2, max_value=5, value=3)

# --- EKSEKUSI PIPELINE NLP ---
df['Abstract'] = df['Abstract'].fillna("")
df['Cleaned_Abstract_String'] = df['Abstract'].apply(clean_and_tokenize)
df['Keywords'] = df['Title'].apply(clean_and_tokenize)
df['Era'] = np.where(df['Year'] < split_year, f"Era Klasik (< {split_year})", f"Era Modern (≥ {split_year})")

# =========================================================================
# [DIAGRAM 4] PROSES GROUPING & DATA EKSPOR
# =========================================================================
st.subheader("📥 Hasil Grouping & Data Ekspor")

semua_kata = []
for doc in df['Cleaned_Abstract_String']:
    semua_kata.extend(doc.split())
hitung_kata = Counter(semua_kata)
df_kata = pd.DataFrame(hitung_kata.items(), columns=['Kata', 'Total_Frekuensi']).sort_values(by='Total_Frekuensi', ascending=False)

top50_kata = df_kata['Kata'].head(50).tolist()
list_tren_lokal = []
daftar_tahun_lokal = sorted(df['Year'].unique())

for thn in daftar_tahun_lokal:
    df_thn = df[df['Year'] == thn]
    kata_per_tahun = []
    for doc in df_thn['Cleaned_Abstract_String']:
        kata_per_tahun.extend(doc.split())
    for kata_target in top50_kata:
        frek = kata_per_tahun.count(kata_target)
        list_tren_lokal.append({'Tahun': int(thn), 'Kata Kunci': kata_target, 'Frekuensi': frek})
        
df_tren_lokal = pd.DataFrame(list_tren_lokal)

# Menyimpan ke file lokal di server
try:
    df_kata.to_csv("data_kata.csv", index=False, encoding='utf-8')
    df_tren_lokal.to_csv("data_tren.csv", index=False, encoding='utf-8')
except:
    pass

col_dl1, col_dl2 = st.columns(2)
with col_dl1:
    st.download_button("💾 Unduh Hasil Grouping: data_kata.csv", data=df_kata.to_csv(index=False).encode('utf-8'), file_name="data_kata.csv", mime="text/csv")
with col_dl2:
    st.download_button("💾 Unduh Hasil Grouping: data_tren.csv", data=df_tren_lokal.to_csv(index=False).encode('utf-8'), file_name="data_tren.csv", mime="text/csv")

st.divider()

# =========================================================================
# VISUALISASI DATA TREND
# =========================================================================
st.subheader("📉 Visualisasi Data Trend")
pilihan_istilah = df_kata['Kata'].head(40).tolist()
if len(pilihan_istilah) > 0:
    kata_kunci_terpilih = st.multiselect("Pilih Istilah Kata Kunci untuk Dilihat Tren Evolusinya:", options=pilihan_istilah, default=[pilihan_istilah[0]] if len(pilihan_istilah) > 0 else [])
    if kata_kunci_terpilih:
        df_visual_tren = df_tren_lokal[df_tren_lokal['Kata Kunci'].isin(kata_kunci_terpilih)]
        fig_tren_line = px.line(df_visual_tren, x='Tahun', y='Frekuensi', color='Kata Kunci', markers=True, title="Grafik Lini Waktu Evolusi Tren Kata Kunci", template="plotly_white")
        fig_tren_line.update_layout(xaxis=dict(tickmode='linear', dtick=1))
        st.plotly_chart(fig_tren_line, use_container_width=True)
else:
    st.warning("Kata kunci tidak ditemukan dalam teks abstrak.")

st.divider()

# =========================================================================
# MODEL UTAMA: TOPIC MODELING (LDA) DENGAN TF-IDF
# =========================================================================
st.subheader("🤖 Pemodelan Topik Laten (LDA) dengan Ekstraksi Fitur TF-IDF")

col_era1, col_era2 = st.columns(2)
eras = sorted(df['Era'].unique())

for i, era in enumerate(eras):
    era_df = df[df['Era'] == era]
    with (col_era1 if i == 0 else col_era2):
        st.info(f"📌 {era} (Total: {len(era_df)} Artikel)")
        valid_docs = [doc for doc in era_df['Cleaned_Abstract_String'].tolist() if len(doc.strip()) > 0]
        
        if len(valid_docs) >= 2:
            try:
                tfidf_vectorizer = TfidfVectorizer(ngram_range=(1, 2), max_df=0.95, min_df=1)
                tfidf_matrix = tfidf_vectorizer.fit_transform(valid_docs)
                
                lda_model = LatentDirichletAllocation(n_components=min(num_topics, len(valid_docs)), random_state=42, max_iter=5)
                lda_model.fit(tfidf_matrix)
                feature_names = tfidf_vectorizer.get_feature_names_out()
                
                for idx, topic in enumerate(lda_model.components_):
                    top_words_idx = topic.argsort()[:-6:-1]
                    words = ", ".join([feature_names[index].replace(" ", "_") for index in top_words_idx])
                    st.markdown(f"**Kluster Topik #{idx+1}** : `{words}`")
            except Exception as e:
                st.write(f"Gagal memproses model topik: {e}")
        else:
            st.write("Jumlah teks dokumen tidak mencukupi untuk pemodelan di era ini.")

st.divider()
st.subheader("🕸️ Jaringan Hubungan Kata Jurnal (Era Modern)")
df_modern = df[df['Era'] == eras[-1]] if len(eras) > 0 else df

if len(df_modern) > 0:
    try:
        G = nx.Graph()
        for keywords in df_modern.head(30)['Keywords'].dropna():
            kw_list = [k.strip() for k in keywords.split() if len(k.strip()) > 3]
            for m in range(len(kw_list)):
                for n in range(m + 1, len(kw_list)):
                    G.add_edge(kw_list[m], kw_list[n])
                    
        if len(G.nodes) > 0:
            fig, ax = plt.subplots(figsize=(12, 5))
            pos = nx.spring_layout(G, k=0.5, seed=42)
            nx.draw(G, pos, ax=ax, with_labels=True, node_color='#0284c7', node_size=150, edge_color='#cbd5e1', font_size=7, font_weight='bold')
            st.pyplot(fig)
        else:
            st.write("Relasi antar kata tidak cukup untuk membentuk jaringan graf.")
    except Exception as e:
        st.write(f"Gagal merender grafik jaringan: {e}")
else:
    st.write("Data Era Modern tidak tersedia.")
