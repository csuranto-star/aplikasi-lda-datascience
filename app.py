import streamlit as st
import pandas as pd
import numpy as np
import nltk
import re
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from gensim import corpora, models
from gensim.models import Phrases
from gensim.models.phrases import Phraser
import networkx as nx
import matplotlib.pyplot as plt
import plotly.express as px
from collections import Counter

# 1. KONFIGURASI HALAMAN UTAMA DASHBOARD
st.set_page_config(
    page_title="Temporal Bibliometric & Topic Modeling Dashboard",
    page_icon="📊",
    layout="wide"
)

# 2. DOWNLOAD & CACHING RESOURCE NLTK
@st.cache_resource
def download_nltk_resources():
    nltk.download('punkt')
    nltk.download('punkt_tab')
    nltk.download('stopwords')

download_nltk_resources()

# 3. IMPLEMENTASI METODOLOGI NLP SESUAI DOKUMEN PENELITIAN
stop_words = set(stopwords.words('english'))
# Menambahkan kata umum akademik non-spesifik ke dalam daftar stopwords
custom_stops = {'paper', 'research', 'study', 'using', 'based', 'results', 'analysis', 'system', 'method', 'approach'}
stop_words.update(custom_stops)

def preprocess_text(text):
    if pd.isna(text) or text == "":
        return []
    
    # Tahap a: Cleaning & Case Folding (Menghapus tanda baca/angka & mengecilkan huruf)
    text_cleaned = re.sub(r'[^a-zA-Z\s]', '', str(text))
    text_lowercased = text_cleaned.lower()
    
    # Tahap b: Tokenization (Memecah kalimat menjadi kata tunggal)
    tokens = word_tokenize(text_lowercased)
    
    # Tahap b: Stopwords Removal & membuang kata pendek (< 4 huruf)
    tokens_filtered = [w for w in tokens if w not in stop_words and len(w) > 3]
    return tokens_filtered

# --- UI HEADER ---
st.title("📊 Dasbor Analisis Temporal Riset Data Science")
st.markdown("Deteksi Evolusi dan Pergeseran Tren Kata Kunci Riset Data Science melalui Analisis Bibliometrik dan Topic Modeling (LDA).")
st.divider()

# --- SIDEBAR: INPUT DATASET ---
st.sidebar.header("📁Input Dataset")
uploaded_file = st.sidebar.file_uploader("Unggah Dataset CSV (Harzing's Publish or Perish):", type=["csv"])

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    
    # Mapping otomatis header kolom dari Publish or Perish
    mapping = {
        'title': 'Title', 'TITLE': 'Title',
        'abstract': 'Abstract', 'ABSTRACT': 'Abstract',
        'year': 'Year', 'YEAR': 'Year'
    }
    for scopus_col, app_col in mapping.items():
        if scopus_col in df.columns:
            df.rename(columns={scopus_col: app_col}, inplace=True)
            
    st.sidebar.success("Dataset berhasil diunggah!")
else:
    st.sidebar.warning("Silakan unggah file CSV Semantic Scholar Anda untuk memulai.")
    st.stop()

# --- SIDEBAR: TEMPORAL SET ---
st.sidebar.header("⏳Temporal Set")
min_year = int(df['Year'].min())
max_year = int(df['Year'].max())
split_year = st.sidebar.slider(
    "Tentukan Tahun Pemisah Era:",
    min_value=min_year + 1,
    max_value=max_year,
    value=int((min_year + max_year) / 2)
)

st.sidebar.header("⚙️ Parameter Topic Modeling")
num_topics = st.sidebar.slider("Jumlah Topik Laten per Era (LDA)", min_value=2, max_value=5, value=3)

# --- EKSEKUSI PIPELINE NLP SESUAI STRUKTUR DOKUMEN ---
df['Abstract'] = df['Abstract'].fillna("")

# Menjalankan Tahap a & b secara massal
tokenized_abstracts = df['Abstract'].apply(preprocess_text).tolist()

# Tahap c: Phrase Identification (Membentuk model Bigram secara otomatis)
# Ini akan mendeteksi kata yang sering muncul berdampingan seperti data_science, deep_learning, machine_learning
bigram_data = Phrases(tokenized_abstracts, min_count=3, threshold=10)
bigram_phraser = Phraser(bigram_data)

# Menyimpan hasil akhir teks yang sudah melewati Phrase Identification ke dataframe
df['Cleaned_Abstract'] = [bigram_phraser[doc] for doc in tokenized_abstracts]

# Fallback untuk Keywords Jaringan Bibliometrik
if 'Keywords' not in df.columns or df['Keywords'].isna().all():
    df['Keywords'] = df['Title'].apply(lambda x: "; ".join(bigram_phraser[preprocess_text(str(x))]))

# Membagi Era secara Temporal
df['Era'] = np.where(df['Year'] < split_year, f"Era Klasik (< {split_year})", f"Era Modern (≥ {split_year})")


# =========================================================================
# [DIAGRAM 4] PROSES GROUPING & DATA EKSPOR
# =========================================================================
st.subheader("📥 Hasil Grouping & Data Ekspor")

# A. Grouping Data Kata (Mencakup kata tunggal dan frasa bigram)
semua_kata_korpus = []
for list_kata in df['Cleaned_Abstract']:
    semua_kata_korpus.extend(list_kata)
    
hitung_kata = Counter(semua_kata_korpus)
df_kata = pd.DataFrame(hitung_kata.items(), columns=['Kata', 'Total_Frekuensi']).sort_values(by='Total_Frekuensi', ascending=False)

# B. Grouping Data Tren Temporal (Top 50 Kata/Frasa)
top50_kata = df_kata['Kata'].head(50).tolist()
list_tren_lokal = []
daftar_tahun_lokal = sorted(df['Year'].dropna().unique())

for thn in daftar_tahun_lokal:
    df_thn = df[df['Year'] == thn]
    kata_per_tahun = []
    for kumpulan_kata in df_thn['Cleaned_Abstract']:
        kata_per_tahun.extend(kumpulan_kata)
        
    for kata_target in top50_kata:
        frek = kata_per_tahun.count(kata_target)
        list_tren_lokal.append({'Tahun': int(thn), 'Kata Kunci': kata_target, 'Frekuensi': frek})
        
df_tren_lokal = pd.DataFrame(list_tren_lokal)

# Menyimpan ke folder internal secara otomatis
df_kata.to_csv("data_kata.csv", index=False, encoding='utf-8')
df_tren_lokal.to_csv("data_tren.csv", index=False, encoding='utf-8')

# Menampilkan tombol unduh fisik kepada user
col_dl1, col_dl2 = st.columns(2)
with col_dl1:
    st.download_button(
        label="💾 Unduh Hasil Grouping: data_kata.csv",
        data=df_kata.to_csv(index=False).encode('utf-8'),
        file_name="data_kata.csv",
        mime="text/csv"
    )
with col_dl2:
    st.download_button(
        label="💾 Unduh Hasil Grouping: data_tren.csv",
        data=df_tren_lokal.to_csv(index=False).encode('utf-8'),
        file_name="data_tren.csv",
        mime="text/csv"
    )

st.divider()

# =========================================================================
# VISUALISASI DATA TREND
# =========================================================================
st.subheader("📉 Visualisasi Data Trend")
pilihan_istilah = df_kata['Kata'].head(40).tolist()
kata_kunci_terpilih = st.multiselect(
    "Pilih Istilah Kata Kunci/Frasa untuk Dilihat Tren Evolusinya:",
    options=pilihan_istilah,
    default=[pilihan_istilah[0], pilihan_istilah[1]] if len(pilihan_istilah) > 1 else pilihan_istilah
)

if kata_kunci_terpilih:
    df_visual_tren = df_tren_lokal[df_tren_lokal['Kata Kunci'].isin(kata_kunci_terpilih)]
    fig_tren_line = px.line(
        df_visual_tren, x='Tahun', y='Frekuensi', color='Kata Kunci', markers=True,
        title="Grafik Lini Waktu Evolusi Tren Kata Kunci/Frasa (NLP Model)",
        labels={'Frekuensi': 'Jumlah Kemunculan', 'Tahun': 'Tahun'},
        template="plotly_white"
    )
    fig_tren_line.update_layout(xaxis=dict(tickmode='linear', dtick=1))
    st.plotly_chart(fig_tren_line, use_container_width=True)

st.divider()

# =========================================================================
# MODEL UTAMA: TOPIC MODELING (LDA) DENGAN PEMBOBOTAN TF-IDF (TAHAP D)
# =========================================================================
st.subheader("🤖 Pemodelan Topik Laten (LDA) dengan Ekstraksi Fitur TF-IDF")
st.markdown("Sistem menerapkan pembobotan untuk mengurangi kata-kata umum sebelum melatih model LDA.")

col_era1, col_era2 = st.columns(2)
eras = sorted(df['Era'].unique())

for i, era in enumerate(eras):
    era_df = df[df['Era'] == era]
    with (col_era1 if i == 0 else col_era2):
        st.info(f"📌 {era}")
        valid_abstracts = [text for text in era_df['Cleaned_Abstract'].tolist() if len(text) > 0]
        
        if len(valid_abstracts) > 0:
            # Membuat Kamus teks
            dictionary = corpora.Dictionary(valid_abstracts)
            # Membuat korpus berbasis Bag-of-Words
            bow_corpus = [dictionary.doc2bow(text) for text in valid_abstracts]
            
            # --- MASUK TAHAP d: FEATURE EXTRACTION (TF-IDF) ---
            tfidf_model = models.TfidfModel(bow_corpus)
            tfidf_corpus = tfidf_model[bow_corpus] # Menerapkan pembobotan TF-IDF ke korpus dokumen
            
            # Melatih Model LDA menggunakan korpus TF-IDF yang sudah bersih dari derau kata umum
            lda_model = models.LdaModel(corpus=tfidf_corpus, id2word=dictionary, num_topics=num_topics, random_state=42, passes=10)
            
            for idx, topic in lda_model.print_topics(-1):
                # Menampilkan frasa gabungan (seperti data_science) secara rapi dengan mengganti underscore (_) menjadi spasi jika diperlukan
                words = ", ".join([w.split("*")[1].replace('"', '').strip() for w in topic.split("+")][:5])
                st.markdown(f"**Kluster Topik #{idx+1}** : `{words}`")
        else:
            st.write("Data tidak cukup.")

st.divider()
st.subheader("🕸️ Jaringan Hubungan Kata Jurnal (Era Modern)")
df_modern = df[df['Era'] == eras[-1]]
if len(df_modern) > 0:
    G = nx.Graph()
    for keywords in df_modern.head(50)['Keywords'].dropna():
        kw_list = [k.strip() for k in keywords.split(';') if k.strip()]
        for m in range(len(kw_list)):
            for n in range(m + 1, len(kw_list)):
                G.add_edge(kw_list[m], kw_list[n])
                
    fig, ax = plt.subplots(figsize=(12, 5))
    pos = nx.spring_layout(G, k=0.4, seed=42)
    nx.draw(G, pos, ax=ax, with_labels=True, node_color='#0284c7', node_size=200, edge_color='#cbd5e1', font_size=8, font_weight='bold')
    st.pyplot(fig)