from flask import Flask, request, jsonify
from flask_cors import CORS 
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

app = Flask(__name__)
CORS(app) # Mengizinkan HTML mengambil data

# --- MASUKKAN KODE DARI IPYNB ANDA DI SINI ---
print("Memuat dan menggabungkan dataset...")

# Membaca ketiga file CSV dengan nama kolom yang tepat
df_steam = pd.read_csv('Data/steam_horror.csv')[['appid', 'name', 'developer', 'genres', 'release_date']]
df_desc = pd.read_csv('Data/steam_desc_horror.csv')[['steam_appid', 'short_description']]
df_media = pd.read_csv('Data/steam_media_horror.csv')[['steam_appid', 'header_image']]

# Menyamakan nama kolom menjadi 'appid' agar bisa digabungkan
df_desc = df_desc.rename(columns={'steam_appid': 'appid'})
df_media = df_media.rename(columns={'steam_appid': 'appid'})

# Menggabungkan ketiga dataframe berdasarkan 'appid'
df = pd.merge(df_steam, df_desc, on='appid')
df = pd.merge(df, df_media, on='appid')

# Membersihkan data yang tidak memiliki deskripsi
df = df.dropna(subset=['short_description'])

# Membatasi 2000 data agar komputasi memori ringan (Bisa dinaikkan jika PC mumpuni)
df = df.head(1000).reset_index(drop=True) 

# Proses NLP menggunakan short_description
tfidf = TfidfVectorizer(stop_words='english', max_features=1000)
tfidf_matrix = tfidf.fit_transform(df['short_description'])
cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)

# Indeks pencarian menggunakan nama game (diubah ke huruf kecil semua agar mudah dicari)
indices = pd.Series(df.index, index=df['name'].str.lower()).drop_duplicates()
print("Sistem siap digunakan!")
# ---------------------------------------------

@app.route('/recommend', methods=['GET'])
def recommend():
    # Mengambil kata kunci pencarian
    kata_kunci = request.args.get('game', '').lower()
    
    # MENCARI SEBAGIAN KATA: Mencari game yang judulnya "mengandung" kata kunci yang diketik
    # na=False mencegah error jika ada judul game yang kosong (NaN)
    pencarian = df[df['name'].str.lower().str.contains(kata_kunci, na=False, regex=False)]
    
    if pencarian.empty:
        return jsonify({"error": f"Game yang mengandung kata '{kata_kunci}' tidak ditemukan"}), 404
        
    # Jika ditemukan banyak game dengan kata tersebut (misal "Batman"), ambil urutan paling atas
    idx = pencarian.index[0]
    
    sim_scores = list(enumerate(cosine_sim[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)[1:7]
    game_indices = [i[0] for i in sim_scores]
    
    hasil = df[['name', 'header_image', 'developer', 'genres', 'release_date', 'short_description']].iloc[game_indices].to_dict(orient='records')
    return jsonify(hasil)

if __name__ == '__main__':
    app.run(port=5000, debug=True)