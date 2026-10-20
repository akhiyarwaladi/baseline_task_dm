# %% PANDUAN - Baca sebelum menjalankan tutorial
# DECISION TREE UNTUK KLASIFIKASI STATUS GIZI BALITA
#
# Google Colab: unggah 01_stunting_balita.csv melalui Files, lalu salin setiap
# blok "# %% TAHAP" ke satu sel kode. Jalankan TAHAP 1-13 berurutan.
# Komentar menjelaskan langkah; print() dan gambar menampilkan output.
# Menjalankan di komputer: python tutorial_decision_tree_stunting.py
# Jika import gagal di Colab: %pip install pandas scikit-learn matplotlib
#
# ALUR: baca CSV -> bersihkan -> siapkan X/y -> SPLIT TRAINING/TESTING
#       -> latih -> evaluasi -> baca pohon utama -> bandingkan -> prediksi.
# Split dilakukan SATU KALI pada TAHAP 5, sebelum pelatihan apa pun.
# Duplikat dibuang dahulu agar baris identik tidak masuk ke kedua bagian.
# Training dipakai fit() untuk belajar; testing dipakai untuk menguji prediksi.
# Semua percobaan memakai training dan testing yang SAMA.
#
# Jika mengubah sesuatu, jalankan ulang:
# - CSV / pembersihan / fitur: dari tahap yang diubah sampai TAHAP 13.
# - Pembagian training/testing: TAHAP 5-13.
# - Pengaturan model utama: TAHAP 6-13.
# - Daftar percobaan: TAHAP 9 atau 10 yang diubah.
# - Gambar model utama: TAHAP 8; gambar setiap max_depth: TAHAP 9.
# - Input satu balita: TAHAP 12-13 saja.
# Semua input prediksi dikumpulkan di TAHAP 12, tepat sebelum prediksi.
# Setelah runtime direset, unggah ulang CSV dan jalankan semua tahap lagi.
# Hasil gambar disimpan otomatis di folder gambar_decision_tree.
# Di komputer, tutup jendela grafik untuk melanjutkan tahap berikutnya.
# Tahap 9 mengekspor empat SVG lengkap; tunggu pesan progres sampai selesai.


# %% TAHAP 1 - Import library
# Tujuan: menyiapkan alat membaca data, melatih model, dan menampilkan hasil.
# Output: pesan bahwa library sudah siap.

from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

print("TAHAP 1 | Library siap.")


# %% TAHAP 2 - Membaca dan memeriksa dataset
# Tujuan: melihat kondisi CSV sebelum diolah.
# Output: ukuran data, lima contoh, tipe kolom, nilai kosong, dan duplikat.
# Input: umur (bulan), jenis kelamin, tinggi badan (cm). Target: Status Gizi.
# head() hanya menampilkan contoh; model nanti memakai seluruh data training.

lokasi_data = Path("01_stunting_balita.csv")  # File unggahan di Colab.
if not lokasi_data.is_file():
    lokasi_data = Path("dataset/01_stunting_balita.csv")  # File di proyek ini.
if not lokasi_data.is_file():
    raise FileNotFoundError("Unggah 01_stunting_balita.csv ke Files di Colab.")

data = pd.read_csv(lokasi_data)

print("TAHAP 2 | Baris:", len(data), "| Kolom:", len(data.columns))
print(data.head().to_string(index=False))
print("\nTipe setiap kolom:")
print(data.dtypes)
print("\nJumlah nilai kosong per kolom:")
print(data.isna().sum())
print("Jumlah duplikat:", data.duplicated().sum())


# %% TAHAP 3 - Membersihkan data sebelum split
# Tujuan: membuang baris kosong dan salinan baris identik di semua kolom.
# Output: jumlah yang dibuang, jumlah tersisa, dan jumlah baris tiap kelas.
# Duplikat dibuang SEBELUM split agar baris identik tidak masuk ke kedua bagian.
# CSV ini: 120999 awal - 0 kosong - 81574 duplikat = 39425 baris bersih.
# Tidak ada ID anak pada CSV; baris unik belum tentu berarti anak unik.

data_bersih = data.dropna().copy()
jumlah_kosong = len(data) - len(data_bersih)
jumlah_duplikat = data_bersih.duplicated().sum()
data_bersih = data_bersih.drop_duplicates().copy()

print("TAHAP 3 | Baris kosong yang dibuang:", jumlah_kosong)
print("Baris duplikat yang dibuang:", jumlah_duplikat)
print("Jumlah baris bersih:", len(data_bersih))
print("\nJumlah baris per Status Gizi:")
print(data_bersih["Status Gizi"].value_counts())


# %% TAHAP 4 - Menyiapkan X (fitur) dan y (target)
# Tujuan: memisahkan tiga input dari jawaban yang akan dipelajari.
# Output: contoh X dan y; jenis kelamin sudah berupa angka.
# Kode jenis kelamin: 0=laki-laki, 1=perempuan; angka ini hanya kode kategori.
# Status Gizi tidak masuk X karena itulah jawaban yang dicari. y tetap teks.
# Pemetaan 0/1 ditetapkan langsung, tanpa belajar statistik dari testing.
# Decision Tree tidak membutuhkan scaling/standardisasi.

kode_kelamin = {"laki-laki": 0, "perempuan": 1}
data_model = data_bersih.copy()
if not data_model["Jenis Kelamin"].isin(kode_kelamin).all():
    raise ValueError("Jenis Kelamin harus 'laki-laki' atau 'perempuan'.")
data_model["Jenis Kelamin"] = data_model["Jenis Kelamin"].map(kode_kelamin)

nama_fitur = ["Umur (bulan)", "Jenis Kelamin", "Tinggi Badan (cm)"]
X = data_model[nama_fitur]
y = data_model["Status Gizi"]

print("TAHAP 4 | X = tiga kolom input:")
print(X.head().to_string(index=False))
print("\ny = label untuk setiap baris X:")
print(y.head().to_string(index=False))


# %% TAHAP 5 - Split training dan testing SATU KALI
# Tujuan: membagi data SEBELUM model belajar; pembagian ini dipakai sampai akhir.
# Output: jumlah training/testing dan proporsi kelas pada keduanya.
# test_size=0.2 berarti testing 20%, sehingga training 80%.
# random_state membuat pengacakan dapat diulang untuk data/urutan yang sama.
# stratify=y menjaga proporsi kelas kurang lebih sama pada kedua bagian.
# Default CSV: 31540 baris training dan 7885 baris testing.

# PENGATURAN SPLIT: jika diubah, ulang TAHAP 5-13.
porsi_testing = 0.2  # Coba 0.3 untuk training 70% dan testing 30%.
seed_split = 42      # Coba 10 untuk mengubah anggota kedua bagian.

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=porsi_testing,
    random_state=seed_split,
    stratify=y,
)

print("TAHAP 5 | Jumlah TRAINING:", len(X_train))
print("Jumlah TESTING:", len(X_test))
print("\nProporsi kelas (%):")
print(pd.DataFrame({
    "Training": y_train.value_counts(normalize=True) * 100,
    "Testing": y_test.value_counts(normalize=True) * 100,
}).round(2))


# %% TAHAP 6 - Melatih model utama
# Tujuan: mempelajari aturan dari X_train dan y_train dengan fit().
# Output: pengaturan model, kedalaman sebenarnya, dan jumlah daun.
# Model utama bernama model; inilah yang digambar dan dipakai prediksi akhir.
# Model mempelajari pertanyaan dari fitur, lalu mengikuti cabang sampai daun.
# max_depth membatasi jumlah pemisahan dari akar ke daun; daun memberi prediksi.
# criterion menentukan ukuran memilih pemisahan: gini atau entropy.
# min_samples_split = minimal baris pada simpul agar boleh dipecah.
# min_samples_leaf = minimal baris yang harus tersisa di setiap daun.

# PENGATURAN MODEL UTAMA: jika diubah, ulang TAHAP 6-13.
kedalaman_pohon = 3          # max_depth: coba 5, 8, atau 12.
kriteria_pemisahan = "gini"  # criterion: coba "entropy".
minimal_split = 2           # min_samples_split: coba 200.
minimal_daun = 1            # min_samples_leaf: coba 50.
seed_model = 42             # Pertahankan saat membandingkan parameter lain.

model = DecisionTreeClassifier(
    max_depth=kedalaman_pohon,
    criterion=kriteria_pemisahan,
    min_samples_split=minimal_split,
    min_samples_leaf=minimal_daun,
    random_state=seed_model,
)
model.fit(X_train, y_train)

print("TAHAP 6 | Model utama selesai dilatih.")
print("max_depth:", kedalaman_pohon, "| criterion:", kriteria_pemisahan)
print("min_samples_split:", minimal_split, "| min_samples_leaf:", minimal_daun)
print("Kedalaman sebenarnya:", model.get_depth(), "| Jumlah daun:", model.get_n_leaves())


# %% TAHAP 7 - Mengukur akurasi TRAINING dan TESTING model utama
# Tujuan: membandingkan prediksi dengan label asli.
# Output: dua akurasi, laporan kelas, confusion matrix, dan lima contoh testing.
# Akurasi training: kecocokan pada data yang sudah dipelajari.
# Akurasi testing: prediksi pada data yang tidak digunakan dalam fit().
# Akurasi = prediksi benar dibagi jumlah baris yang dinilai.
# Precision: berapa bagian prediksi kelas A yang benar-benar A.
# Recall: berapa bagian data asli kelas A yang berhasil ditemukan.
# F1-score: gabungan precision/recall; support: jumlah data asli setiap kelas.
# Confusion matrix: baris=asli, kolom=prediksi; diagonal berarti benar.

prediksi_train = model.predict(X_train)
prediksi_test = model.predict(X_test)
akurasi_train = accuracy_score(y_train, prediksi_train)
akurasi_test = accuracy_score(y_test, prediksi_test)

print(f"TAHAP 7 | Akurasi TRAINING: {akurasi_train:.2%}")
print(f"Akurasi TESTING: {akurasi_test:.2%}")
print("Prediksi testing benar:", (prediksi_test == y_test).sum(), "dari", len(y_test))

# Pembanding sederhana: selalu menebak kelas terbanyak pada training.
kelas_terbanyak = y_train.mode()[0]
akurasi_pembanding = accuracy_score(y_test, [kelas_terbanyak] * len(y_test))
print(f"Jika selalu menebak '{kelas_terbanyak}': {akurasi_pembanding:.2%}")
print("\nLaporan per kelas pada TESTING:")
# zero_division=0 menampilkan 0 jika metrik suatu kelas tidak dapat dihitung.
print(classification_report(y_test, prediksi_test, zero_division=0))

matriks = confusion_matrix(y_test, prediksi_test, labels=model.classes_)
tabel_matriks = pd.DataFrame(matriks, index=model.classes_, columns=model.classes_)
tabel_matriks.index.name = "Asli"
tabel_matriks.columns.name = "Prediksi"
print("Confusion matrix TESTING:")
print(tabel_matriks.to_string())

contoh_hasil = X_test.head(5).copy()
contoh_hasil["Asli"] = y_test.head(5)
contoh_hasil["Prediksi"] = prediksi_test[:5]
print("\nLima contoh testing (jenis kelamin: 0=laki-laki, 1=perempuan):")
print(contoh_hasil.to_string(index=False))

# Default depth=3: training sekitar 60,29%, testing sekitar 60,10%.
# Jika keduanya rendah, aturan mungkin terlalu sederhana (underfitting).
# Jika training tinggi tetapi testing jauh lebih rendah, mungkin overfitting.
# Periksa juga recall tiap kelas; akurasi total saja belum cukup.


# %% TAHAP 8 - Membaca pohon utama sebelum membandingkan model
# Tujuan: melihat aturan yang dipelajari model pada TAHAP 6.
# Output: gambar pohon, kedalaman sebenarnya, dan bobot penggunaan fitur.
# Dari akar: kondisi <= benar ke KIRI, salah (>) ke KANAN, sampai daun.
# samples = jumlah baris training di simpul; value = jumlah per kelas sesuai
# urutan yang dicetak; class = kelas prediksi. Bobot fitur bukan sebab-akibat.

# PENGATURAN GAMBAR: jika diubah, ulang TAHAP 8 saja.
kedalaman_gambar = 3  # max_depth pada plot_tree: membatasi tampilan gambar.
ukuran_huruf = 8      # fontsize: coba 10 untuk teks lebih besar.
warna_pohon = True    # filled: coba False untuk menghilangkan warna kelas.

print("TAHAP 8 | Kedalaman model:", model.get_depth(), "| Tampilan hingga:", kedalaman_gambar)
print("Urutan kelas pada value:", list(model.classes_))
print("Baca dari akar: <= ke kiri, > ke kanan, lalu lihat class pada daun.")
folder_gambar = Path("gambar_decision_tree")
folder_gambar.mkdir(exist_ok=True)
daun_tergambar = min(model.get_n_leaves(), 2 ** (kedalaman_gambar + 1))
plt.figure(figsize=(max(20, daun_tergambar * 2.5), 10))
plot_tree(
    model,
    max_depth=kedalaman_gambar,
    feature_names=nama_fitur,
    class_names=list(model.classes_),
    filled=warna_pohon,
    rounded=True,
    impurity=False,
    fontsize=ukuran_huruf,
)
plt.title(f"Model utama: kedalaman {model.get_depth()} | Tampilan hingga {kedalaman_gambar}")
plt.tight_layout()
plt.savefig(folder_gambar / "pohon_model_utama.png", dpi=120, bbox_inches="tight")
plt.show()
plt.close()

print("Bobot penggunaan fitur (0 berarti tidak dipakai model ini):")
print(pd.Series(model.feature_importances_, index=nama_fitur).round(3))

# kedalaman_pohon di TAHAP 6 mengubah aturan model; perlu fit() ulang.
# kedalaman_gambar di TAHAP 8 hanya mengubah gambar; akurasi tetap.
# Kotak (...) berarti ada cabang lanjutan yang disembunyikan dari gambar.


# %% TAHAP 9 - Membandingkan max_depth pada TRAINING dan TESTING yang sama
# Tujuan: melihat perubahan jika hanya kedalaman pohon diganti.
# Output: gambar SETIAP kedalaman, tabel dua akurasi, dan grafik perbandingan.
# Semua percobaan memakai X_train/X_test dari TAHAP 5; tidak ada split lagi.
# Parameter lain mengikuti TAHAP 6 dan tetap selama perbandingan.
# model_coba adalah model percobaan; model utama tetap bernama model.
# for mengulangi langkah latih dan evaluasi TAHAP 6-7 untuk setiap kedalaman.
# append menyimpan satu baris hasil; DataFrame menyusunnya menjadi tabel.
# Gambar kecil tampil lengkap. Gambar besar menampilkan cabang awal agar terbaca.
# Kotak (...) berarti masih ada cabang lanjutan. Seluruh cabang disimpan ke SVG.
# Di Colab, buka folder gambar_decision_tree pada Files dan unduh SVG untuk
# memperbesar pohon lengkap. Batas gambar tidak mengubah aturan atau akurasi.
# Cuplikan depth 8 dan 12 bisa tampak sama karena cabang awalnya sama;
# lihat jumlah daun dan SVG lengkap untuk membandingkan seluruh strukturnya.

# PENGATURAN PERCOBAAN DAN GAMBAR: jika diubah, jalankan ulang TAHAP 9.
daftar_kedalaman = [3, 5, 8, 12]
batas_tampilan_pohon = 4  # Kedalaman gambar di Colab; SVG selalu lengkap.
folder_gambar = Path("gambar_decision_tree")
folder_gambar.mkdir(exist_ok=True)

print("TAHAP 9 | Mulai perbandingan max_depth. Tunggu sampai semua kedalaman selesai.")
print("Urutan kelas pada value:", list(model.classes_))
hasil_kedalaman = []
for kedalaman in daftar_kedalaman:
    print(f"\nMencoba max_depth={kedalaman} ...", flush=True)
    model_coba = DecisionTreeClassifier(
        max_depth=kedalaman,
        criterion=kriteria_pemisahan,
        min_samples_split=minimal_split,
        min_samples_leaf=minimal_daun,
        random_state=seed_model,
    )
    model_coba.fit(X_train, y_train)
    skor_train = accuracy_score(y_train, model_coba.predict(X_train))
    skor_test = accuracy_score(y_test, model_coba.predict(X_test))
    hasil_kedalaman.append({
        "max_depth": kedalaman,
        "Training (%)": skor_train * 100,
        "Testing (%)": skor_test * 100,
        "Selisih (poin %)": (skor_train - skor_test) * 100,
        "Jumlah daun": model_coba.get_n_leaves(),
    })

    print(f"Training={skor_train:.2%} | Testing={skor_test:.2%} | Daun={model_coba.get_n_leaves()}")
    print("Menyimpan pohon lengkap ke SVG; pohon besar memerlukan waktu lebih lama.", flush=True)

    # Simpan pohon LENGKAP sebagai SVG yang bisa diperbesar tanpa teks pecah.
    # DPI kecil menjaga proses ekspor ringan; ketajaman SVG tetap berbasis vektor.
    gambar_lengkap = plt.figure(
        figsize=(max(20, model_coba.get_n_leaves() * 2.5),
                 max(10, (model_coba.get_depth() + 1) * 1.5)),
        dpi=10,
    )
    plot_tree(
        model_coba,
        feature_names=nama_fitur,
        class_names=list(model_coba.classes_),
        filled=True, rounded=True, impurity=False, fontsize=8,
    )
    plt.title(f"Pohon lengkap max_depth={kedalaman} | Training={skor_train:.2%} | Testing={skor_test:.2%}")
    lokasi_svg = folder_gambar / f"pohon_max_depth_{kedalaman}.svg"
    plt.savefig(lokasi_svg, format="svg")
    plt.close(gambar_lengkap)

    # Tampilkan pohon dari model percobaan yang SAMA dengan baris tabelnya.
    kedalaman_tampil = min(model_coba.get_depth(), batas_tampilan_pohon)
    daun_tampil = min(model_coba.get_n_leaves(), 2 ** (kedalaman_tampil + 1))
    plt.figure(figsize=(max(20, daun_tampil * 2.5), 10))
    plot_tree(
        model_coba,
        max_depth=kedalaman_tampil,
        feature_names=nama_fitur,
        class_names=list(model_coba.classes_),
        filled=True, rounded=True, impurity=False, fontsize=8,
    )
    plt.title(
        f"max_depth={kedalaman} | Training={skor_train:.2%} | Testing={skor_test:.2%}\n"
        f"Kedalaman model={model_coba.get_depth()} | Tampilan hingga={kedalaman_tampil}"
    )
    plt.tight_layout()
    plt.savefig(folder_gambar / f"cuplikan_max_depth_{kedalaman}.png", dpi=120, bbox_inches="tight")
    plt.show()
    plt.close()
    if model_coba.get_depth() > kedalaman_tampil:
        print(f"max_depth={kedalaman}: gambar di atas adalah cuplikan; (...) berarti ada cabang lanjutan.")
    print("Pohon LENGKAP tersimpan di:", lokasi_svg)

tabel_kedalaman = pd.DataFrame(hasil_kedalaman)
print("\nTAHAP 9 | Hasil perbandingan max_depth dengan TRAINING/TESTING yang sama:")
print(tabel_kedalaman.round(2).to_string(index=False))

plt.figure(figsize=(8, 4))
plt.plot(tabel_kedalaman["max_depth"], tabel_kedalaman["Training (%)"], marker="o", label="Training")
plt.plot(tabel_kedalaman["max_depth"], tabel_kedalaman["Testing (%)"], marker="o", label="Testing")
plt.xticks(daftar_kedalaman)
plt.ylim(0, 100)
plt.xlabel("max_depth")
plt.ylabel("Akurasi (%)")
plt.title("Perbandingan akurasi training dan testing")
plt.legend()
plt.tight_layout()
plt.savefig(folder_gambar / "perbandingan_akurasi.png", dpi=120, bbox_inches="tight")
plt.show()
plt.close()

# Baris dengan kedalaman/pengaturan sama seperti model utama harus menghasilkan
# akurasi yang sama dengan TAHAP 7. Testing naik belum tentu selalu terjadi.
# Karena testing dipakai berulang untuk latihan perbandingan, skor ini belum
# menjadi penilaian akhir yang independen untuk model yang dipilih dari tabel.


# %% TAHAP 10 - Membandingkan parameter lainnya satu per satu
# Tujuan: mengubah satu parameter dan melihat akurasi TRAINING serta TESTING.
# Output: tabel perbandingan; pembagian data tetap sama seperti TAHAP 5.
# Acuan: depth=8, criterion=gini, min_split=2, min_leaf=1.
# Setiap baris sesudah Acuan mengubah satu nilai saja dari Acuan.
# Empat nilai tiap baris dibaca berurutan oleh for: nama, kriteria, split, daun.
# Batas split/daun lebih besar bisa menyederhanakan pohon; skor bisa berubah.
# Nilai 200 dan 50 adalah contoh latihan, bukan nilai terbaik yang ditetapkan.

# ISI PERCOBAAN: nama, criterion, min_samples_split, min_samples_leaf.
kedalaman_parameter = 8
percobaan_parameter = [
    ["Acuan",                 "gini",      2,  1],
    ["Ubah criterion",        "entropy",   2,  1],
    ["Ubah min_samples_split", "gini",    200,  1],
    ["Ubah min_samples_leaf",  "gini",      2, 50],
]

hasil_parameter = []
for nama, kriteria_uji, split_uji, daun_uji in percobaan_parameter:
    model_parameter = DecisionTreeClassifier(
        max_depth=kedalaman_parameter,
        criterion=kriteria_uji,
        min_samples_split=split_uji,
        min_samples_leaf=daun_uji,
        random_state=seed_model,
    )
    model_parameter.fit(X_train, y_train)
    hasil_parameter.append({
        "Percobaan": nama,
        "criterion": kriteria_uji,
        "min_samples_split": split_uji,
        "min_samples_leaf": daun_uji,
        "Training (%)": accuracy_score(y_train, model_parameter.predict(X_train)) * 100,
        "Testing (%)": accuracy_score(y_test, model_parameter.predict(X_test)) * 100,
        "Jumlah daun": model_parameter.get_n_leaves(),
    })

tabel_parameter = pd.DataFrame(hasil_parameter)
print(f"TAHAP 10 | Parameter lain dengan max_depth tetap {kedalaman_parameter}:")
print(tabel_parameter.round(2).to_string(index=False))

# Untuk memakai pengaturan percobaan pada gambar dan prediksi akhir, isi nilai
# tersebut di blok MODEL UTAMA TAHAP 6, lalu jalankan ulang TAHAP 6-13.


# %% TAHAP 11 - Melihat contoh training untuk diisi ulang
# Tujuan: menyediakan contoh asli sebelum mencoba input manual.
# Output: tiga nilai input dan label asli baris training ke-3 (iloc[[2]]).
# Default: umur=56, jenis kelamin=1, tinggi=99.7, label=normal.
# Jika pembagian data diganti, contoh bisa berubah; salin nilai terbaru.

contoh_training = X_train.iloc[[2]].copy()
label_asli = y_train.iloc[2]

print("TAHAP 11 | Contoh training (0=laki-laki, 1=perempuan):")
print(contoh_training.to_string(index=False))
print("Status Gizi asli contoh ini:", label_asli)


# %% TAHAP 12 - ISI SEMUA INPUT PREDIKSI DI SINI
# Tujuan: mengisi tiga parameter di satu tempat, tepat sebelum prediksi.
# Output: ringkasan input; prediksi baru muncul pada TAHAP 13.
# Cukup ubah TIGA BARIS berikut. Nilai awal disalin dari contoh TAHAP 11.
# Status Gizi tidak diisi; itu jawaban yang akan dihasilkan model.

umur_bulan = 56       # Umur dalam bulan, 0 sampai 60.
jenis_kelamin = 1     # 0=laki-laki, 1=perempuan.
tinggi_cm = 99.7      # Tinggi dalam cm; gunakan titik untuk desimal.

print("TAHAP 12 | Umur (bulan):", umur_bulan)
print("Jenis kelamin (0=laki-laki, 1=perempuan):", jenis_kelamin)
print("Tinggi (cm):", tinggi_cm)


# %% TAHAP 13 - Melakukan prediksi dengan model utama
# Tujuan: memberikan input TAHAP 12 kepada model yang sudah dilatih.
# Output: satu label prediksi Status Gizi.
# [[...]] = satu baris berisi umur, jenis kelamin, tinggi, sesuai urutan X.
# [0] mengambil hasil untuk baris pertama. Tidak ada input tambahan di sini.

data_prediksi = pd.DataFrame(
    [[umur_bulan, jenis_kelamin, tinggi_cm]],
    columns=nama_fitur,
)
hasil_prediksi = model.predict(data_prediksi)
print("TAHAP 13 | Model utama yang dipakai: max_depth =", model.max_depth,
      "| criterion =", model.criterion)
print("Hasil prediksi Status Gizi:", hasil_prediksi[0])

# LATIHAN INPUT: pada TAHAP 12, ubah tinggi_cm dari 99.7 menjadi 90.0,
# lalu ulang TAHAP 12-13. Umur, kelamin, dan model tetap.
# Model awal: 56 bulan, kelamin 1, tinggi 99.7 -> normal; tinggi 90.0 -> severely stunted.
# Jika hasil tetap sama saat input diubah, input mungkin menghasilkan kelas sama.
# Mengubah input cukup mengulang TAHAP 12-13; mengubah model mengulang TAHAP 6-13.
# Label asli contoh TAHAP 11 hanya berlaku saat ketiga inputnya sama persis.
# Satu contoh training belum mengukur kualitas model; lihat testing TAHAP 7, 9, 10.
# Model percobaan tidak otomatis mengganti model utama. Untuk memakai hasil
# percobaan, ubah pengaturan di TAHAP 6 lalu jalankan ulang TAHAP 6-13.
# Tutorial ini mempelajari label pada CSV untuk latihan machine learning.
#
# Referensi:
# https://scikit-learn.org/stable/modules/tree.html
# https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.train_test_split.html
# https://scikit-learn.org/stable/modules/generated/sklearn.tree.DecisionTreeClassifier.html
# https://scikit-learn.org/stable/modules/generated/sklearn.tree.plot_tree.html
