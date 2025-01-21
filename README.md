# Laporan Proyek Machine Learning - Farhan Ramadhan

## Domain Proyek

Proyek ini berfokus pada analisis data transaksi di industri retail, khususnya pada sektor supermarket. Dalam lingkungan bisnis yang sangat kompetitif, pengelolaan data transaksi pelanggan menjadi kunci untuk memahami pola pembelian, preferensi pelanggan, serta faktor yang memengaruhi kepuasan pelanggan.

Urgensi proyek ini adalah untuk memberikan wawasan yang mendalam mengenai tren penjualan dan preferensi pelanggan, serta mendukung pengambilan keputusan strategis berbasis data. Solusi yang diusulkan mencakup implementasi model prediksi berbasis machine learning untuk meningkatkan efisiensi operasional dan kepuasan pelanggan.

  Format Referensi: [Sales of Supermarket](https://www.kaggle.com/datasets/lovishbansal123/sales-of-a-supermarket) 

## Business Understanding

Supermarket ini ingin memanfaatkan data transaksi historis untuk meningkatkan kepuasan pelanggan dengan memahami pola pembelian dan preferensi mereka. Selain itu, supermarket berupaya memprediksi penjualan di masa depan untuk mendukung pengelolaan inventori yang lebih optimal dan strategi pemasaran yang efektif. Berdasarkan data transaksi yang mencakup informasi tentang pelanggan, produk, metode pembayaran, dan ulasan pelanggan, pengambilan keputusan strategis difokuskan pada peningkatan personalisasi layanan dan pengoptimalan pengelolaan inventori untuk mendukung pertumbuhan bisnis.


### Problem Statements

Menjelaskan pernyataan masalah latar belakang:
- Pernyataan Masalah 1: Bagaimana supermarket dapat meningkatkan kepuasan pelanggan dengan memahami pola pembelian dan preferensi mereka berdasarkan data historis?.
- Pernyataan Masalah 2: Bagaimana memprediksi penjualan di masa depan untuk membantu supermarket mengoptimalkan pengelolaan inventori dan strategi pemasaran?.

### Goals

Menjelaskan tujuan dari pernyataan masalah:
- Jawaban pernyataan masalah 1: Menggunakan analitik berbasis data untuk memberikan wawasan tentang pola pembelian pelanggan, yang dapat digunakan untuk merancang promosi yang lebih efektif dan personalisasi layanan.
- Jawaban pernyataan masalah 2: Membantu supermarket dalam merencanakan strategi bisnis berbasis prediksi penjualan yang akurat untuk mengoptimalkan pengelolaan inventori, meningkatkan pendapatan, dan mengurangi risiko kekurangan atau kelebihan stok.


## Data Understanding

Dataset terdiri dari 1000 baris dan 12 kolom. Informasi data meliputi fitur seperti cabang supermarket, jenis pelanggan, kategori produk, metode pembayaran, dan penilaian pelanggan. Sumber atau tautan untuk mengunduh dataset. [Kaggle](https://www.kaggle.com/datasets/lovishbansal123/sales-of-a-supermarket).

Kondisi Data:
- Tidak ditemukan missing values atau data duplikat.
- Data numerik perlu distandarisasi, dan fitur kategori perlu diencoding.
  
Variabel Dataset:
- Branch: Cabang supermarket (kategori: A, B, C).
- City: Kota lokasi supermarket (kategori: Yangon, Naypyitaw, Mandalay).
- Customer type: Jenis pelanggan (Member atau Normal).
- Gender: Jenis kelamin pelanggan (Male atau Female).
- Product line: Kategori produk.
- Unit price: Harga per unit (numerik).
- Quantity: Jumlah unit yang dibeli (numerik).
- Tax 5%: Pajak dari total harga (numerik).
- Total: Nilai total transaksi (numerik).
- Date: Tanggal transaksi (datetime).
- Payment: Metode pembayaran.
- Rating: Penilaian pelanggan.

  Note: beberapa column dari dataset aslinya dihapus karna column tersebut tidak diperlukan untuk analisis model

## Explorarory Data Analysis 


## Data Preparation
Pada bagian ini saya menerapkan beberapa teknik preparation data yang terdiri dari sebagai berikut:
- Encoding fitur kategorikal
- Mereduksi dimensi menggunakan Principal Component Analysis (PCA)
- Membagi dataset yang akan digunakan pada proses model development menjadi train dan test menggunakan fungsi train_test_split dari library scikit-learn saya membaginya menjadi 80% train 20% test
- Melakukan Standarisasi untuk fitur numerikal menggunakan StandardScaler.

**Rubrik/Kriteria Tambahan (Opsional)**: 
- encoding fitur kategorikal saya gunakan untuk merubah isi data kategorikal menjadi numerikal menggunakan fungsi get_dummies, tujuannya untuk merepresentasikan kategori sebagai variabel biner, menghilangkan ambiguitas, meningkatkan kinerja model, dan menghindari bias dari model.
- Reduksi menggunakan PCA digunakan karna untuk mengurangi kompleksitas model, mangatasi curse dimentionality, menghilangkan redudansi, dan mengatasi noise pada data. dengan tujuan untuk menurunkan dimensi dataset, mempertahankan informasi penting, efisiensi komputasi, meningkatkan akurasi model, dan mempermudah interpretasi.
- Membagi dataset menjadi train dan test, karna akan digunakan untuk evaluasi kinerja model, menghindari overfitting, mengukur generalisasi, menyimulasikan data baru, memvalidasi hasil model. dengan tujuan melatih model dengan data latih, menguji moel model dengan data uji, membantu proses eksperimen, meningkatkan akurasi prediksi, menjaga objektivitas.
- stadarisasi menggunakan standardscaler digunakan untuk menormalisasi data, memaksimalkan performa model, mengurangi variabilitas fitur, mengoptimalkan perhitungan jarak.

## Modeling
melakukan modeling dengan membandingkan 3 algoritma yaitu k-nearest neighbors, random forest, dan boosting dan memilih model algoritma yang memberikan hasil prediksi terbaik:
- k-NN, adalah algoritma supervised learning yang digunakan untuk klasifikasi dan regresi. Algoritma ini bekerja dengan mencari data tetangga terdekat (neighbors) berdasarkan jarak tertentu, kemudian menentukan prediksi berdasarkan mayoritas (klasifikasi) atau rata-rata (regresi) nilai tetangganya, parameter yang digunakan: n_neighbors=10 yang berfungsi untuk model menggunakan 10 tetangga terdekat dari data baru untuk menentukan nilai outputnya (klasifikasi atau regresi).
- Random Forest, Random Forest adalah algoritma ensemble learning berbasis Decision Tree yang digunakan untuk klasifikasi dan regresi. Algoritma ini membangun banyak decision tree secara acak dan menggabungkan hasilnya untuk meningkatkan akurasi dan mengurangi overfitting, parameter yang digunakan: n_estimators=100 berfungsi untuk menentukan jumlah pohon (trees) dalam ensemble dan jumlah pohon yang akan digunakan oleh model untuk membuat prediksi akhir melalui agregasi (misalnya, voting mayoritas untuk klasifikasi atau rata-rata untuk regresi), max_depth=50 berfunsgi untuk menentukan kedalaman maksimum setiap pohon keputusan (decision tree) & membatasi kedalaman pohon untuk mengontrol kompleksitas model, random_state=123 berfungsi untuk menetapkan seed untuk generator angka acak dan memastikan hasil yang reproducible (hasil yang sama setiap kali model dijalankan dengan data yang sama), n_jobs=-1 berfungsi untuk menentukan jumlah CPU yang digunakan untuk menjalankan pekerjaan secara paralel & mengatur bagaimana pekerjaan dilakukan untuk mempercepat komputasi.
- Boosting Algoritm, Boosting adalah teknik ensemble learning yang menggabungkan prediksi dari beberapa model lemah (weak learners) secara berurutan untuk menciptakan model yang lebih kuat (strong learner). Pada setiap langkah, model mencoba untuk memperbaiki kesalahan prediksi model sebelumnya, parameter yang digunakan: learning_rate=0.05 fungsinya untuk mengatur besar langkah pembaruan model pada setiap iterasi boosting, mengalikan kontribusi setiap model individu (weak learner) terhadap prediksi akhir, & menentukan seberapa besar dampak dari setiap model individu terhadap hasil akhir, random_state=55 berfungsi untuk Seed generator angka acak, memberikan konsistensi pada proses randomisasi dalam boosting (misalnya, pemilihan subset data atau fitur), memastikan bahwa hasil yang didapat reproducible (hasil yang sama setiap kali model dijalankan dengan parameter yang sama).

**Rubrik/Kriteria Tambahan (Opsional)**: 
- Menjelaskan kelebihan dan kekurangan dari setiap algoritma yang digunakan.
![Image](https://github.com/user-attachments/assets/015c1e2d-6e53-4f9a-a2ed-dfbe6c6b3125)


## Evaluation
evaluasi metrik yang digunakan adalah mean squared error(MSE) digunakan untuk memberikan penalti lebih besar pada kesalahan yang besar. Saat menghitung nilai Mean Squared Error pada data train dan test, kita membaginya dengan nilai 1e2 untuk menghindari skala yang terlalu besar.

hasil evaluasi berdasarkan metrik yang digunakan:

KNN (K-Nearest Neighbors):
Train MSE: 55.87
Test MSE: 64.15
Hasil ini menunjukkan bahwa model KNN memiliki kesalahan kuadrat rata-rata yang cukup besar, baik pada data train maupun test. Selain itu, perbedaan antara train dan test cukup signifikan, menunjukkan potensi overfitting. Model bekerja lebih baik pada data training daripada data testing.

RF (Random Forest):
Train MSE: 0.14
Test MSE: 0.85
Random Forest menunjukkan performa yang sangat baik pada data training, dengan MSE mendekati nol. Namun, pada data test, meskipun performanya masih relatif baik (MSE rendah), selisih antara train dan test cukup besar. Hal ini menunjukkan kemungkinan overfitting, di mana model terlalu "menghafal" data training.

Boosting (AdaBoost):
Train MSE: 35.69
Test MSE: 41.58
Boosting menunjukkan performa yang lebih seimbang antara data training dan testing dibandingkan dengan model lainnya. Meskipun nilai MSE-nya tidak serendah Random Forest pada test data, kesenjangan antara train dan test lebih kecil, menunjukkan bahwa Boosting mungkin lebih generalizable.

Kesimpulan

KNN: Kurang cocok untuk dataset ini karena performa kurang baik (MSE tinggi) dan potensi overfitting.
RF: Memiliki performa sangat baik pada training.
Boosting: Memberikan keseimbangan terbaik antara train dan test, menjadikannya pilihan terbaik jika tujuan untuk generalisasi yang baik ke data baru.

