# Laporan Proyek Machine Learning - Farhan Ramadhan

## Domain Proyek

Proyek ini berfokus pada analisis data transaksi di industri retail, khususnya di sektor supermarket. Supermarket beroperasi dalam lingkungan bisnis yang sangat kompetitif, di mana pengelolaan data transaksi pelanggan adalah kunci untuk memahami pola pembelian, preferensi pelanggan, dan faktor yang memengaruhi kepuasan mereka.

  Format Referensi: [Sales of Supermarket](https://www.kaggle.com/datasets/lovishbansal123/sales-of-a-supermarket) 

## Business Understanding

Supermarket ini ingin memanfaatkan data transaksi historis untuk meningkatkan kinerja bisnis, baik dalam hal optimalisasi pendapatan maupun pemahaman kepuasan pelanggan. Berdasarkan data transaksi yang berisi informasi tentang pelanggan, produk, metode pembayaran, dan ulasan pelanggan, terdapat dua area fokus utama untuk pengambilan keputusan strategis.


### Problem Statements

Menjelaskan pernyataan masalah latar belakang:
- Pernyataan Masalah 1: Rantai supermarket ingin memahami pola dan tren dalam data penjualan mereka untuk meningkatkan proses pengambilan keputusan. Namun, mereka menghadapi tantangan dalam mengidentifikasi prediktor signifikan terhadap kinerja penjualan serta outlier dalam dataset mereka. Ketidakhadiran proses prapengolahan data dan rekayasa fitur yang memadai menyebabkan wawasan dan prediksi yang kurang optimal.
- Pernyataan Masalah 2: Supermarket ingin menerapkan model prediktif untuk memprediksi kinerja penjualan, tetapi belum yakin tentang akurasi dan efisiensi dari berbagai algoritma pembelajaran mesin. Mereka memerlukan evaluasi komprehensif terhadap model seperti KNN, Random Forest, dan Boosting untuk menentukan pendekatan yang paling efektif dalam meminimalkan kesalahan.

### Goals

Menjelaskan tujuan dari pernyataan masalah:
- Jawaban pernyataan masalah 1: Mengembangkan pipeline analitik prediktif secara menyeluruh yang mencakup pembersihan data, analisis data eksploratori (EDA), dan prapengolahan fitur. Menggunakan teknik statistik dan pembelajaran mesin untuk mendeteksi outlier dan mengidentifikasi faktor kunci yang memengaruhi penjualan.
- Jawaban pernyataan masalah 2: Membandingkan kinerja berbagai algoritma pembelajaran mesin dengan menggunakan Mean Squared Error (MSE) sebagai metrik evaluasi. Mengoptimalkan model untuk mengidentifikasi model yang memberikan keseimbangan terbaik antara akurasi pelatihan dan generalisasi pada data yang belum pernah dilihat.


## Data Understanding
Paragraf awal bagian ini menjelaskan informasi mengenai data yang Anda gunakan dalam proyek. Sertakan juga sumber atau tautan untuk mengunduh dataset. [Kaggle](https://www.kaggle.com/datasets/lovishbansal123/sales-of-a-supermarket).

Selanjutnya uraikanlah seluruh variabel atau fitur pada data. Sebagai contoh:  

### Variabel-variabel pada dataset adalah sebagai berikut:
- Branch: Cabang supermarket (kategori: A, B, C).
- City: Kota lokasi supermarket (kategori: Yangon, Naypyitaw, Mandalay).
- Customer type: Jenis pelanggan (Member atau Normal).
- Gender: Jenis kelamin pelanggan (Male atau Female).
- Product line: Kategori produk (Health and beauty, Electronic accessories, dll.).
- Unit price: Harga per unit produk (numerik).
- Quantity: Jumlah unit yang dibeli (numerik).
- Tax 5%: Pajak 5% dari total harga (numerik).
- Total: Total nilai transaksi (numerik).
- Date: Tanggal transaksi (datetime dalam format teks).
- Payment: Metode pembayaran (Cash, Credit card, Ewallet).
- Rating: Penilaian pelanggan (skala 1-10).

  Note: saya menghapus beberapa column dari dataset aslinya karna column tersebut tidak diperlukan untuk analisis model saya

**Rubrik/Kriteria Tambahan (Opsional)**:
- Melakukan beberapa tahapan yang diperlukan untuk memahami data, teknik visualisasi data atau exploratory data analysis, yang saya gunakan dalam notebook adalah EDA univariate dan EDA multivariate
- saya menggunakan visualisasi menggunakan bar chart, histogram dan boxplot pada tiap-tiap fitur
- saya membuat analysis data multivariate dengan membandingkan tiap fitur yang ada, dengan perbandingan multivariate categorical features vs target, lalu analisis multivariate numerical features menggunakan visualisasi boxplot, heatmap dan pairplot.

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

- hasil evaluasi berdasarkan metrik yang digunakan:
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

- Kesimpulan
KNN: Kurang cocok untuk dataset ini karena performa kurang baik (MSE tinggi) dan potensi overfitting.
RF: Memiliki performa sangat baik pada training.
Boosting: Memberikan keseimbangan terbaik antara train dan test, menjadikannya pilihan terbaik jika tujuan untuk generalisasi yang baik ke data baru.

