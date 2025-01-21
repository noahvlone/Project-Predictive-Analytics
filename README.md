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
- EDA Univariate tiap Fitur dengan Barplot dan Histogram Hasil Analisis:
  
  untuk fitur kategorikal:
  1. cabang supermarket yang paling banyak pembelinya adalah cabang supermarket A diikuti cabang B dan C.
  2. distribusi kota dengan penjualan yang paling besar terletak di kota Yangon lalu kota Mandalay pada urutan kedua dan kota Naypyitaw urutan ketiga.
  3. customer dengan tipe member/langganan sedikit lebih unggul dibandingkan dengan customer non member/normal, Customer supermarket paling banyak adalah Female/Perempuan.
  4. jenis kategori produk yang paling diminati adalah Fashion accessories, Food and beverages, dan Electronic accessories,
  5. metode pembayaran paling banyak digunakan customer adalah E-wallet karna praktis.

  untuk fitur numerikal:
  1. harga barang disupermarket yang paling banyak dibeli adalah barang dengan range harga 30-75 dollar,
  2. rata-rata jumlah barang/quantity yang dibeli disupermarket adalah 5 barang,
  3. rata-rata pajak pembelian berjumlah 15 dollar per customer,
  4. rata-rata total pembelian produk disupermarket berkisar 322.2 dollar per customer, pembelian total terkecil 10.6 dollar, dan pembelian total terbesar 1042 dollar, dan
  5. rating rata-rata dari supermarket tersebut adalah 6.9.
  
- EDA Multivariate tiap fitur dengan Boxplot, Pairplot, dan Heatmap, berikut hasil yang didapatkan:
  1. Insight Total vs Branch: Cabang C memiliki performa yang paling tinggi dalam hal nilai rata-rata dan distribusi transaksi total dibandingkan cabang lainnya. Namun, variasi di cabang ini juga yang paling besar. Jika fokus pada stabilitas, cabang A menunjukkan performa dengan variasi yang lebih kecil.
  2. Insight Total vs City: Kota Naypyitaw menunjukkan performa tertinggi dalam hal rata-rata transaksi dan nilai transaksi maksimum, namun juga memiliki variasi yang paling besar. Kota Yangon memiliki nilai transaksi yang lebih konsisten dibandingkan kota lainnya. Analisis ini dapat digunakan untuk strategi bisnis yang lebih spesifik, seperti personalisasi layanan di Naypyitaw untuk transaksi bernilai tinggi dan stabilitas operasional di Yangon.
  3. Insight Total vs Customer type: Pelanggan Member cenderung memiliki nilai transaksi rata-rata yang sedikit lebih tinggi dibandingkan pelanggan Normal, dengan distribusi transaksi yang hampir sama secara statistik. Hal ini menunjukkan adanya kecenderungan bahwa pelanggan Member mungkin memiliki preferensi atau pola belanja yang lebih besar dibandingkan pelanggan Normal.
  4. Insight Total vs Gender: Pelanggan Female cenderung memiliki nilai transaksi rata-rata yang lebih tinggi dibandingkan pelanggan Male pada semua persentil (25%, 50%, dan 75%). Meskipun nilai maksimum antara Female dan Male hampir sama, nilai transaksi pelanggan Female menunjukkan kecenderungan lebih tinggi secara keseluruhan. Hal ini mungkin mengindikasikan bahwa pelanggan Female memiliki daya beli atau preferensi belanja yang lebih besar.
  5. Insight Total vs Product line: Home and lifestyle adalah lini produk dengan rata-rata dan median transaksi tertinggi, menunjukkan popularitas atau nilai produk yang lebih tinggi dibanding kategori lain. Fashion accessories memiliki nilai transaksi rata-rata dan median yang lebih rendah, meskipun memiliki nilai maksimum tertinggi. Ini menunjukkan adanya beberapa transaksi besar yang meningkatkan maksimum, tetapi secara keseluruhan transaksi di kategori ini cenderung lebih kecil. Sports and travel dan Food and beverages menunjukkan distribusi nilai transaksi yang relatif merata, dengan rata-rata yang berada di tengah-tengah kategori lainnya.
  6. Insight Total vs Payment: Transaksi dengan uang tunai (Cash) menunjukkan nilai rata-rata dan median yang lebih tinggi, mengindikasikan bahwa metode ini digunakan lebih sering untuk transaksi dengan jumlah menengah ke atas. Transaksi dengan Ewallet menunjukkan konsistensi (penyebaran lebih kecil), sedangkan kartu kredit memiliki variasi transaksi yang lebih besar. Metode pembayaran berbeda memiliki keunikan masing-masing dalam pola transaksi.
  7. Insight Pairplot: Total transaksi terutama dipengaruhi oleh Unit Price dan Quantity. Pajak 5% sepenuhnya linier terhadap total belanja. Rating tampaknya tidak berkorelasi dengan variabel-variabel lainnya. Data menunjukkan distribusi yang masuk akal, dengan beberapa pola hubungan yang diharapkan, terutama antara total dan variabel lainnya seperti harga satuan, jumlah barang, dan pajak.
  8. Insight Heatmap: Total belanja dipengaruhi terutama oleh Unit Price dan Quantity, serta secara langsung oleh Tax 5%. Rating tidak memiliki hubungan signifikan dengan variabel lainnya, mengindikasikan bahwa kepuasan pelanggan tidak berkaitan dengan aspek finansial dalam data ini. Pajak 5% hanya bertindak sebagai komponen langsung dari total belanja tanpa hubungan independen lainnya.

## Data Preparation
Menerapkan beberapa teknik preparation data yang terdiri dari sebagai berikut:
- Mengecek Duplicate Data
- Mengecek Missing Values Data
- Mengubah Tipe Data kolom Date menjadi datetime
- Encoding fitur kategorikal
- Mereduksi dimensi menggunakan Principal Component Analysis (PCA)
- Membagi dataset yang akan digunakan pada proses model development menjadi train dan test menggunakan fungsi train_test_split dari library scikit-learn saya membaginya menjadi 80% train 20% test
- Melakukan Standarisasi untuk fitur numerikal menggunakan StandardScaler.
- Menghapus Kolom Date karna tidak diperlukan untuk modeling

Penjelasan Tahap yang dilakukan sesuai Notebook: 
- Mengecek duplikasi data menggunakan fungsi duplicated().sum()
- Mengecek missing value data menggunakan fungsi isnull().sum()
- Encoding fitur kategorikal saya gunakan untuk merubah isi data kategorikal menjadi numerikal menggunakan fungsi get_dummies, tujuannya untuk merepresentasikan kategori sebagai variabel biner, menghilangkan ambiguitas, meningkatkan kinerja model, dan menghindari bias dari model.
- Reduksi menggunakan PCA digunakan karna untuk mengurangi kompleksitas model, mangatasi curse dimentionality, menghilangkan redudansi, dan mengatasi noise pada data. dengan tujuan untuk menurunkan dimensi dataset, mempertahankan informasi penting, efisiensi komputasi, meningkatkan akurasi model, dan mempermudah interpretasi.
- Membagi dataset menjadi train dan test, karna akan digunakan untuk evaluasi kinerja model, menghindari overfitting, mengukur generalisasi, menyimulasikan data baru, memvalidasi hasil model. dengan tujuan melatih model dengan data latih, menguji moel model dengan data uji, membantu proses eksperimen, meningkatkan akurasi prediksi, menjaga objektivitas.
- Stadarisasi menggunakan standardscaler digunakan untuk menormalisasi data, memaksimalkan performa model, mengurangi variabilitas fitur, mengoptimalkan perhitungan jarak.
- Menghapus/drop kolom Date karena algoritma machine learning tidak dapat secara langsung menangani data dengan tipe datetime. Model memerlukan data numerik atau data yang telah dikonversi menjadi fitur relevan untuk melakukan analisis.

## Modeling
melakukan modeling dengan membandingkan 3 algoritma yaitu k-nearest neighbors, random forest, dan boosting dan memilih model algoritma yang memberikan hasil prediksi terbaik:
- k-NN, adalah algoritma supervised learning yang digunakan untuk klasifikasi dan regresi. Algoritma ini bekerja dengan mencari data tetangga terdekat (neighbors) berdasarkan jarak tertentu, kemudian menentukan prediksi berdasarkan mayoritas (klasifikasi) atau rata-rata (regresi) nilai tetangganya, parameter yang digunakan: n_neighbors=10 yang berfungsi untuk model menggunakan 10 tetangga terdekat dari data baru untuk menentukan nilai outputnya (klasifikasi atau regresi).
- Random Forest, Random Forest adalah algoritma ensemble learning berbasis Decision Tree yang digunakan untuk klasifikasi dan regresi. Algoritma ini membangun banyak decision tree secara acak dan menggabungkan hasilnya untuk meningkatkan akurasi dan mengurangi overfitting, parameter yang digunakan: n_estimators=100 berfungsi untuk menentukan jumlah pohon (trees) dalam ensemble dan jumlah pohon yang akan digunakan oleh model untuk membuat prediksi akhir melalui agregasi (misalnya, voting mayoritas untuk klasifikasi atau rata-rata untuk regresi), max_depth=50 berfunsgi untuk menentukan kedalaman maksimum setiap pohon keputusan (decision tree) & membatasi kedalaman pohon untuk mengontrol kompleksitas model, random_state=123 berfungsi untuk menetapkan seed untuk generator angka acak dan memastikan hasil yang reproducible (hasil yang sama setiap kali model dijalankan dengan data yang sama), n_jobs=-1 berfungsi untuk menentukan jumlah CPU yang digunakan untuk menjalankan pekerjaan secara paralel & mengatur bagaimana pekerjaan dilakukan untuk mempercepat komputasi.
- Boosting Algoritm, Boosting adalah teknik ensemble learning yang menggabungkan prediksi dari beberapa model lemah (weak learners) secara berurutan untuk menciptakan model yang lebih kuat (strong learner). Pada setiap langkah, model mencoba untuk memperbaiki kesalahan prediksi model sebelumnya, parameter yang digunakan: learning_rate=0.05 fungsinya untuk mengatur besar langkah pembaruan model pada setiap iterasi boosting, mengalikan kontribusi setiap model individu (weak learner) terhadap prediksi akhir, & menentukan seberapa besar dampak dari setiap model individu terhadap hasil akhir, random_state=55 berfungsi untuk Seed generator angka acak, memberikan konsistensi pada proses randomisasi dalam boosting (misalnya, pemilihan subset data atau fitur), memastikan bahwa hasil yang didapat reproducible (hasil yang sama setiap kali model dijalankan dengan parameter yang sama).

**Kelbihan Kekurangan Algorithm**: 
- kelebihan dan kekurangan dari setiap algoritma yang digunakan.
![Image](https://github.com/user-attachments/assets/015c1e2d-6e53-4f9a-a2ed-dfbe6c6b3125)


## Evaluation
Evaluasi metrik yang digunakan adalah mean squared error(MSE) digunakan untuk memberikan penalti lebih besar pada kesalahan yang besar. Saat menghitung nilai Mean Squared Error pada data train dan test, kita membaginya dengan nilai 1e2 untuk menghindari skala yang terlalu besar.

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

## Evaluasi dan Dampak Model terhadap Business Understanding

1. Menjawab Problem Statements:
   
Pernyataan Masalah 1: Bagaimana supermarket dapat meningkatkan kepuasan pelanggan dengan memahami pola pembelian dan preferensi mereka berdasarkan data historis?

Evaluasi:
- Insight dari EDA memberikan wawasan mengenai preferensi pelanggan berdasarkan kategori produk, metode pembayaran, dan jenis pelanggan (Member/Normal). Hal ini dapat digunakan untuk merancang strategi personalisasi layanan, seperti penawaran promosi pada kategori produk yang diminati (misalnya, Fashion Accessories dan Food and Beverages) atau memfokuskan layanan pelanggan pada pelanggan Member yang cenderung memiliki nilai transaksi lebih tinggi.
- Model prediktif (Boosting) dapat membantu memahami faktor-faktor yang memengaruhi penjualan, misalnya dengan memprediksi kategori produk yang paling laris atau pola pembelian yang sering dilakukan pelanggan.

Pernyataan Masalah 2: Bagaimana memprediksi penjualan di masa depan untuk membantu supermarket mengoptimalkan pengelolaan inventori dan strategi pemasaran?

Evaluasi:
- Model Boosting memiliki kinerja yang paling baik dalam hal keseimbangan antara data training dan testing (Train MSE: 35.69, Test MSE: 41.58). Dengan tingkat kesalahan yang lebih kecil dan generalisasi yang lebih baik dibandingkan model lain, Boosting dapat diandalkan untuk memprediksi penjualan di masa depan.
- Prediksi ini dapat digunakan untuk mengatur inventori secara lebih efisien, seperti mempersiapkan stok barang pada kategori yang memiliki prediksi penjualan tinggi (misalnya Food and Beverages atau Home and Lifestyle).

2. Pencapaian Goals:
   
- Goal 1: Memberikan wawasan berbasis data untuk merancang promosi yang lebih efektif dan personalisasi layanan.
Berdasarkan hasil EDA, supermarket dapat memprioritaskan metode pembayaran E-wallet yang banyak digunakan oleh pelanggan atau fokus pada pelanggan Female yang memiliki daya beli lebih tinggi.
Laporan ini menunjukkan bahwa strategi berbasis data dapat memberikan dampak langsung pada pengelolaan promosi dan personalisasi layanan.

- Goal 2: Membantu perencanaan bisnis berbasis prediksi penjualan yang akurat.
Dengan Boosting yang memiliki generalisasi lebih baik, prediksi yang dihasilkan cukup akurat untuk mendukung pengelolaan inventori dan perencanaan strategi pemasaran.
Random Forest juga memberikan akurasi yang baik pada data training, namun risiko overfitting menjadi perhatian untuk digunakan dalam data baru.

3. Dampak Solusi terhadap Bisnis:
   
- Peningkatan Kepuasan Pelanggan: Analisis preferensi pelanggan berdasarkan data historis memberikan supermarket peluang untuk meningkatkan pengalaman pelanggan melalui personalisasi promosi dan layanan.
- Efisiensi Operasional: Dengan prediksi penjualan yang lebih akurat, supermarket dapat mengurangi risiko kekurangan atau kelebihan stok, sehingga meningkatkan efisiensi operasional.
- Strategi Pemasaran yang Lebih Tepat: Informasi tentang kategori produk, metode pembayaran, dan segmentasi pelanggan memungkinkan supermarket merancang kampanye pemasaran yang lebih efektif dan terarah.

Kesimpulan:

Hasil evaluasi menunjukkan bahwa proyek ini telah berhasil menjawab problem statements dan mencapai goals yang diharapkan:
- Model prediktif (Boosting) memberikan solusi untuk memprediksi penjualan dengan generalisasi yang lebih baik.
- Analisis data historis (EDA) memberikan wawasan tentang pola pembelian pelanggan, yang dapat digunakan untuk meningkatkan kepuasan pelanggan dan efisiensi bisnis.

