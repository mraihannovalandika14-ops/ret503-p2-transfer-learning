# Dokumen Desain Pipeline Persepsi
- **Misi proyek:** Robot delivery otonom mendeteksi kendaraan (mobil dan motor) di sekitarnya dengan kamera, agar dapat menghindar dan menyesuaikan jalur.
- **Kelas objek:** Car (mobil), Bike (sepeda motor). Contoh foto ada di `dataset_raw/Car` dan `dataset_raw/Bike`.
- **Kamera & dudukan:** Kamera RGB tunggal, resolusi 640×480 (sama dengan resolusi pengambilan data). Tinggi pemasangan <<... cm>>, sudut <<... derajat>>, jarak kerja <<... m>>.
- **Unit komputasi:** Mini PC <<tipe/CPU, RAM>>, <<tanpa GPU / dengan GPU>>.
- **Target kinerja:** akurasi val ≥ 90%, ≥ 15 FPS onboard (anggaran inferensi ±35 ms dari 67 ms per frame).
- **Kandidat model:** MobileNetV3-Small (ringan, ≈2,5 juta parameter, sesuai mini PC; terukur 19,0 ms di laptop) dan ResNet-18 (baseline akurasi, ≈11,7 juta parameter; terukur 44,9 ms di laptop).
- **Strategi TL:** Fine-tuning parsial (layer4 + fc), dengan feature extraction sebagai baseline; data sedikit dan ada domain gap antara foto Kaggle dan kamera robot.
- **Rencana data:** 200 citra/kelas dari Kaggle (train) + foto kamera robot ≥ 50/kelas dengan variasi cahaya (terang, redup), jarak, dan sudut.
- **Risiko:** (1) Domain gap Kaggle vs kamera robot, mitigasi: sebagian foto robot masuk train dan tambah data robot yang seimbang antar kelas; (2) Data leakage, mitigasi: split per kondisi cahaya; (3) FPS tidak tercapai, mitigasi: pakai MobileNetV3-Small.