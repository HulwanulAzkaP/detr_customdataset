# DETR on Custom Dataset with Roboflow

## Deskripsi
Proyek ini mengimplementasikan DETR (DEtection TRansformer) untuk deteksi objek menggunakan dataset COCO dari Roboflow. DETR adalah pendekatan modern untuk deteksi objek yang memanfaatkan Transformers.

## Roboflow
Import Roboflow version menjadi **coco JSON** kemudian download filenya dan extract kedalam folder kerja `dataset/`

## Struktur Folder
- `config/`: Berisi file konfigurasi.
- `data/`: Kode untuk DataLoader dan kelas dataset.
- `dataset/`: Berisi dataset yang telah di ekstrasi
- `models/`: Definisi model DETR.
- `training/`: Script untuk pelatihan dan evaluasi model.
- `utils/`: Fungsi bantuan untuk anotasi dan visualisasi (opsional).
- `scripts/`: Script untuk menyiapkan lingkungan.
- `main.py`: Skrip utama untuk menjalankan pelatihan, evaluasi, atau inferensi.
- `requirements.txt`: Daftar dependensi proyek.

## Instalasi
### Langkah-langkah:
1. Clone repositori ini:
git clone https://github.com/HulwanulAzkaP/detr_customdataset

    cd detr_customdataset

2. Buat virtual environment:
```
python -m venv .venv
source .venv/bin/activate  # Linux/Mac
.venv\Scripts\activate     # Windows
```

3. Instal dependensi:
```pip install -r requirements.txt``` atau ```python scripts/setup_environtments.py```

## Penggunaan
### 1. Menyiapkan Dataset
Pastikan dataset berada di folder `dataset/` dengan struktur berikut:
```
dataset/
├── train/
│   ├── images/
│   └── _annotations.coco.json
├── valid/
│   ├── images/
│   └── _annotations.coco.json
└── test/
    ├── images/
    └── _annotations.coco.json
```
### 2. Menjalankan Skrip
- Untuk melatih model:

    ```python main.py```

### 3. Konfigurasi
Edit file `config/config.py` untuk menyesuaikan pengaturan seperti `num_epochs`, `batch_size`, `learning_rate`. `Threshold`, `Dan lain sebagainya`

## Contoh Hasil
Setelah pelatihan dan evaluasi, hasil metrik akan ditampilkan:
```
Epoch 20, 
Training Loss: 0.5846
Evaluating on test set...
Precision: 0.85
Recall: 0.80
F1-Score: 0.82
```
## Tujuan
Project ini kami susun guna menyelesaikan **Tugas Akhir** S1 Sains Data di **Telkom University Purwokerto** dengan judul _"Deteksi Citra Kebakaran Hutan Menggunakan Metode Deteksi Objek DETR"_

## Lisensi
Proyek ini dilisensikan di bawah lisensi MIT.
