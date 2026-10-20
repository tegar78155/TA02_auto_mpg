# AutoMPG Analytics

Prediksi efisiensi bahan bakar mobil menggunakan Simple Linear Regression pada dataset Auto MPG.

## Instalasi
pip install -r requirements.txt

## Menjalankan
python app.py

## Struktur
- app.py: backend Flask
- templates/index.html: halaman utama
- static/: aset frontend
- model_regresi_auto_mpg.joblib: model tersimpan
- auto-mpg.data: dataset asli

## Deployment Vercel
1. Pastikan aplikasi berjalan lokal: `python app.py`
2. Push repositori ke GitHub.
3. Import repositori di Vercel.
4. Root Directory: project root.
5. Build Command: kosong (Vercel auto).
6. Install Command: auto dari requirements.txt.
7. Deploy.
8. Uji URL `/` dan prediksi 3000 pound.

Environment variables: tidak wajib. `SECRET_KEY` opsional.
Aset statis disajikan dari folder `public/`.

- N=398, train=318, test=80
- Intercept=46.7821, koefisien weight=-0.007805
- R²=0.7230, RMSE=3.8594, MAE=3.1178
- Contoh prediksi weight=3000 -> 23.37 MPG
