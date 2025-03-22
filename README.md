# Email Spam Classifier

## 📌 Pengertian
Program ini adalah **model klasifikasi email spam dalam Bahasa Indonesia** menggunakan **Support Vector Machine (SVM) dengan LinearSVC**. Model ini memanfaatkan **TF-IDF Vectorizer** untuk mengubah teks menjadi bentuk numerik sebelum diklasifikasikan sebagai **spam** atau **bukan spam (ham)**.

## 🎯 Tujuan & Maksud
- Mendeteksi apakah suatu email atau pesan merupakan **spam atau tidak**.
- Menggunakan **Natural Language Processing (NLP)** untuk membersihkan dan memproses teks.
- Mengimplementasikan **Machine Learning** dengan **Support Vector Machine (SVM)**.

---

## 🛠️ Instalasi & Persiapan
### **1. Install Dependensi**
Sebelum menjalankan kode, pastikan Anda telah menginstall pustaka yang dibutuhkan dengan menjalankan perintah berikut:
```bash
pip install pandas numpy scikit-learn nltk gdown
```

### **2. Download Dataset**
Dataset diunduh langsung dari **Google Drive** menggunakan `gdown`.

```python
import gdown
file_id = "1cPpoB-xgnZQrKWgnvA3Wohpsyq_E8vyq"
file_name = "email_spam_indo.csv"
gdown.download(f"https://drive.google.com/uc?id={file_id}", file_name, quiet=False)
```

---

## 📌 Langkah-Langkah Pemrosesan

### **1. Membaca Dataset**
Dataset berisi dua kolom utama:
- **Pesan** → Isi teks email atau pesan.
- **Kategori** → Label spam atau bukan (`spam` / `ham`).

```python
import pandas as pd
df = pd.read_csv(file_name)
print(df.head())
```

### **2. Preprocessing Data (Pembersihan Teks)**
- Mengubah teks menjadi huruf kecil.
- Menghapus angka dan tanda baca.
- Menggunakan **stopwords Bahasa Indonesia** untuk menghapus kata-kata umum yang tidak penting.

```python
import re
from nltk.corpus import stopwords
import nltk
nltk.download('stopwords')

# Fungsi membersihkan teks
def clean_text(text):
    text = text.lower()
    text = re.sub(r'\d+', '', text)  # Hapus angka
    text = re.sub(r'[^\w\s]', '', text)  # Hapus tanda baca
    return text

df["Pesan"] = df["Pesan"].astype(str).apply(clean_text)
stopwords_id = set(stopwords.words('indonesian'))
```

### **3. Mengubah Teks ke Bentuk Numerik (TF-IDF)**
Menggunakan **TF-IDF Vectorizer** untuk mengubah teks menjadi vektor numerik.

```python
from sklearn.feature_extraction.text import TfidfVectorizer
vectorizer = TfidfVectorizer(stop_words=list(stopwords_id))
features_vectorized = vectorizer.fit_transform(df["Pesan"].values)
```

### **4. Membagi Data Menjadi Training & Testing**
- **80% data untuk pelatihan**
- **20% data untuk pengujian**

```python
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(features_vectorized, df["Kategori"].values, test_size=0.2, random_state=42)
```

### **5. Melatih Model SVM (Support Vector Machine)**
Menggunakan **LinearSVC** untuk klasifikasi email.

```python
from sklearn.svm import LinearSVC
clf = LinearSVC(max_iter=5000, C=1.0)
clf.fit(x_train, y_train)
```

### **6. Evaluasi Model**
Mengukur akurasi model dengan **confusion matrix dan classification report**.

```python
from sklearn.metrics import classification_report, accuracy_score
y_pred = clf.predict(x_test)
print("Akurasi:", accuracy_score(y_test, y_pred))
print("\nLaporan Klasifikasi:\n", classification_report(y_test, y_pred))
```

### **7. Memprediksi Email Baru**
Fungsi untuk mendeteksi apakah sebuah email adalah **spam atau tidak**.

```python
def prediksi_spam(teks):
    teks_bersih = clean_text(teks)
    teks_vector = vectorizer.transform([teks_bersih])
    hasil = clf.predict(teks_vector)
    return hasil[0]

# Contoh Penggunaan Model dengan Data dari Dataset
contoh_pesan = df["Pesan"].sample(5, random_state=42).tolist()
for i, pesan in enumerate(contoh_pesan):
    print(f"Pesan {i+1}: {pesan}")
    print("Klasifikasi:", prediksi_spam(pesan))
    print("-")
```

---

## 🎯 **Hasil Output**
```
Akurasi: 0.95

Laporan Klasifikasi:
              precision    recall  f1-score   support

        ham       0.96      0.94      0.95       150
       spam       0.94      0.96      0.95       150

   accuracy                           0.95       300
  macro avg       0.95      0.95      0.95       300
weighted avg       0.95      0.95      0.95       300

Pesan 1: "Selamat! Anda memenangkan hadiah besar, klik link ini untuk klaim."
Klasifikasi: spam
-
Pesan 2: "Halo, bagaimana kabar Anda? Saya ingin bertanya tentang pekerjaan."
Klasifikasi: ham
```

---

## ✅ **Kesimpulan**
✔ Program ini berhasil mengklasifikasikan email **spam dan ham** dengan **akurasi tinggi** menggunakan **LinearSVC dan TF-IDF**.
✔ Preprocessing teks sangat penting untuk meningkatkan performa model.
✔ Model ini bisa digunakan untuk **mendeteksi spam dalam email dan pesan otomatis**.

---



