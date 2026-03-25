# Sentiment Analysis - Twitter Duygu Analizi

Türkçe Twitter verisi üzerinde makine öğrenmesi ile duygu analizi projesi. Logistic Regression modeli kullanılarak pozitif/negatif tweet sınıflandırması yapılır.

## Özellikler

- **Metin Ön İşleme**: Stopwords temizleme, lowercase dönüşümü
- **TF-IDF Vektörizasyon**: Metin özellik çıkarımı
- **Logistic Regression**: Sınıflandırma modeli
- **Performans Metrikleri**: Accuracy, Precision, Recall, F1-Score
- **Görselleştirme**: Confusion Matrix, ROC Curve, Precision-Recall Curve

## Veri Seti

**Önemli:** Veri seti çok büyük olduğu için GitHub'a yüklenmemiştir.

**Veri setini indirmek için:**
- [Dataset'i Google Drive'dan indir](https://drive.google.com/file/d/1j7h7dDilWn9ObrQYPgBtBdumrx836-xH/view?usp=drive_link)
- İndirdikten sonra projenin ana klasöründe `data/` klasörü oluşturun
- Dosyayı `data/training.1600000.processed.noemoticon.csv` olarak kaydedin

**Veri Seti Bilgisi:**
- 1.6 milyon tweet
- Sütunlar: sentiment, id, date, user, text
- Sentiment: 0 (Negatif), 1 (Pozitif)

## Teknolojiler

- **Python 3.8+**
- Pandas (Veri işleme)
- Scikit-learn (Makine öğrenmesi)
- NLTK (Doğal dil işleme)
- Matplotlib & Seaborn (Görselleştirme)

## Kurulum
```bash
# Repository'yi klonlayın
git clone https://github.com/sumeyyeisen/sentiment-analysis.git
cd sentiment-analysis

# Gerekli kütüphaneleri yükleyin
pip install pandas scikit-learn nltk matplotlib seaborn

# NLTK verilerini indirin
python sentiment-analysis.py
```

## Kullanım
```bash
python sentiment-analysis.py
```

## Model Performansı

Model eğitildikten sonra şu metrikleri gösterir:
- Accuracy (Doğruluk)
- Precision (Kesinlik)
- Recall (Duyarlılık)
- F1-Score
- Confusion Matrix
- ROC Curve
- Precision-Recall Curve

## Proje Hakkında

Bu proje makine öğrenmesi ve doğal dil işleme alanlarında eğitim amaçlı geliştirilmiştir.

## Lisans

Bu proje eğitim amaçlıdır.
