# Reuko-1B: Mini T5 Pipeline

Reuko-1B, T5 tabanlı Question Answering (QA) ve Text Summarization görevleri için geliştirilmiş bir mini AI pipeline'dır.

## 🚀 Özellikler

- **Question Answering**: SQuAD dataset ile eğitilmiş QA modeli
- **Text Summarization**: CNN/DailyMail dataset ile özetleme
- **Hızlı Prototipleme**: Mini dataset'ler ile hızlı test
- **Kolay Kullanım**: Jupyter notebook ile interaktif geliştirme

## 📋 Gereksinimler

- Python 3.10+
- PyTorch 2.0+
- Transformers 4.30+
- CUDA (opsiyonel, GPU desteği için)

## 🛠️ Kurulum

1. **Repository'yi klonla:**
```bash
git clone <repo-url>
cd reuko-1B
```

2. **Bağımlılıkları yükle:**
```bash
pip install -r requirements.txt
```

3. **Notebook'u çalıştır:**
```bash
jupyter notebook reuko_mini_pipeline.ipynb
```

## 📚 Kullanım

### Hızlı Başlangıç

1. `reuko_mini_pipeline.ipynb` dosyasını aç
2. Tüm hücreleri sırasıyla çalıştır
3. Model otomatik olarak eğitilir ve test edilir

### QA Örneği

```python
question = "Who developed Python?"
context = "Python was created by Guido van Rossum in 1991."
answer = test_qa(question, context)
print(f"Answer: {answer}")
```

### Summarization Örneği

```python
text = "Long article text here..."
summary = test_summarization(text)
print(f"Summary: {summary}")
```

## 🎯 Model Performansı

- **Model**: T5-small (60M parametreler)
- **Training**: 2 epoch, mini dataset
- **Hız**: ~5 saniye/batch (CPU)
- **Bellek**: ~2GB RAM

## 📁 Proje Yapısı

```
reuko-1B/
├── reuko_mini_pipeline.ipynb   # Ana notebook
├── config.py                   # Konfigürasyon
├── utils.py                    # Yardımcı fonksiyonlar
├── requirements.txt            # Bağımlılıklar
├── setup.py                    # Kurulum scripti
├── README.md                   # Bu dosya
└── reuko_mini/                 # Model çıktıları
    ├── qa_model/              # QA modeli
    └── summary_model/         # Summarization modeli
```

## ⚙️ Konfigürasyon

`config.py` dosyasında aşağıdaki parametreleri değiştirebilirsiniz:

- `BATCH_SIZE`: Batch büyüklüğü (varsayılan: 4)
- `NUM_EPOCHS`: Epoch sayısı (varsayılan: 2)
- `QA_TRAIN_SIZE`: QA training veri boyutu (varsayılan: 1000)
- `LEARNING_RATE`: Öğrenme oranı (varsayılan: 5e-4)

## 🔧 Geliştirme

### Yeni Veri Seti Ekleme

```python
from utils import ReukoUtils

utils = ReukoUtils(tokenizer)
processed_data = utils.preprocess_qa_batch(new_dataset)
```

### Model İyileştirme

1. **Daha fazla veri**: `config.py`'da veri boyutlarını artır
2. **Daha uzun eğitim**: `NUM_EPOCHS` değerini artır
3. **Daha büyük model**: `MODEL_NAME`'i `t5-base` olarak değiştir

## 📊 Sonuçlar

Notebook çalıştırıldığında:
- Training loss grafikleri gösterilir
- QA örnekleri test edilir
- Summarization örnekleri test edilir
- Model performans metrikleri yazdırılır

## 🤝 Katkıda Bulunma

1. Fork yapın
2. Feature branch oluşturun (`git checkout -b feature/amazing-feature`)
3. Değişikliklerinizi commit edin (`git commit -m 'Add some AmazingFeature'`)
4. Branch'inizi push edin (`git push origin feature/amazing-feature`)
5. Pull Request açın

## 📄 Lisans

Bu proje MIT lisansı altında lisanslanmıştır. Detaylar için [LICENSE](LICENSE) dosyasına bakın.

## 🚀 Sonraki Adımlar

- [ ] Multi-task learning implementation
- [ ] ROUGE/BLEU evaluation metrics
- [ ] Model distillation for speed
- [ ] Web API deployment
- [ ] Docker containerization
- [ ] Gradio UI integration

## 📞 İletişim

- GitHub: [Reuko-1B](https://github.com/your-username/reuko-1B)
- Issues: [GitHub Issues](https://github.com/your-username/reuko-1B/issues)

---

Made with ❤️ by Rei Calasso
