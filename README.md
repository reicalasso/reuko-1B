# Reuko-1B: Production-Ready 1B+ Parameter NLP Pipeline

Reuko-1B, T5 tabanlı 1 milyar ve üzeri parametreli modelleri **Question Answering (QA)** ve **Text Summarization** görevleri için eğitmek ve kullanmak üzere tasarlanmış, modern ve ölçeklenebilir bir AI pipeline'ıdır.

## 🔥 Modern Özellikler

- **Büyük Model Desteği**: 1B+ parametreli özel T5 mimarisi ve Hugging Face modelleri.
- **Verimli Eğitim**: **DeepSpeed ZeRO** (Stage 2/3) ile bellek optimizasyonu.
- **PEFT (LoRA)**: **Parameter-Efficient Fine-Tuning** ile daha az kaynakla hızlı eğitim.
- **Yüksek Performans**: **BF16/FP16 mixed-precision** ve **fused optimizers** desteği.
- **Multi-GPU Hazır**: **Hugging Face Accelerate** ile kolayca çoklu GPU'ya ölçeklendirme.
- **İzleme ve Loglama**: **Weights & Biases (W&B)** entegrasyonu ile canlı metrik takibi.
- **Modüler ve Genişletilebilir**: Temiz, profesyonel ve ölçeklenebilir kod yapısı.

## ⚙️ Donanım Gereksinimleri

1B+ parametreli bir modeli eğitmek güçlü bir donanım gerektirir:

- **Minimum**: 1 x NVIDIA RTX 3090 / 4090 (24GB VRAM)
- **Önerilen**: 1-4 x NVIDIA A100 / H100 (40GB+ VRAM)
- **CPU RAM**: 64GB+
- **Depolama**: 100GB+ (Dataset'ler, cache ve model checkpoint'leri için)

## 🛠️ Kurulum

1.  **Projeyi Klonlayın:**
    ```bash
    git clone <repo-url>
    cd reuko-1B
    ```

2.  **Bağımlılıkları Yükleyin:**
    ```bash
    pip install -r requirements.txt
    ```

3.  **Paketi Geliştirme Modunda Yükleyin:**
    Bu komut, `reuko-train` gibi CLI araçlarını kullanılabilir hale getirir.
    ```bash
    pip install -e .
    ```

## 🚀 Eğitimi Başlatma

Eğitimi başlatmak için `reuko-train` komutunu ve `config.yaml` dosyasını kullanacaksınız.

### Tek GPU ile Eğitim (DeepSpeed + LoRA)

Bu en yaygın ve verimli yöntemdir.

```bash
# Hem QA hem de Summarization görevleri için modeli eğit
# DeepSpeed, eğitimi otomatik olarak yönetecek
deepspeed --num_gpus=1 reuko_1b/cli/train.py --config config.yaml --task both
```

### Sadece Tek Bir Görev İçin Eğitim

```bash
# Sadece Question Answering (QA) modelini eğit
deepspeed --num_gpus=1 reuko_1b/cli/train.py --config config.yaml --task qa

# Sadece Summarization modelini eğit
deepspeed --num_gpus=1 reuko_1b/cli/train.py --config config.yaml --task summarization
```

### Çoklu GPU ile Eğitim

Eğer 2 adet GPU'nuz varsa, `num_gpus` parametresini değiştirmeniz yeterlidir.

```bash
# 2 GPU ile QA modelini eğit
deepspeed --num_gpus=2 reuko_1b/cli/train.py --config config.yaml --task qa
```

### Eğitim Sırasında İzleme (Monitoring)

Eğer `config.yaml` dosyasında `use_wandb: true` ayarı aktif ise, eğitim başladığında size bir **Weights & Biases** linki verilecektir. Bu linki tarayıcınızda açarak eğitiminizi canlı olarak takip edebilirsiniz.

## ⚙️ Yapılandırma (`config.yaml`)

Eğitimi özelleştirmek için `config.yaml` dosyasını düzenleyebilirsiniz:

- **`model.name`**: Hugging Face'den baz alınacak model.
- **`training.batch_size`**: GPU başına batch boyutu (Genellikle 1B modeller için 1'dir).
- **`training.gradient_accumulation_steps`**: Etkin batch boyutunu artırmak için.
- **`peft.enabled`**: LoRA ile eğitimi aç/kapa.
- **`deepspeed.enabled`**: DeepSpeed'i aç/kapa.

## 🐍 Python API ile Kullanım

Eğitilmiş modelleri kullanmak için `example_usage.py` dosyasına göz atın.

```bash
python example_usage.py
```

## 📁 Proje Yapısı

```
reuko-1B/
├── reuko_1b/               # Ana Python paketi
│   ├── cli/                # Komut satırı arayüzleri (train, test)
│   ├── models/             # Model mimarileri (base, custom_t5)
│   ├── pipeline/           # Eğitim ve inference pipeline'ları
│   └── utils/              # Yardımcı modüller (config, logger)
├── config.yaml             # Ana konfigürasyon dosyası
├── ds_config.json          # DeepSpeed konfigürasyonu
├── setup.py                # Paket kurulum script'i
├── requirements.txt        # Gerekli kütüphaneler
└── README.md               # Bu doküman
```
---

Made with ❤️ by Rei Calasso
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
