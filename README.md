# Reuko-1B Pipeline

Bu proje, Qwen modelleri üzerinde SFT (Supervised Fine-Tuning), RLHF (Reinforcement Learning with Human Feedback), distillation ve değerlendirme işlemlerini gerçekleştiren bir pipeline sunar.

## Gereksinimler

Aşağıdaki bağımlılıkları yüklemek için `requirements.txt` dosyasını kullanabilirsiniz:

```bash
pip install -r requirements.txt
```

## Kullanım

### 1️⃣ Veri Hazırlama
Veri setini tokenize edip diske kaydetmek için:
```bash
python scripts/prepare_data.py --dataset tatsu-lab/alpaca --model Qwen/Qwen2.5-0.5B
```

### 2️⃣ SFT Eğitimi
SFT eğitimi başlatmak için:
```bash
python scripts/train_sft.py
```

### 3️⃣ RLHF Eğitimi
RLHF eğitimi başlatmak için:
```bash
python scripts/train_rl_grpo.py
```

### 4️⃣ Distillation
Distillation işlemini başlatmak için:
```bash
python scripts/distill.py
```

### 5️⃣ Değerlendirme
Modeli değerlendirmek için:
```bash
python scripts/eval_hf.py
```

### 6️⃣ Model Servisi
Modeli vLLM ile servis etmek için:
```bash
bash scripts/serve_vllm.sh outputs/distilled 8000
```

## Pipeline
Tüm adımları otomatik olarak çalıştırmak için:
```bash
bash run.sh
```

## Yapılandırma Dosyaları
Yapılandırma dosyaları `configs/` klasöründe bulunur. Her bir adım için özelleştirilebilir parametreler içerir:
- `configs/sft.yaml`
- `configs/rl_grpo.yaml`
- `configs/distill.yaml`
- `configs/eval.yaml`

## Lisans
Bu proje MIT lisansı altındadır.
