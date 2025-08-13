#!/bin/bash
set -e

echo "🚀 Reuko-1B Pipeline Başlatılıyor"
MODEL_OUTPUT_DIR="outputs/distilled"

# 1️⃣ Veri hazırlama
echo "📦 Veri hazırlanıyor..."
python scripts/prepare_data.py --dataset tatsu-lab/alpaca --model Qwen/Qwen2.5-0.5B --output_dir data/

# 2️⃣ SFT eğitimi
echo "🎓 SFT eğitimi başlıyor..."
python scripts/train_sft.py

# 3️⃣ RLHF / GRPO eğitimi
echo "🤖 RLHF eğitimi başlıyor..."
python scripts/train_rl_grpo.py

# 4️⃣ Distillation
echo "🔬 Distillation başlıyor..."
python scripts/distill.py

# 5️⃣ Evaluation
echo "📊 Model değerlendirmesi yapılıyor..."
python scripts/eval_hf.py

# 6️⃣ vLLM servisi başlat
PORT=8000
echo "🚀 vLLM server başlatılıyor, model: $MODEL_OUTPUT_DIR, port: $PORT"
bash scripts/serve_vllm.sh $MODEL_OUTPUT_DIR $PORT

echo "✅ Pipeline tamamlandı. Model artık serviste."
