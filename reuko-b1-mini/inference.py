import sys
from pathlib import Path

# Projenin kök dizinini Python path'ine ekle
sys.path.append(str(Path(__file__).parent.parent))

import torch
import argparse
import os
import glob
from typing import Union
from transformers import AutoTokenizer, GPT2LMHeadModel, pipeline
from utils.logging_utils import setup_logging, get_logger
from utils.config_utils import load_config

# Logging'i ayarla
setup_logging()
logger = get_logger(__name__)

def find_latest_checkpoint(model_base_dir: str) -> Union[str, None]:
    """
    Belirtilen dizindeki en son Hugging Face Trainer checkpoint'ini bulur.
    'checkpoint-1000', 'checkpoint-2000' gibi klasörleri arar.
    """
    logger.info(f"'{model_base_dir}' içinde en son checkpoint aranıyor...")
    checkpoints = glob.glob(os.path.join(model_base_dir, "checkpoint-*"))
    if not checkpoints:
        logger.warning(f"Checkpoint bulunamadı: {model_base_dir}")
        return None

    latest_checkpoint = max(checkpoints, key=lambda p: int(p.split('-')[-1]))
    logger.info(f"En son checkpoint bulundu: {latest_checkpoint}")
    return latest_checkpoint

def main():
    parser = argparse.ArgumentParser(description="Eğitilmiş bir modelle metin üretin.")
    parser.add_argument("--prompt", type=str, required=True, help="Metin üretimi için başlangıç metni.")
    parser.add_argument("--config", type=str, default="config.yaml", help="Yapılandırma dosyası.")
    parser.add_argument("--model_path", type=str, default=None, help="Modelin veya checkpoint'in yükleneceği dizin. Belirtilmezse, config'deki output_dir'den en son checkpoint aranır.")
    parser.add_argument("--max_length", type=int, default=150, help="Üretilecek metnin maksimum uzunluğu.")
    parser.add_argument("--temperature", type=float, default=0.9, help="Örnekleme sıcaklığı. Yaratıcılığı artırır.")
    parser.add_argument("--top_k", type=int, default=50, help="Top-k örnekleme için k değeri.")
    parser.add_argument("--top_p", type=float, default=0.95, help="Top-p (nucleus) örnekleme için p değeri.")
    parser.add_argument("--repetition_penalty", type=float, default=1.2, help="Tekrarları cezalandırma faktörü.")
    parser.add_argument("--no_sample", action="store_true", help="Örnekleme yerine greedy decoding kullan.")
    args = parser.parse_args()

    model_path = args.model_path
    if model_path is None:
        try:
            config = load_config(args.config)
            output_dir = config['training_args']['output_dir']
            # En son checkpoint'i veya eğitimin sonunda kaydedilen final modeli ara
            model_path = find_latest_checkpoint(output_dir)
            if model_path is None:
                final_model_path = os.path.join(output_dir, "final_model")
                if os.path.exists(final_model_path):
                    logger.info(f"Checkpoint bulunamadı, final model kullanılıyor: {final_model_path}")
                    model_path = final_model_path
                else:
                    logger.error(f"Ne bir checkpoint ne de 'final_model' bulundu: {output_dir}")
                    return
        except (FileNotFoundError, KeyError) as e:
            logger.error(f"Yapılandırma hatası: {e}. Lütfen --model_path ile bir model yolu belirtin.", exc_info=True)
            return

    try:
        device = 0 if torch.cuda.is_available() else -1
        torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32
        logger.info(f"Kullanılan cihaz: {'GPU' if device == 0 else 'CPU'}, Veri tipi: {torch_dtype}")

        logger.info(f"Model şu yoldan yükleniyor: {model_path}")
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        model = GPT2LMHeadModel.from_pretrained(model_path, torch_dtype=torch_dtype)

        generator = pipeline("text-generation", model=model, tokenizer=tokenizer, device=device)

        generation_kwargs = {
            "max_length": args.max_length,
            "num_return_sequences": 1,
            "temperature": args.temperature,
            "top_k": args.top_k,
            "top_p": args.top_p,
            "repetition_penalty": args.repetition_penalty,
            "do_sample": not args.no_sample,
            "pad_token_id": tokenizer.eos_token_id  # Açıkça pad token ID'sini belirt
        }

        logger.info(f"Üretim parametreleri: {generation_kwargs}")

        output = generator(args.prompt, **generation_kwargs)

        print("\n--- Üretilen Metin ---")
        print(output[0]['generated_text'])
        print("----------------------\n")

    except OSError:
        logger.error(f"Belirtilen yolda ({model_path}) geçerli bir model bulunamadı.", exc_info=True)
    except Exception as e:
        logger.error(f"Metin üretimi sırasında beklenmedik bir hata oluştu: {e}", exc_info=True)

if __name__ == "__main__":
    main()
