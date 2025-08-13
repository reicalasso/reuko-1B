import sys
import os
import argparse
from pathlib import Path
from typing import Union

from utils.logging_utils import setup_logging, get_logger
from utils.config_utils import load_config, get_deepspeed_config
from data.dataset import get_tokenizer, load_and_tokenize_dataset
from model.gpt_model import build_model
from train.trainer import get_trainer

# Proje genelinde logging'i ayarla
setup_logging()
logger = get_logger(__name__)

def main(config_path: str, resume_from_checkpoint: Union[str, bool]):
    """
    Ana eğitim sürecini yönetir.
    1. Yapılandırmayı yükler.
    2. Tokenizer ve veri setini hazırlar.
    3. Modeli oluşturur.
    4. Trainer'ı yapılandırır ve eğitimi başlatır.
    """
    try:
        # 1. Yapılandırmayı yükle
        config = load_config(config_path)
        model_config = config['model_config']
        dataset_config = config['dataset_config']
        training_args_dict = config['training_args']
        deepspeed_config_path = get_deepspeed_config(config)

        # Ortam değişkenlerini ayarla (W&B gibi entegrasyonlar için)
        if training_args_dict.get("report_to") == "wandb":
            os.environ["WANDB_PROJECT"] = config.get(
                "wandb_project", "reuko-b1-mini-experiment"
            )
            logger.info(f"W&B projesi ayarlandı: {os.environ['WANDB_PROJECT']}")

        # 2. Tokenizer ve veri setini hazırlar
        tokenizer = get_tokenizer()
        train_dataset, val_dataset = load_and_tokenize_dataset(
            dataset_config, tokenizer
        )

        # 3. Modeli oluşturur
        model = build_model(tokenizer, model_config)

        # 4. Trainer'ı yapılandırır ve eğitimi başlatır
        trainer = get_trainer(
            model=model,
            tokenizer=tokenizer,
            train_dataset=train_dataset,
            val_dataset=val_dataset,
            training_args_dict=training_args_dict,
            deepspeed_config=deepspeed_config_path
        )

        logger.info("Eğitim başlatılıyor...")
        trainer.train(resume_from_checkpoint=resume_from_checkpoint)
        logger.info("Eğitim tamamlandı.")

        # Eğitilmiş modeli ve tokenizer'ı son bir kez kaydet
        final_path = os.path.join(
            training_args_dict['output_dir'], "final_model"
        )
        logger.info(f"Son model ve tokenizer şuraya kaydediliyor: {final_path}")
        trainer.save_model(final_path)
        tokenizer.save_pretrained(final_path)
        logger.info("Model kaydetme işlemi tamamlandı.")

    except (FileNotFoundError, KeyError, ValueError) as e:
        logger.error(f"Kritik bir hata oluştu: {e}", exc_info=True)
        # Hata durumunda programı sonlandır
        return
    except Exception as e:
        logger.error(f"Beklenmedik bir hata oluştu: {e}", exc_info=True)
        raise

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Yapılandırılabilir LLM eğitim betiği."
    )
    parser.add_argument(
        "--config",
        type=str,
        default="config.yaml",
        help="Yapılandırma dosyasının yolu (varsayılan: config.yaml)."
    )
    parser.add_argument(
        "--resume_from_checkpoint",
        action="store_true",
        help="Eğitime en son kontrol noktasından devam etmek için bu bayrağı ekleyin."
    )
    args = parser.parse_args()

    # `resume_from_checkpoint` argümanını bool'a çevir
    # Eğer flag varsa True, yoksa en son checkpoint'i aramak yerine False olacak.
    # Trainer, `True` değeri aldığında en son checkpoint'i otomatik bulur.
    main(
        config_path=args.config,
        resume_from_checkpoint=args.resume_from_checkpoint
    )
