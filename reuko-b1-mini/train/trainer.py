from transformers import Trainer, TrainingArguments, DataCollatorForLanguageModeling
from utils.logging_utils import get_logger

logger = get_logger(__name__)

def get_trainer(model, tokenizer, train_dataset, val_dataset, training_args_dict: dict, deepspeed_config: str = None):
    """
    Eğitim için bir Hugging Face Trainer nesnesi oluşturur ve yapılandırır.

    Args:
        model: Eğitilecek model.
        tokenizer: Tokenizer nesnesi.
        train_dataset: Eğitim veri seti.
        val_dataset: Doğrulama veri seti.
        training_args_dict (dict): TrainingArguments için yapılandırma sözlüğü.
        deepspeed_config (str, optional): DeepSpeed yapılandırma dosyasının yolu.

    Returns:
        Trainer: Yapılandırılmış Trainer nesnesi.
    """
    logger.info("TrainingArguments oluşturuluyor...")

    # DeepSpeed yapılandırmasını ekle
    if deepspeed_config:
        training_args_dict['deepspeed'] = deepspeed_config

    # Desteklenmeyen argümanları kaldır
    unsupported_args = ['evaluation_strategy', 'eval_steps', 'save_strategy', 'save_steps']
    for arg in unsupported_args:
        if arg in training_args_dict:
            logger.warning(f"Desteklenmeyen argüman kaldırılıyor: {arg}")
            training_args_dict.pop(arg)

    training_args = TrainingArguments(**training_args_dict)
    logger.info(f"Eğitim argümanları: {training_args.to_json_string()}")

    # Dil modellemesi için veri harmanlayıcı (data collator)
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False  # Masked Language Modeling değil, Causal Language Modeling yapıyoruz
    )
    logger.info("Data collator oluşturuldu.")

    # Trainer nesnesini oluştur
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        tokenizer=tokenizer,
        data_collator=data_collator,
    )
    logger.info("Trainer başarıyla oluşturuldu.")

    return trainer
