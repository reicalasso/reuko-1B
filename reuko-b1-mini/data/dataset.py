from transformers import AutoTokenizer
from datasets import load_dataset
from utils.logging_utils import get_logger

logger = get_logger(__name__)


def get_tokenizer(model_name_or_path: str = "gpt2"):
    """
    Belirtilen model için bir tokenizer yükler ve yapılandırır.
    Pad token'ını EOS token'ı olarak ayarlar.
    """
    logger.info(f"Tokenizer yükleniyor: {model_name_or_path}")
    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
    tokenizer.pad_token = tokenizer.eos_token
    logger.info("Tokenizer başarıyla yüklendi ve yapılandırıldı.")
    return tokenizer


def load_and_tokenize_dataset(dataset_config: dict, tokenizer):
    """
    Veri setini yükler, tokenize eder ve eğitim için hazırlar.
    """
    logger.info(f"Veri seti yükleniyor: {dataset_config['name']} ({dataset_config.get('subset')})")
    try:
        dataset = load_dataset(dataset_config['name'], dataset_config.get('subset'))
    except Exception as e:
        logger.error(f"Veri seti yüklenirken hata oluştu: {e}", exc_info=True)
        raise

    max_length = dataset_config.get('max_length', 1024)

    def tokenize_function(examples):
        # Veri setindeki metin sütununu tokenize et
        return tokenizer(
            examples["text"],
            truncation=True,
            max_length=max_length,
            padding="max_length",  # Tüm örnekleri aynı uzunluğa getirmek için
        )

    logger.info("Veri seti tokenize ediliyor...")
    # 'text' sütunu dışındaki tüm sütunları kaldırarak belleği verimli kullan
    tokenized_dataset = dataset.map(
        tokenize_function,
        batched=True,
        remove_columns=dataset["train"].column_names
    )
    logger.info("Veri seti başarıyla tokenize edildi.")

    if 'train' not in tokenized_dataset or 'validation' not in tokenized_dataset:
        logger.error("Tokenize edilmiş veri setinde 'train' veya 'validation' bölünmeleri bulunamadı.")
        raise ValueError("Veri setinde 'train' ve 'validation' bölünmeleri olmalıdır.")

    return tokenized_dataset['train'], tokenized_dataset['validation']
