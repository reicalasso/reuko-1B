from transformers import GPT2LMHeadModel, GPT2Config
from utils.logging_utils import get_logger

logger = get_logger(__name__)

def build_model(tokenizer, model_config: dict):
    """
    Verilen yapılandırma ile yeni bir GPT-2 modeli oluşturur.

    Args:
        tokenizer: Eğitim için kullanılacak tokenizer.
        model_config (dict): Modelin katman sayısı, gizli boyut gibi
                             parametrelerini içeren sözlük.

    Returns:
        GPT2LMHeadModel: Yapılandırılmış ve başlatılmamış (untrained) model.
    """
    logger.info("Model yapılandırması oluşturuluyor...")
    config = GPT2Config(
        vocab_size=len(tokenizer),
        n_positions=model_config.get('n_positions', 1024),
        n_ctx=model_config.get('n_ctx', 1024),
        n_embd=model_config.get('n_embd', 768),
        n_layer=model_config.get('n_layer', 12),
        n_head=model_config.get('n_head', 12),
        # Diğer önemli parametreler buraya eklenebilir
    )
    logger.info(f"Model yapılandırması tamamlandı: {config.to_dict()}")

    logger.info("GPT2LMHeadModel oluşturuluyor...")
    model = GPT2LMHeadModel(config)
    logger.info(f"Model başarıyla oluşturuldu. Parametre sayısı: {model.num_parameters():,}")
    
    return model
