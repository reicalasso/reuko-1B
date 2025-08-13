import yaml
from utils.logging_utils import get_logger

logger = get_logger(__name__)

def load_config(config_path: str) -> dict:
    """
    Belirtilen yoldan YAML yapılandırma dosyasını yükler.

    Args:
        config_path (str): Yapılandırma dosyasının yolu.

    Returns:
        dict: Yüklenen yapılandırma verisi.
    
    Raises:
        FileNotFoundError: Belirtilen yolda dosya bulunamazsa.
        yaml.YAMLError: Dosya geçerli bir YAML formatında değilse.
    """
    logger.info(f"Yapılandırma dosyası yükleniyor: {config_path}")
    try:
        with open(config_path, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
        logger.info("Yapılandırma başarıyla yüklendi.")
        return config
    except FileNotFoundError:
        logger.error(f"Yapılandırma dosyası bulunamadı: {config_path}")
        raise
    except yaml.YAMLError as e:
        logger.error(f"YAML dosyasını okurken hata oluştu: {config_path}", exc_info=True)
        raise

def get_deepspeed_config(config: dict):
    """
    Yapılandırma dosyasından DeepSpeed yapılandırma yolunu alır.
    Eğer belirtilmemişse None döner.
    """
    return config.get("deepspeed_config", None)
