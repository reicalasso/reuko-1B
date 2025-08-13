import logging

def setup_logging():
    """
    Proje genelinde kullanılacak temel günlükleme (logging) yapılandırmasını ayarlar.
    """
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    # Transformers kütüphanesinin log seviyesini düşürerek daha temiz bir çıktı sağla
    logging.getLogger("transformers").setLevel(logging.WARNING)

def get_logger(name: str):
    """
    Belirtilen isimle yeni bir logger nesnesi oluşturur veya mevcut olanı döndürür.

    Args:
        name (str): Logger'a verilecek isim. Genellikle __name__ kullanılır.

    Returns:
        logging.Logger: Yapılandırılmış logger nesnesi.
    """
    return logging.getLogger(name)
