"""
Reuko-1B Usage Examples
"""

from reuko_1b import ReukoInference, ConfigManager, T5QAModel, T5SummarizationModel

# =================
# 1. HIZLI BAŞLANGIÇ
# =================

def quick_start():
    """En basit kullanım"""
    
    # Inference pipeline
    inference = ReukoInference()
    
    # QA
    qa_result = inference.answer_question(
        question="What is artificial intelligence?",
        context="Artificial intelligence (AI) is intelligence demonstrated by machines, in contrast to the natural intelligence displayed by humans."
    )
    print(f"Answer: {qa_result['answer']}")
    
    # Summarization
    text = """
    Machine learning is a method of data analysis that automates analytical model building. 
    It is a branch of artificial intelligence based on the idea that systems can learn from data, 
    identify patterns and make decisions with minimal human intervention.
    """
    
    summary_result = inference.summarize_text(text)
    print(f"Summary: {summary_result['summary']}")

# =================
# 2. 1B MODEL KULLANIMI  
# =================

def use_1b_model():
    """1B parametre model kullanımı"""
    
    # 1B Model ile QA
    qa_model = T5QAModel(use_1b_model=True)
    
    result = qa_model.answer_question(
        "What is the capital of France?",
        "France is a country in Western Europe. Paris is the capital and most populous city."
    )
    
    print(f"1B Model Answer: {result['answer']}")
    print(f"Model Parameters: {result['model_parameters']:,}")
    print(f"Memory Usage: {result['model_memory_gb']:.2f} GB")

# =================
# 3. BATCH PROCESSING
# =================

def batch_processing():
    """Toplu işleme"""
    
    inference = ReukoInference()
    
    # Batch QA
    questions = [
        "What is Python?",
        "Who created Linux?",
        "When was the internet invented?"
    ]
    
    contexts = [
        "Python is a high-level programming language.",
        "Linux was created by Linus Torvalds in 1991.",
        "The internet was invented in the late 1960s by ARPANET."
    ]
    
    batch_results = inference.batch_qa(questions, contexts)
    
    for i, result in enumerate(batch_results):
        print(f"Q{i+1}: {result['question']}")
        print(f"A{i+1}: {result['answer']}\n")
    
    # Batch Summarization
    articles = [
        "Long article 1 about technology...",
        "Long article 2 about science...",
        "Long article 3 about business..."
    ]
    
    summaries = inference.batch_summarization(articles)
    
    for i, summary in enumerate(summaries):
        print(f"Article {i+1} Summary: {summary['summary']}")

# =================
# 4. CUSTOM CONFIG
# =================

def custom_configuration():
    """Özel konfigürasyon"""
    
    # Config dosyası yükle
    config_manager = ConfigManager("config.yaml")
    config = config_manager.config
    
    # Config'i değiştir
    config.model.temperature = 0.5
    config.model.top_p = 0.8
    config.training.batch_size = 2
    
    # Model oluştur
    inference = ReukoInference(config)
    
    # Test et
    result = inference.answer_question(
        "Test question?",
        "Test context information here."
    )
    
    print(f"Custom config result: {result}")

# =================
# 5. MULTI-TASK MODEL
# =================

def multi_task_usage():
    """Multi-task model kullanımı"""
    
    from reuko_1b.models.t5_model import T5MultiTaskModel
    
    # Multi-task model
    model = T5MultiTaskModel(use_1b_model=True)
    
    # QA task
    qa_result = model.process_task(
        task_type="qa",
        question="What is machine learning?",
        context="Machine learning is a subset of AI that enables computers to learn without explicit programming."
    )
    
    print(f"Multi-task QA: {qa_result}")
    
    # Summarization task
    sum_result = model.process_task(
        task_type="summarization",
        text="Long text to be summarized goes here..."
    )
    
    print(f"Multi-task Summary: {sum_result}")

# =================
# 6. TRAINING PIPELINE
# =================

def custom_training():
    """Custom training pipeline"""
    
    from reuko_1b import ReukoTrainer, ConfigManager
    
    # Config yükle
    config_manager = ConfigManager("config.yaml")
    config = config_manager.config
    
    # Training parametrelerini özelleştir
    config.training.num_epochs = 1
    config.data.qa_train_size = 1000  # Küçük test için
    
    # Trainer oluştur
    trainer = ReukoTrainer(config)
    
    # Model bilgilerini göster
    model_info = trainer.get_model_info()
    print(f"Model Info: {model_info}")
    
    # QA training (test için küçük dataset)
    # results = trainer.train_qa()
    # print(f"Training Results: {results}")

# =================
# MAIN EXECUTION
# =================

if __name__ == "__main__":
    print("=== Reuko-1B Usage Examples ===\n")
    
    print("1. Quick Start:")
    quick_start()
    print("\n" + "="*50 + "\n")
    
    print("2. Batch Processing:")
    batch_processing()
    print("\n" + "="*50 + "\n")
    
    print("3. Custom Configuration:")
    custom_configuration()
    print("\n" + "="*50 + "\n")
    
    # 1B model examples (GPU gerekli)
    if torch.cuda.is_available():
        print("4. 1B Model Usage:")
        use_1b_model()
        print("\n" + "="*50 + "\n")
        
        print("5. Multi-task Usage:")
        multi_task_usage()
    else:
        print("GPU not available - skipping 1B model examples")
    
    print("\n🎉 All examples completed!")
