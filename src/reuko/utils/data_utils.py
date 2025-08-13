from datasets import load_from_disk

def load_tokenized_dataset(path: str):
    return load_from_disk(path)
