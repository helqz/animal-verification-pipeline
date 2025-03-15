from datasets import load_dataset, ClassLabel, Sequence, load_from_disk
from pathlib import Path

def download_and_process_data():
    # Download raw data
    dataset = load_dataset('conll2003', trust_remote_code=True, cache_dir='../data/ner_raw')

    ANIMALS = {"dog", "cat", "elephant", "horse", "spider", "cow", "sheep"}

    def convert_to_iob(example):
        new_tags = []
        for token in example['tokens']:
            # Check if token is an animal (case-insensitive)
            if token.lower() in ANIMALS:
                new_tags.append(1)  # B-ANIMAL
            else:
                new_tags.append(0)  # O
        return {'ner_tags': new_tags}

    # Process dataset
    dataset = dataset.map(convert_to_iob)
    
    # Update features
    new_labels = ["O", "B-ANIMAL"]
    for split in dataset:
        features = dataset[split].features
        features["ner_tags"] = Sequence(ClassLabel(names=new_labels))
        dataset[split] = dataset[split].cast(features)

    dataset.save_to_disk('data/ner')

def load_processed_data(processed_dir='data/ner'):
    download_and_process_data()
    processed_path = Path(processed_dir)

    if not processed_path.exists():
        raise FileNotFoundError(f"No processed data found at {processed_path}. ")
    
    dataset = load_from_disk(processed_dir)

    return dataset

