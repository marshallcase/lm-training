# scripts/prepare_wikitext_hf.py
from datasets import load_dataset
import os
import argparse

def prepare_wikitext(config="wikitext-103-v1", output_dir="data/processed"):
    """
    Download and prepare WikiText dataset from Hugging Face.
    
    Args:
        config: Dataset configuration (wikitext-103-v1, wikitext-2-v1)
        output_dir: Directory to save processed data
    """
    print(f"Loading {config} dataset from Hugging Face...")
    
    # Load dataset
    dataset = load_dataset("wikitext", config)
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Process train split
    train_text = "\n\n".join([text for text in dataset["train"]["text"] if text.strip()])
    train_path = os.path.join(output_dir, "train.txt")
    with open(train_path, 'w', encoding='utf-8') as f:
        f.write(train_text)
    
    # Process validation split
    valid_text = "\n\n".join([text for text in dataset["validation"]["text"] if text.strip()])
    valid_path = os.path.join(output_dir, "valid.txt")
    with open(valid_path, 'w', encoding='utf-8') as f:
        f.write(valid_text)
    
    # Process test split (optional)
    test_text = "\n\n".join([text for text in dataset["test"]["text"] if text.strip()])
    test_path = os.path.join(output_dir, "test.txt")
    with open(test_path, 'w', encoding='utf-8') as f:
        f.write(test_text)
    
    # Print statistics
    print(f"Dataset prepared successfully:")
    print(f"  Training: {len(train_text)/1e6:.2f}M chars, ~{len(train_text.split())/1e3:.1f}K words")
    print(f"  Validation: {len(valid_text)/1e6:.2f}M chars, ~{len(valid_text.split())/1e3:.1f}K words")
    print(f"  Test: {len(test_text)/1e6:.2f}M chars, ~{len(test_text.split())/1e3:.1f}K words")
    print(f"\nFiles saved to:")
    print(f"  {train_path}")
    print(f"  {valid_path}")
    print(f"  {test_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Prepare WikiText dataset from Hugging Face")
    parser.add_argument("--config", type=str, default="wikitext-103-v1", 
                       choices=["wikitext-103-v1", "wikitext-2-v1"],
                       help="Dataset configuration")
    parser.add_argument("--output_dir", type=str, default="data/processed", 
                       help="Output directory")
    args = parser.parse_args()
    
    prepare_wikitext(args.config, args.output_dir)