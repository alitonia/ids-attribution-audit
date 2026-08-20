import urllib.request
import json
import pandas as pd
from pathlib import Path

# Note: In a real scenario we could fetch from HF or an official download link.
# We will use huggingface_hub to fetch the filtered 5% version of CICIoT2023.

def prepare():
    out_dir = Path(__file__).resolve().parents[3] / "data" / "ciciot2023"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_file = out_dir / "ciciot2023_5percent.parquet"
    
    if out_file.exists():
        print(f"File {out_file} already exists.")
        return

    print("Fetching CICIoT2023 5% sample from Hugging Face...")
    try:
        from datasets import load_dataset
        
        print("Downloading casscloud/CIC-IoT-2023 dataset...")
        # Use a small subset (e.g. random split)
        ds = load_dataset("casscloud/CIC-IoT-2023", "random", split="train")
        
        print("Converting to pandas...")
        df = ds.to_pandas()
        
        # Ensure it has 'label'
        if 'label' not in df.columns and 'label' in ds.features:
            pass # should be there
            
        # The dataset has 'label' as an integer mapping.
        # Let's map it back or keep it. Actually, wait!
        # The original code expects 'Attack' column and binary 'label' column.
        # casscloud provides 'Label' (0-33) and we can map >0 to 1 for binary.
        # Let's just save it exactly as the model expects.
        if 'Label' in df.columns and 'label' not in df.columns:
            df['label'] = (df['Label'] > 0).astype(int)
            df['Attack'] = df['Label'].astype(str)
            
        df.to_parquet(out_file)
        print(f"Successfully saved {len(df)} rows to {out_file}")
    except Exception as e:
        import traceback
        traceback.print_exc()
        print(f"Error downloading dataset: {e}")

if __name__ == "__main__":
    prepare()
