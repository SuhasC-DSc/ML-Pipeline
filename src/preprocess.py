import pandas as pd
import yaml
import os

params=yaml.safe_load(open("params.yaml"))["preprocess"]

def preprocess(input_path, output_path):
    df = pd.read_csv(input_path)
    
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df.to_csv(output_path, header=None, index=False)
    print(f"Preprocessed data saved to {output_path}")

if __name__ == "__main__":
    preprocess(input_path=params["input"], output_path=params["output"])