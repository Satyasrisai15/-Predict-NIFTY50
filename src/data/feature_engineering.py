import os
from pathlib import Path
from typing import Union

import pandas as pd
from tqdm import tqdm

DAYS_3M = 63
DAYS_1Y = 252
DAYS_3Y = 756

BASE_DIR = Path(__file__).parent.parent.parent
RAW_DIR = BASE_DIR / "data" / "raw"
PROCESSED_DIR = BASE_DIR / "data" / "processed"

def load_stock_csv(filepath: Union[str, Path]) -> pd.DataFrame:
    df = pd.read_csv(filepath, index_col=0, parse_dates=True)
    return df


def create_target_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    
    df["Target_3M"] = df["Close"].shift(-DAYS_3M)
    df["Target_1Y"] = df["Close"].shift(-DAYS_1Y)
    df["Target_3Y"] = df["Close"].shift(-DAYS_3Y)
    
    return df


def calculate_returns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    
    df["Return_3M"] = (df["Target_3M"] - df["Close"]) / df["Close"]
    df["Return_1Y"] = (df["Target_1Y"] - df["Close"]) / df["Close"]
    df["Return_3Y"] = (df["Target_3Y"] - df["Close"]) / df["Close"]
    
    return df


def process_stock(input_path: Union[str, Path], output_path: Union[str, Path]) -> bool:

    try:
        df = load_stock_csv(input_path)
        
        if "Close" not in df.columns:
            print(f"  ⚠️  Warning: 'Close' column not found in {input_path}")
            return False
        
        df = create_target_columns(df)
        df = calculate_returns(df)
        df.to_csv(output_path)
        return True
        
    except Exception as e:
        print(f"  ❌ Error processing {input_path}: {e}")
        return False


def process_all_stocks():
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    
    csv_files = list(RAW_DIR.glob("*.csv"))
    
    if not csv_files:
        print(f"No CSV files found in {RAW_DIR}")
        return
    
    print(f"Processing {len(csv_files)} stock files...")
    print(f"Input directory: {RAW_DIR}")
    print(f"Output directory: {PROCESSED_DIR}\n")
    
    successful = 0
    failed = []
    
    for csv_file in tqdm(csv_files, desc="Processing stocks", unit="stock"):
        output_path = PROCESSED_DIR / csv_file.name
        
        if process_stock(csv_file, output_path):
            successful += 1
        else:
            failed.append(csv_file.name)
    
    print(f"\n{'='*50}")
    print(f"Processing complete!")
    print(f"  ✅ Successfully processed: {successful}/{len(csv_files)} stocks")
    
    if failed:
        print(f"  ❌ Failed to process: {len(failed)} stocks")
        print(f"     Failed files: {', '.join(failed)}")
    
    if successful > 0:
        sample_file = next(PROCESSED_DIR.glob("*.csv"))
        sample_df = pd.read_csv(sample_file, index_col=0, parse_dates=True)
        print(f"\nNew columns added:")
        print(f"  - Target_3M, Target_1Y, Target_3Y (future prices)")
        print(f"  - Return_3M, Return_1Y, Return_3Y (percentage returns)")
        print(f"\nSample from {sample_file.name}:")
        print(sample_df[["Close", "Target_3M", "Return_3M", "Target_1Y", "Return_1Y"]].head(3).to_string())


def process_single_stock(ticker: str):
    input_path = RAW_DIR / f"{ticker}.csv"
    output_path = PROCESSED_DIR / f"{ticker}.csv"
    
    if not input_path.exists():
        print(f"❌ File not found: {input_path}")
        return
    
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    
    print(f"Processing {ticker}...")
    if process_stock(input_path, output_path):
        print(f"✅ Saved to {output_path}")
        
        df = pd.read_csv(output_path, index_col=0, parse_dates=True)
        print(f"\nColumns: {list(df.columns)}")
        print(f"\nSample data:")
        print(df[["Close", "Target_3M", "Return_3M", "Target_1Y", "Return_1Y"]].head(5).to_string())
    else:
        print(f"❌ Failed to process {ticker}")


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1:
        ticker = sys.argv[1]
        process_single_stock(ticker)
    else:
        process_all_stocks()
