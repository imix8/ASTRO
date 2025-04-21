import json
import pandas as pd

def parse_log_file(file_path):
    records = []

    with open(file_path, 'r') as f:
        for line in f:
            try:
                log_entry = json.loads(line.strip())
                records.append(log_entry)
            except json.JSONDecodeError as e:
                print(f"Failed to parse line: {e}")

    return pd.DataFrame(records)

if __name__ == "__main__":
    log_file = "./logs/log.txt"  # replace with your actual log file name
    df = parse_log_file(log_file)

    # Preview the dataframe
    print(df.head())

    # Save to CSV if needed
    df.to_csv("parsed_logs.csv", index=False)
