import argparse
import pandas as pd
import matplotlib.pyplot as plt

def plot_data(file_path, x_column, y_columns):
    # Load the dataset
    if file_path.endswith('.csv'):
        df = pd.read_csv(file_path)
    elif file_path.endswith('.json'):
        df = pd.read_json(file_path, lines=True)
    else:
        raise ValueError("File must be a .csv or .json")

    # Add derived accuracy columns
    df["train_accuracy"] = 100 - df["train_class_error"]
    df["test_accuracy"] = 100 - df["test_class_error"]
    
    # Plot the data
    plt.figure(figsize=(10, 6))
    for y_col in y_columns:
        if y_col not in df.columns:
            print(f"[WARNING] '{y_col}' not found in file columns.")
            continue
        plt.plot(df[x_column], df[y_col], label=y_col)

    plt.xlabel(x_column)
    plt.ylabel("Value")
    plt.title("Log Data Plot")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Plot selected log metrics.")
    parser.add_argument("--file", type=str, required=True, help="Path to CSV or JSON file")
    parser.add_argument("--x", type=str, required=True, help="Column to use as x-axis")
    parser.add_argument("--y", nargs='+', required=True, help="Column(s) to plot as y-axis")

    args = parser.parse_args()
    plot_data(args.file, args.x, args.y)
