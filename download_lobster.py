import urllib.request
import os

print("=========================================================")
print("  Downloading LOBSTER Benchmark Data (AAPL 10-Levels)    ")
print("=========================================================\n")

base_url = "https://huggingface.co/datasets/totalorganfailure/lobster-data/resolve/main/LOBSTER_SampleFile_AAPL_2012-06-21_10/"
orderbook_url = base_url + "AAPL_2012-06-21_34200000_57600000_orderbook_10.csv"
message_url = base_url + "AAPL_2012-06-21_34200000_57600000_message_10.csv"

data_dir = "src/data/"
os.makedirs(data_dir, exist_ok=True)

orderbook_path = os.path.join(data_dir, "LOBSTER_AAPL_orderbook.csv")
message_path = os.path.join(data_dir, "LOBSTER_AAPL_message.csv")

print("-> Downloading Orderbook file... (This may take a minute)")
urllib.request.urlretrieve(orderbook_url, orderbook_path)
print(f"   Saved to: {orderbook_path}")

print("\n-> Downloading Message file...")
urllib.request.urlretrieve(message_url, message_path)
print(f"   Saved to: {message_path}")

print("\n=========================================================")
print("  Data Download Complete! Ready for Model Generalization ")
print("=========================================================")
