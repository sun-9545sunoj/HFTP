import urllib.request
import os
import subprocess

print("=========================================================")
print("  Downloading Multiple LOBSTER Benchmark Stocks (10L)    ")
print("=========================================================\n")

base_url = "https://huggingface.co/datasets/totalorganfailure/lobster-data/resolve/main/LOBSTER_SampleFile_{stock}_2012-06-21_10/{stock}_2012-06-21_34200000_57600000_{ftype}_10.csv"
stocks = ["AMZN", "GOOG", "MSFT", "INTC"]
data_dir = "src/data/"
os.makedirs(data_dir, exist_ok=True)

for stock in stocks:
    print(f"-> Downloading {stock}...")
    orderbook_url = base_url.format(stock=stock, ftype="orderbook")
    message_url = base_url.format(stock=stock, ftype="message")
    
    orderbook_path = os.path.join(data_dir, f"LOBSTER_{stock}_orderbook.csv")
    message_path = os.path.join(data_dir, f"LOBSTER_{stock}_message.csv")
    
    if not os.path.exists(orderbook_path):
        subprocess.run(["curl", "-L", "-o", orderbook_path, orderbook_url], check=True)
    if not os.path.exists(message_path):
        subprocess.run(["curl", "-L", "-o", message_path, message_url], check=True)
    print(f"   [DONE] {stock}")

print("\n=========================================================")
print("  Downloads Complete! Proceeding to Data Processing... ")
print("=========================================================")
