"""
ONLY for testing the code when you do not have the real dataset yet.
Creates a small fake dataset (CICIoT2023-style columns) in ./dataset_demo
Run:  python make_demo_data.py   then   python run_all.py --data dataset_demo --frac 1
DO NOT use results from this data in your paper - use the real dataset.
"""
from pathlib import Path
import numpy as np
import pandas as pd

COLS = ["flow_duration","Header_Length","Protocol Type","Duration","Rate","Srate","Drate","fin_flag_number",
        "syn_flag_number","rst_flag_number","psh_flag_number","ack_flag_number","ece_flag_number","cwr_flag_number",
        "ack_count","syn_count","fin_count","urg_count","rst_count","HTTP","HTTPS","DNS","Telnet","SMTP","SSH","IRC",
        "TCP","UDP","DHCP","ARP","ICMP","IGMP","IPv","LLC","Tot sum","Min","Max","AVG","Std","Tot size","IAT",
        "Number","Magnitue","Radius","Covariance","Variance","Weight"]
ATTACKS = ["DDoS-ICMP_Flood","DDoS-UDP_Flood","DoS-TCP_Flood","Mirai-greeth_flood","Recon-PortScan","DictionaryBruteForce"]

rng = np.random.RandomState(0)
out = Path(__file__).resolve().parent / "dataset_demo"
out.mkdir(exist_ok=True)
for i in range(3):
    n = 15000
    labels = rng.choice(["BENIGN"] + ATTACKS, n, p=[0.08] + [0.92 / len(ATTACKS)] * len(ATTACKS))
    X = rng.gamma(2.0, 2.0, (n, len(COLS)))
    for j, l in enumerate(ATTACKS, 1):
        m = labels == l
        X[m, j * 3:(j * 3) + 4] += rng.normal(2 + j, 1.5, (m.sum(), 4))
    X[labels == "BENIGN", :6] *= 0.7
    df = pd.DataFrame(X, columns=COLS)
    df.loc[rng.rand(n) < 0.01, "Std"] = np.inf
    df.loc[rng.rand(n) < 0.01, "Variance"] = np.nan
    df["label"] = labels
    df.to_csv(out / f"Merged{i+1:02d}.csv", index=False)
print("Demo data written to", out)
