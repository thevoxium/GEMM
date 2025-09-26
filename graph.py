
import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("results.csv")

plt.figure(figsize=(8,6))
plt.plot(df["size"], df["naive"], marker="o", label="Naive")
plt.plot(df["size"], df["blocked"], marker="s", label="Blocked")
plt.plot(df["size"], df["packed"], marker="^", label="Packed")

plt.yscale("log")

plt.xlabel("Matrix Size (n = k = m)")
plt.ylabel("Time (ms)")
plt.title("GEMM Performance Comparison")
plt.legend()
plt.grid(True, which="both", ls="--", alpha=0.5)

plt.tight_layout()
plt.savefig("all_gemm_benchmark.png", dpi=300)
plt.show()
