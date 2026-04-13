import pandas as pd
import matplotlib.pyplot as plt

# 🔹 Put your FULL file path here (Windows format)
# file_path = r"C:\Users\Edawi\OneDrive\Desktop\work\csv\time.csv"
file_path = r"C:\Users\Edawi\OneDrive\Desktop\work\outputs\Map3\run_2026-04-02_120802_RL\timeseries.csv"

# Load CSV
df = pd.read_csv(file_path)

# Sort by step (important for clean graph)
df = df.sort_values("step")

# Create figure
fig, ax1 = plt.subplots(figsize=(12, 6))

# 🔹 Left axis → Cumulative Reward
ax1.plot(df["step"], df["cumulative_reward"], label="Cumulative Reward")
ax1.set_xlabel("Steps")
ax1.set_ylabel("Cumulative Reward")

# 🔹 Right axis → Queue + Vehicle Count
ax2 = ax1.twinx()
ax2.plot(df["step"], df["total_queue"], label="Total Queue")
ax2.plot(df["step"], df["vehicle_count"], label="Vehicle Count")
ax2.set_ylabel("Queue / Vehicle Count")

# 🔹 Combine legends from both axes
lines_1, labels_1 = ax1.get_legend_handles_labels()
lines_2, labels_2 = ax2.get_legend_handles_labels()
ax1.legend(lines_1 + lines_2, labels_1 + labels_2)

# Title + grid
plt.title("Simulation Metrics vs Steps (Dual Axis)")
plt.grid(True)

# Layout fix
plt.tight_layout()

# Show graph
plt.show()
