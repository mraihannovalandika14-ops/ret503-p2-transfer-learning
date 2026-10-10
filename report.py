import json, matplotlib.pyplot as plt
rows = []
plt.figure(figsize=(6, 4))
for mode in ["feature", "partial", "scratch"]:
    with open(f"results/{mode}.json") as f: r = json.load(f)
    rows.append(f"| {mode} | {r['best_val_acc']*100:.1f}% | {r['train_time_s']}s | {r['epoch_acc90'] or '-'} |")
    plt.plot([h["epoch"] for h in r["history"]], [h["val_acc"] for h in r["history"]], marker="o", label=mode)
plt.xlabel("Epoch"); plt.ylabel("Akurasi val"); plt.legend(); plt.grid(alpha=.3); plt.tight_layout()
plt.savefig("results/akurasi_per_epoch.png", dpi=150)
tabel = "| Mode | Akurasi val terbaik | Waktu latih | Epoch acc ≥ 90% |\n|---|---|---|---|\n" + "\n".join(rows)
with open("results/tabel.md", "w", encoding="utf-8") as f: f.write(tabel)
print(tabel)