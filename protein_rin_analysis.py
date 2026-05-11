import MDAnalysis as mda
from MDAnalysis.analysis import distances
import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

systems = {
    "WT": ("protein_4fdd.tpr", "4fdd.xtc"),
    "R521C": ("protein_r521c.tpr", "r521c.xtc"),
    "R521H": ("protein_r521h.tpr", "r521h.xtc"),
    "P525L": ("protein_p525l.tpr", "p525l.xtc"),
}

cutoff = 5.0
stride = 10
contact_threshold = 0.3


def build_network(tpr, xtc):
    u = mda.Universe(tpr, xtc)
    ca = u.select_atoms("protein and name CA")

    resids = ca.resids
    n_frames = len(u.trajectory[::stride])

    contact_counts = {}

    for ts in u.trajectory[::stride]:
        coords = ca.positions
        dist = distances.distance_array(coords, coords)

        for i in range(len(resids)):
            for j in range(i+1, len(resids)):
                if dist[i, j] < cutoff:
                    key = (resids[i], resids[j])
                    contact_counts[key] = contact_counts.get(key, 0) + 1

    G = nx.Graph()

    for (i, j), count in contact_counts.items():
        if count >= contact_threshold * n_frames:
            G.add_edge(i, j)

    return G

u_ref = mda.Universe(*systems["WT"])  #wild_type as reference
ca_ref = u_ref.select_atoms("protein and name CA")
all_res = np.unique(ca_ref.resids)

results = {}

for name, (tpr, xtc) in systems.items():
    print(f"\nProcessing {name}...")

    G = build_network(tpr, xtc)
    betweenness = nx.betweenness_centrality(G)

    results[name] = betweenness

    top = sorted(betweenness.items(), key=lambda x: x[1], reverse=True)[:10]
    print(f"\nTop residues ({name}):")
    for r, val in top:
        print(f"Residue {r}: {val:.4f}")

heatmap_data = []

for name in results:
    row = []
    for r in all_res:
        row.append(results[name].get(r, 0))
    heatmap_data.append(row)

heatmap_data = np.array(heatmap_data)

plt.figure(figsize=(12,4))

sns.heatmap(
    heatmap_data,
    cmap="viridis",
    yticklabels=list(results.keys()),
    xticklabels=50  # show every 50 residues
)

plt.xlabel("Residue Index")
plt.ylabel("System")
plt.title("RIN Betweenness Centrality Heatmap")
plt.tight_layout()
plt.savefig("RIN_heatmap.png", dpi=600)
plt.close()


plt.figure(figsize=(10,6))

for name in results:
    vals = np.array([results[name].get(r, 0) for r in all_res])

    window = 15
    smooth_vals = np.convolve(vals, np.ones(window)/window, mode='same')

    plt.plot(all_res, smooth_vals, label=name)

plt.xticks(np.arange(all_res.min(), all_res.max(), 100))

plt.xlabel("Residue ID")
plt.ylabel("Betweenness Centrality (Smoothed)")
plt.title("RIN Comparison (Smoothed)")
plt.legend()
plt.tight_layout()
plt.savefig("RIN_lineplot.png", dpi=600)
plt.close()


for name in results:
    with open(f"{name}_centrality.txt", "w") as f:
        for r in all_res:
            f.write(f"{r} {results[name].get(r, 0)}\n")

print("\nAnalysis complete.")
print("Generated:")
print("- RIN_heatmap.png")
print("- RIN_lineplot.png")
print("- *_centrality.txt")
