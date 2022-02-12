import numpy as np
from generator import generate_nodes
from tqdm import tqdm

epss = np.logspace(-4, -2, 30)
epss_stat = np.zeros_like(epss)
ks = np.linspace(128, 2048, 30).astype(int)
ks_stat = np.zeros_like(ks, dtype=float)
hs = np.linspace(4, 64, 30).astype(int)
hs_stat = np.zeros_like(hs, dtype=float)
q = 0.25
n = 1_000_000
repetitions = 30

split_deviation = 0.0 #n // 10
rank = q * n

print("Generating random data")
data = np.random.standard_normal(n)

print("Sorting the data")
sorted_data = np.sort(data)

print("Started testing number of nodes (k) in the network")
for i, k in tqdm(enumerate(ks)):
    for _ in range(repetitions):
        network, root, data, height = generate_nodes(n, k, 8, data, split_deviation=split_deviation, graph_partition=True)
        network.reset()

        root.new_task(epss[0])
        value, rank = root.rank_query(rank, return_rank=True)
        ks_stat[i] += network.non_system_stats
        # print(network.stat)
ks_stat /= repetitions
print(ks_stat)
