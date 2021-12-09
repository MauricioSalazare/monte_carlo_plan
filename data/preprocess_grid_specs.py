import pandas as  pd
import numpy as np

file_name_network = r"raw_data/network_data/Nodes_34.csv"
grid = pd.read_csv(file_name_network)
n_nodes = len(grid)

np.random.seed(12345)
installed_pv_kwp = [25, 50, 75, 100, 125, 150, 200]  # Possible upper bound installed capacity in the nodes
random_pv_capacity = np.random.choice(installed_pv_kwp, n_nodes)

cluster_labels = [0, 1, 2]  # 3 Clusters of consumption
random_cluster_labels = np.random.choice(cluster_labels, n_nodes)

grid["cluster"] = random_cluster_labels
# grid["kwp"] = random_pv_capacity  # TODO: I tuned it this by hand (Do not overwrite)
#
# Reset slack node
# grid.loc[0, ["cluster", "kwp"]] = [0, 0]  # TODO: I tuned it this by hand (Do not overwrite)
#
# file_name_network_processed = r"processed_data/network_data/Nodes_34.csv"
# grid.to_csv(file_name_network_processed, index=False)