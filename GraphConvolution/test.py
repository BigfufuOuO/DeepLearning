import pandas as pd
import os

dataset = 'citeseer'
current_path = os.path.abspath(os.path.dirname(__file__))
print("current_path:", current_path)
data_path = os.path.join(current_path, 'data/')
data_path = os.path.join(data_path, dataset + '/' + dataset)

raw_data_content = pd.read_csv(data_path + '.content', sep='\t', header=None)
raw_data_cite = pd.read_csv(data_path + '.cites', sep='\t', header=None)

nodes = list(raw_data_content.iloc[:, 0])
print(len(nodes))
anothernodes = list(raw_data_cite.iloc[:, 0]) + list(raw_data_cite.iloc[:, 1])
unique_nodes = list(set(anothernodes))
print(len(unique_nodes))
del_nodes = []

for i in range(len(nodes)):
    nodes[i] = str(nodes[i])

for j in range(len(unique_nodes)):
    unique_nodes[j] = str(unique_nodes[j])
    
for unique_node in unique_nodes:
    if unique_node not in nodes:
        del_nodes.append(unique_node)
        
print(del_nodes)
        
with open(data_path + '.cites', 'r') as f:
    lines = f.readlines()
    for line in lines:
        if str(line.split('\t')[0]) in del_nodes or str(line.strip().split('\t')[1]) in del_nodes:
            lines.remove(line)
            
with open(data_path + '.cites', 'w') as f:
    f.writelines(lines)