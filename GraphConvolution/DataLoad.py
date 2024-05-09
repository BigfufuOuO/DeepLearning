# Get data from dataset
import panda as pd
import numpy as np

Cora_raw_data_content = pd.read_csv('data/cora/cora.content', sep='\t', header=None)
Cora_raw_data_site = pd.read_csv('data/cora/cora.cites', sep='\t', header=None)

