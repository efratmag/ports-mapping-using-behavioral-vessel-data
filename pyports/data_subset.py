"""
script that takes only subset of data-
1. only three vessels categories: tankers, cargo_containers, cargo_other
2. only anchoring
"""

import pandas as pd
import os

FILE_NAME = 'all_activities.csv.gz'  # df of all activities
PATH = '/Users/EF/PycharmProjects/ports-mapping-using-behavioral-vessel-data/features' # features folder

# read full df in chunks
tfr = pd.read_csv(os.path.join(PATH, FILE_NAME),
                  compression='gzip',
                  low_memory=False,
                  chunksize=1000,
                  iterator=True)
df = pd.concat(tfr, ignore_index=True)

df_sub = df[df.activity=='anchoring']

df_for_clustering = df_sub.loc[:, ['firstBlip_lng', 'firstBlip_lat']]

df_for_clustering.to_csv(os.path.join(PATH, 'df_for_clustering.csv'))

