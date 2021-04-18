"""
script that takes only subset of data-
1. only three vessels categories: tankers, cargo_containers, cargo_other
2. only anchoring
"""

import pandas as pd
import os

FILE_NAME = 'all_activities.csv.gz'  # df of all activities
PATH = '/Users/EF/PycharmProjects/ports-mapping-using-behavioral-vessel-data/features' # features folder
ACTIVITY = 'anchoring'

# read full df in chunks
tfr = pd.read_csv(os.path.join(PATH, FILE_NAME),
                  compression='gzip',
                  low_memory=False,
                  chunksize=1000,
                  iterator=True)
df = pd.concat(tfr, ignore_index=True)

df_sub = df[df.activity == ACTIVITY]

cols = ['_id', 'vesselId', 'startDate', 'endDate', 'duration', 'firstBlip_lng',
       'firstBlip_lat', 'lastBlip_lng', 'lastBlip_lat', 'vessel_class_calc',
       'vessel_subclass_documented', 'vessel_deadweight', 'vessel_size', 'vessel_draught', 'firstBlip_polygon_id',
        'firstBlip_polygon_area_type', 'lastBlip_polygon_id', 'lastBlip_polygon_area_type',
       'firstBlip_in_polygon', 'lastBlip_in_polygon', 'polygonId', 'polygonType']

df_for_clustering = df_sub.loc[:, cols]

# change class categories
conditions = [
        (df_for_clustering["vessel_class_calc"] == 'Cargo') & (df_for_clustering["vessel_subclass_documented"] == 'Container Vessel'),
        (df_for_clustering["vessel_class_calc"] == 'Cargo') & (df_for_clustering["vessel_subclass_documented"] != 'Container Vessel'),
        (df_for_clustering["vessel_class_calc"] == 'Tanker')
    ]
choices = ["cargo_container", "cargo_other", "tanker"]
df_for_clustering["class_new"] = np.select(conditions, choices)

df_for_clustering = df_for_clustering[df_for_clustering.class_new != '0'] # take only cargo and tanker vessels

df_for_clustering.to_csv(os.path.join(PATH, f'df_for_clustering_{ACTIVITY}.csv'))

