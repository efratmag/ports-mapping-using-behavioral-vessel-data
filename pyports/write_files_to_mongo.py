import pandas as pd
import pymongo
import boto3
import sys
from bson import ObjectId
from shapely.geometry import Point

client = pymongo.MongoClient('mongodb://alon:winward1234@getstarted-ec2.cluster-cqssxwivnzqi.eu-central-1.docdb.amazonaws.com:27017/?ssl=true&ssl_ca_certs=rds-combined-ca-bundle.pem&replicaSet=rs0&readPreference=secondaryPreferred&retryWrites=false')
db = client.activities
col = db.drifting

s3 = boto3.resource(
    service_name='s3',
    region_name='us-east-2')
obj = s3.Bucket('windward-project').Object('drifting.csv').get()

df = pd.read_csv(obj['Head'])

df['_id'] = df['_id'].map(ObjectId)
df['vesselId'] = df['vesselId'].map(ObjectId)
df['firstBlip'] = df.apply(lambda x: Point(x['firstBlip_lng'], x['firstBlip_lat']),axis=1)
df['lastBlip'] = df.apply(lambda x: Point(x['lastBlip_lng'], x['lastBlip_lat']),axis=1)
df['firstBlip_polygon_id'] = df.apply(lambda x: [ObjectId(i) for i in x['firstBlip_polygon_id'].split(',')] if not pd.isna(x['firstBlip_polygon_id']) else None, axis=1)

print('iii')
