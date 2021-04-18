import pandas as pd
import pymongo
import boto3
import itertools
import sys
from bson import ObjectId
from shapely.geometry import Point
from sshtunnel import SSHTunnelForwarder
import pymongo
import base64
import io
import paramiko


#
# with open('/Users/aloncohen/Downloads/march8.pem', 'rb') as f:
#     blob = base64.b64encode(f.read())
#
# SSH_KEY_BLOB = blob.decode('utf-8')
#
# SSH_KEY_BLOB_DECODED = base64.b64decode(SSH_KEY_BLOB)
# SSH_KEY = SSH_KEY_BLOB_DECODED.decode('utf-8')
#
#
# # pass key to parmiko to get your pkey
# pkey = paramiko.RSAKey.from_private_key(io.StringIO(SSH_KEY))
#
# server = SSHTunnelForwarder(
#     'ec2-18-156-198-181.eu-central-1.compute.amazonaws.com',
#     ssh_username='ec2-user',
#     ssh_pkey=pkey,
#     remote_bind_address=('18.156.198.181', 27017)
# )
#
# server = SSHTunnelForwarder(
#     'ec2-18-156-198-181.eu-central-1.compute.amazonaws.com',
#     ssh_username='ec2-user',
#     ssh_pkey='/Users/aloncohen/Downloads/march8.pem',
#     remote_bind_address=('127.0.0.1', 27017)
# )

server.start()
client = pymongo.MongoClient('mongodb://alon:winward1234@getstarted-ec2.cluster-cqssxwivnzqi.eu-central-1.docdb.amazonaws.com:27017/?ssl=true&ssl_ca_certs=rds-combined-ca-bundle.pem&replicaSet=rs0&readPreference=secondaryPreferred&retryWrites=false')
# client = pymongo.MongoClient('mongodb://alon:winward1234@getstarted-ec2.cluster-cqssxwivnzqi.eu-central-1.docdb.amazonaws.com:27017/?ssl=true&ssl_ca_certs=/Users/aloncohen/Downloads/rds-combined-ca-bundle.pem&replicaSet=rs0&readPreference=secondaryPreferred&retryWrites=false', server.local_bind_port)
db = client.activities
col = db.drifting

s3 = boto3.resource(
    service_name='s3',
    region_name='us-east-2')
obj = s3.Bucket('windward-project').Object('drifting.csv').get()

df = pd.read_csv(obj['Body'])

df.drop(['enrichers', 'firstBlip', 'lastBlip'], axis=1, inplace=True)

df['createdDate'] = pd.to_datetime(df['createdDate'])
df['startDate'] = pd.to_datetime(df['startDate'])
df['endDate'] = pd.to_datetime(df['endDate'])

df['_id'] = df['_id'].map(ObjectId)
df['vesselId'] = df['vesselId'].map(ObjectId)
df['firstBlip'] = df.apply(lambda x: {'lng': x['firstBlip_lng'], 'lat': x['firstBlip_lat']}, axis=1)
df['lastBlip'] = df.apply(lambda x: {'lng': x['lastBlip_lng'], 'lat': x['lastBlip_lat']} if not pd.isna(x['lastBlip_lng']) else None, axis=1)
df['firstBlip_polygon_id'] = df.apply(lambda x: [ObjectId(i.strip()) for i in x['firstBlip_polygon_id'].split(',')] if not pd.isna(x['firstBlip_polygon_id']) else None, axis=1)
df['lastBlip_polygon_id'] = df.apply(lambda x: [ObjectId(i.strip()) for i in x['lastBlip_polygon_id'].split(',')] if not pd.isna(x['lastBlip_polygon_id']) else None, axis=1)

df.drop(['firstBlip_lng', 'firstBlip_lat', 'lastBlip_lng', 'lastBlip_lat'], axis=1, inplace=True)

df['endDate'] = df['endDate'].astype(object).where(df.endDate.notnull(), None)

items = df.to_dict('records')


def chunker(iterable, n):

    it = iter(iterable)
    while True:
        chunk = tuple(itertools.islice(it, n))
        if not chunk:
            return
        yield chunk


for chunk in chunker(items, 5000):

    col.insert_many(chunk)


print('done!')