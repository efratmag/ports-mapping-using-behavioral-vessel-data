from geo_utils import get_bounding_box, isin_box
import os
import fire
import pandas as pd

ACTIVITIES_FILES = ['mooring.csv.gz', 'drifting.csv.gz', 'port_calls.csv.gz', 'anchoring.csv.gz']

CONSTANTS_FILES = ['vessels.csv.gz', 'polygons.json', 'ports.json']


def extract_coordinates(df, col='firstBlip'):

    df[[col+'_lng', col+'_lat']] = df[col].apply(eval).apply(lambda x: x['geometry']['coordinates']).apply(pd.Series)

    return df


def main(lat, lng, import_path, export_path, distance=100, debug=True):

    """

    :param lat:
    :param lng:
    :param import_path:
    :param export_path:
    :param distance:
    :param debug:
    :return:
    """

    results_list = []

    bounding_box = get_bounding_box(lat, lng, distance)

    files_list = os.listdir(import_path)

    nrows = 10000 if debug else None

    for file_name in files_list:

        file_path = os.path.join(import_path, file_name)

        if file_name in ACTIVITIES_FILES:
            print('loading file %s' % file_name)
            df = pd.read_csv(file_path, compression='gzip', nrows=nrows)
            df = extract_coordinates(df, 'firstBlip')
            df = df[['_id', 'vesselId', 'firstBlip_lat', 'firstBlip_lng']]

            df = df[df.apply(lambda x: isin_box(x['firstBlip_lat'], x['firstBlip_lng'], bounding_box), axis=1)]


if __name__ == "__main__":
    fire.Fire(main)

