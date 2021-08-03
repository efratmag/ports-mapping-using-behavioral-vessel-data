from pyports.get_metadata import *
import unittest
import json
import pandas as pd


class TestGetWWPolygons(unittest.TestCase):

    def test_get_ww_polygons(self):

        polygons_type = ['Dock', 'Dock', 'Dock', 'Port', 'Port', 'Port', 'Port',
                         'PortWaitingArea', 'PortWaitingArea', 'PortWaitingArea']

        polygons_file_path = os.getcwd().replace('/tests', '/data_samples_for_tests')

        with open(os.path.join(polygons_file_path, 'polygons.json'), 'r') as polygons_file:
            polygons_file = json.load(polygons_file)
        polygons_df = pd.DataFrame.from_dict(polygons_file, orient='index')

        get_ww_df = get_ww_polygons(polygons_file_path)

        self.assertEqual(get_ww_df["polygon_area_type"].to_list(), polygons_type, "incorrect polygons_type values")
        self.assertEqual(len(polygons_df), len(get_ww_df), "incorrect records counts")
        self.assertEqual(get_ww_df['polygon_id'].to_list(), polygons_df.index.to_list(), "incorrect vessels ids")


class TestGetVesselsInfo(unittest.TestCase):

    def test_get_vessels_info(self):

        vessels_type = ['other', 'cargo_other', 'other', 'other', 'other', 'other',
                        'cargo_other', 'cargo_other', 'other', 'cargo_other']

        vessels_file_path = os.getcwd().replace('/tests', '/data_samples_for_tests')

        with open(os.path.join(vessels_file_path, 'vessels.json'), 'r') as vessels_file:
            vessels_file = json.load(vessels_file)
        vessels_df = pd.DataFrame.from_dict(vessels_file, orient='index')

        get_vessels_df = get_vessels_info(vessels_file_path)

        self.assertEqual(get_vessels_df["vessel_class_new"].to_list(), vessels_type, "incorrect vessels_type values")
        self.assertEqual(len(vessels_df), len(get_vessels_df), "incorrect records counts")
        self.assertEqual(get_vessels_df['vesselId'].to_list(), vessels_df.index.to_list(), "incorrect vessels ids")


class TestGetPortsInfo(unittest.TestCase):

    def test_get_ports_info(self):

        ports_country = ['Japan', 'Norway and Svalbard', 'Canada', 'Peru', 'Algeria',
                        'United States', 'Madagascar', 'Bulgaria', 'Japan', 'United States']

        ports_file_path = os.getcwd().replace('/tests', '/data_samples_for_tests')

        with open(os.path.join(ports_file_path, 'ports.json'), 'r') as ports_file:
            ports_file = json.load(ports_file)
        ports_df = pd.DataFrame.from_dict(ports_file, orient='index')

        get_ports_df = get_ports_info(ports_file_path)

        self.assertEqual(get_ports_df["country"].to_list(), ports_country, "incorrect ports_country values")
        self.assertEqual(len(ports_df), len(get_ports_df), "incorrect records counts")
        self.assertEqual(get_ports_df['PortId'].to_list(), ports_df.index.to_list(), "incorrect vessels ids")


class TestGetShoreLineLayer(unittest.TestCase):

    def test_shoreline_layer(self):

        shoreline_file_path = os.getcwd().replace('/tests', '/data_samples_for_tests')

        shoreline_df = gpd.read_file(os.path.join(shoreline_file_path, 'shoreline_layer.geojson'))

        main_land, get_shoreline_df = get_shoreline_layer(shoreline_file_path)

        self.assertEqual(len(main_land), 4, "incorrect polygons counts of main_land")
        self.assertEqual(len(shoreline_df), len(list(get_shoreline_df)), "incorrect records counts")
        self.assertAlmostEqual(shoreline_df['geometry'].area.sum(), get_shoreline_df.area, 5, "incorrect area")

