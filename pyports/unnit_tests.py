import unittest
from shapely.geometry import Polygon
import geopandas as gpd

from pyports.geo_utils import haversine, calc_polygon_area_sq_unit, inflate_polygon, calc_cluster_density, merge_polygons


TEST_COORDINATES_A = ((34.66941833496094, 32.13172203278829),
                      (34.75902557373047, 32.13172203278829),
                      (34.75902557373047, 32.214543910232216),
                      (34.66941833496094, 32.214543910232216),
                      (34.66941833496094, 32.13172203278829))

TEST_COORDINATES_B = ((-114.37591552734374, 31.438037173124464),
                      (-114.79339599609375, 31.236288641793006),
                      (-114.202880859375, 30.85743710875022),
                      (-113.8787841796875, 31.19165800392904),
                      (-114.37591552734374, 31.438037173124464))

TEST_POLYGON_A = Polygon(TEST_COORDINATES_A)
TEST_POLYGON_B = Polygon(TEST_COORDINATES_B)

TEST_GEO_DF = gpd.GeoDataFrame([TEST_POLYGON_A, TEST_POLYGON_B], columns=['geometry'])


class TestHaversine(unittest.TestCase):

    def test_distance(self):
        use_case_input = ((32.075851, 34.752094), (32.058722, 34.741194))
        distance = haversine(*use_case_input)
        self.assertEqual(round(distance, 2), 2.17, "haversine was not calculated properly - wrong distance")


class TestCalcPolygonArea(unittest.TestCase):

    def test_area(self):

        sqkm_area = calc_polygon_area_sq_unit(TEST_POLYGON_A, 'sqkm')
        sqmi_area = calc_polygon_area_sq_unit(TEST_POLYGON_A, 'sqmi')
        self.assertEqual(round(sqkm_area, 2), 77.68, "area (sqkm) was not calculated properly - wrong area")
        self.assertEqual(round(sqmi_area, 2), 29.99, "area (sqmi) was not calculated properly - wrong area")


class TestInflatePolygon(unittest.TestCase):

    def test_inflation_area(self):

        inflated_polygon = inflate_polygon(TEST_POLYGON_A, 1000)
        inflated_polygon_area = calc_polygon_area_sq_unit(inflated_polygon)
        self.assertEqual(round(inflated_polygon_area, 2), 116.02,
                         "area (sqkm) of the inflated polygon indicates an issue with inflation logic - wrong area")

    def test_polygon_centroid(self):

        inflated_polygon = inflate_polygon(TEST_POLYGON_A, 1000)
        pre_inflation_centroid = TEST_POLYGON_A.centroid
        post_inflation_centroid = inflated_polygon.centroid

        self.assertEqual(round(pre_inflation_centroid.y, 4), round(post_inflation_centroid.y, 4),
                         "polygon inflation was not performed properly - wrong centroid latitude")

        self.assertEqual(round(pre_inflation_centroid.x, 4), round(post_inflation_centroid.x, 4),
                         "polygon inflation was not performed properly - wrong centroid longitude")


class TestDensity(unittest.TestCase):

    def test_density(self):
        density = calc_cluster_density(TEST_COORDINATES_A)

        self.assertEqual(round(density, 2), 84.66, "density was not calculated properly - density values")


class TestMergePolygons(unittest.TestCase):

    def test_objects_merging(self):
        merged_df = merge_polygons(TEST_GEO_DF)
        self.assertEqual(len(list(merged_df)), 2, "polygons merging was not performed properly - wrong polygons count")

    def test_merged_polygon_area(self):
        merged_polygons_area = merge_polygons(TEST_GEO_DF).area
        polygons_area = TEST_POLYGON_A.area + TEST_POLYGON_B.area
        self.assertEqual(merged_polygons_area, polygons_area, "polygons merging was not performed properly - aras are not equal")