import unittest
from shapely.geometry import Polygon, MultiPolygon
import geopandas as gpd
import numpy as np

from pyports.geo_utils import haversine, calc_polygon_area_sq_unit, inflate_polygon, calc_cluster_density, \
    merge_polygons, calc_polygon_distance_from_nearest_ww_polygon, polygon_intersection, get_multipolygon_exterior, \
    polygon_to_wgs84, polygon_to_meters


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

TEST_COORDINATES_C = ((-113.675537109375, 31.052933985705163),
                      (-114.69177246093749, 30.951702416409052),
                      (-114.32647705078125, 30.327842001072675),
                      (-113.675537109375, 31.052933985705163))

TEST_COORDINATES_METERS = ((3263387.947548054, 3572884.4743992453),
                           (3271822.562175513, 3572884.4743992453),
                           (3271822.562175513, 3582093.846986187),
                           (3263387.947548054, 3582093.846986187),
                           (3263387.947548054, 3572884.4743992453))

TEST_POLYGON_A = Polygon(TEST_COORDINATES_A)
TEST_POLYGON_B = Polygon(TEST_COORDINATES_B)
TEST_POLYGON_C = Polygon(TEST_COORDINATES_C)  # polygons A & C intersects
TEST_POLYGON_METERS = Polygon(TEST_COORDINATES_METERS)  # polygons A & WGS84 are identical (on different CRS)

TEST_GEO_DF = gpd.GeoDataFrame([{"geometry": TEST_POLYGON_A, "polygon_area_type": "PortWaitingArea"},
                               {"geometry": TEST_POLYGON_B, "polygon_area_type": "Port"}])


class TestHaversine(unittest.TestCase):

    def test_distance(self):
        use_case_input = ((32.075851, 34.752094), (32.058722, 34.741194))
        distance = haversine(*use_case_input)
        self.assertAlmostEqual(distance, 2.17, 2, "haversine was not calculated properly - wrong distance")


class TestCalcPolygonArea(unittest.TestCase):

    def test_area(self):

        sqkm_area = calc_polygon_area_sq_unit(TEST_POLYGON_A, 'sqkm')
        sqmi_area = calc_polygon_area_sq_unit(TEST_POLYGON_A, 'sqmi')
        self.assertAlmostEqual(sqkm_area, 77.68, 2, "area (sqkm) was not calculated properly - wrong area")
        self.assertAlmostEqual(sqmi_area, 29.99, 2, "area (sqmi) was not calculated properly - wrong area")


class TestInflatePolygon(unittest.TestCase):

    def test_inflation_area(self):

        inflated_polygon = inflate_polygon(TEST_POLYGON_A, 1000)
        inflated_polygon_area = calc_polygon_area_sq_unit(inflated_polygon)
        self.assertAlmostEqual(inflated_polygon_area, 116.02, 2,
                               "area (sqkm) of the inflated polygon indicates an issue with inflation logic - wrong area")

    def test_polygon_centroid(self):

        inflated_polygon = inflate_polygon(TEST_POLYGON_A, 1000)
        pre_inflation_centroid = TEST_POLYGON_A.centroid
        post_inflation_centroid = inflated_polygon.centroid

        self.assertAlmostEqual(pre_inflation_centroid.y, post_inflation_centroid.y, 5,
                               "polygon inflation was not performed properly - wrong centroid latitude")

        self.assertAlmostEqual(pre_inflation_centroid.x, post_inflation_centroid.x, 5,
                               "polygon inflation was not performed properly - wrong centroid longitude")


class TestDensity(unittest.TestCase):

    def test_density_calc(self):
        density = calc_cluster_density(TEST_COORDINATES_A)

        self.assertAlmostEqual(density, 84.66, 2, "density was not calculated properly - density values")


class TestMergePolygons(unittest.TestCase):

    def test_merged_polygons_counts(self):
        merged_df = merge_polygons(TEST_GEO_DF)
        self.assertCountEqual(merged_df, [TEST_POLYGON_A, TEST_POLYGON_B],
                              "polygons merging was not performed properly - wrong polygons count")

    def test_merged_polygon_area(self):
        merged_polygons_area = merge_polygons(TEST_GEO_DF).area
        polygons_area = TEST_POLYGON_A.area + TEST_POLYGON_B.area
        self.assertEqual(merged_polygons_area, polygons_area, "polygons merging was not performed properly - areas are not equal")


class TestNearestWwPolygon(unittest.TestCase):

    def test_distance_to_ww_polygons(self):

        ww_polygons_centroids = np.array(([34.83489990234375, 32.676372772089834],
                                          [30.130004882812496, 31.423975737976697]))

        nearest_polygon_dist = calc_polygon_distance_from_nearest_ww_polygon(TEST_POLYGON_A, ww_polygons_centroids)
        self.assertAlmostEqual(nearest_polygon_dist, 351.51, 2,
                               "nearest polygon was not calculated properly - wrong nearest polygon distance")


class TestPolygonIntersection(unittest.TestCase):

    def test_polygons_intersection(self):

        waiting_area_intersection = polygon_intersection(TEST_POLYGON_C, TEST_GEO_DF, "pwa")
        ports_intersection = polygon_intersection(TEST_POLYGON_C, TEST_GEO_DF, "ports")

        self.assertEqual(waiting_area_intersection, 0,
                         'waiting areas intersection was not calculated properly - wrong % of intersection')

        self.assertAlmostEqual(ports_intersection, 7.379, 2,
                               'ports areas intersection was not calculated properly - wrong % of intersection')


class TestGetMultipolygonExterior(unittest.TestCase):

    def test_multipolygon_exterior(self):

        multipolygon = MultiPolygon([TEST_POLYGON_A, TEST_POLYGON_B])
        multipolygon_exterior = set(get_multipolygon_exterior(multipolygon))
        polygons_exterior = set(TEST_POLYGON_A.exterior.coords).union(set(TEST_POLYGON_B.exterior.coords))

        self.assertSetEqual(multipolygon_exterior, polygons_exterior,
                            'multipolygon_exterior was not calculated properly - wrong coordinates')


class TestPolygonToWgs84(unittest.TestCase):

    def test_polygon_to_wgs84(self):
        _, wgs84_polygon = polygon_to_wgs84(TEST_POLYGON_METERS)
        self.assertNotEqual(wgs84_polygon, TEST_POLYGON_METERS,
                            "polygon_to_wgs84 was not performed properly - the polygon was not changed")
        self.assertAlmostEqual(wgs84_polygon.centroid.y, 32.173132971, 6,
                               "polygon_to_wgs84 was not performed properly - wrong centroid latitude")
        self.assertAlmostEqual(wgs84_polygon.centroid.x, 34.567696060, 6,
                               "polygon_to_wgs84 was not performed properly - wrong centroid latitude")

