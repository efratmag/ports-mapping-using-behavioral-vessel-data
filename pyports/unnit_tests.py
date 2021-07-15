import unittest

from pyports.geo_utils import haversine


class TestHaversine(unittest.TestCase):

    def test_myfuncc(self):
        use_case_input = ((32.075851, 34.752094), (32.058722, 34.741194))
        distance = haversine(*use_case_input)
        self.assertEqual(round(distance, 2), 2.17, "haversine was not calculated - wrong distance")
