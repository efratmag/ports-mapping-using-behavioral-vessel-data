from pyports.generate_activity_data import *
import unittest


class TestExtractCoordinates(unittest.TestCase):

    def test_extract_coordinates(self):

        anchoring_path = os.getcwd().replace('/tests', '/data_samples_for_tests/anchoring.csv')

        df = pd.read_csv(anchoring_path)

        coordinates_df = extract_coordinates(df, 'firstBlip')

        self.assertEqual(coordinates_df['firstBlip_lat'].apply(type).unique()[0], float, "incorrect firstBlip lat type")
        self.assertEqual(coordinates_df['firstBlip_lon'].apply(type).unique()[0], float, "incorrect firstBlip lon type")

        self.assertTrue(-90 <= coordinates_df['firstBlip_lat'].min() <= 90, "invalid firstBlip lat (out of range -90 -> 90)")
        self.assertTrue(-180 <= coordinates_df['firstBlip_lon'].min() <= 180, "invalid firstBlip lat (out of range -180 -> 180)")
