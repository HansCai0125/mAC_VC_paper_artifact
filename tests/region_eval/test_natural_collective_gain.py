import unittest

from region_eval.natural_collective_gain import analyze_document


class NaturalCollectiveGainTests(unittest.TestCase):
    def test_two_complementary_halves_create_collective_gain(self):
        document = {
            "images": [{
                "image_id": "image",
                "width": 20,
                "height": 20,
                "gts": [{"id": "gt", "class_id": 0, "xyxy": [0, 0, 10, 10]}],
                "predictions": [
                    {"id": "left", "class_id": 0, "score": .9,
                     "xyxy": [0, 0, 5, 10]},
                    {"id": "right", "class_id": 0, "score": .8,
                     "xyxy": [5, 0, 10, 10]},
                ],
            }]
        }
        row = analyze_document(document)[0]
        self.assertAlmostEqual(row["AC"], 1.0)
        self.assertAlmostEqual(row["BestIoGT"], 0.5)
        self.assertAlmostEqual(row["CollectiveGain"], 0.5)
        self.assertAlmostEqual(row["ABO"], 0.5)
        self.assertEqual(row["marginal_contributor_count"], 2)

    def test_oversized_box_has_no_collective_gain(self):
        document = {
            "images": [{
                "image_id": "image",
                "width": 20,
                "height": 20,
                "gts": [{"id": "gt", "class_id": 0, "xyxy": [0, 0, 10, 10]}],
                "predictions": [{
                    "id": "large", "class_id": 0, "score": .9,
                    "xyxy": [0, 0, 20, 20],
                }],
            }]
        }
        row = analyze_document(document)[0]
        self.assertAlmostEqual(row["AC"], 1.0)
        self.assertAlmostEqual(row["BestIoGT"], 1.0)
        self.assertAlmostEqual(row["CollectiveGain"], 0.0)
        self.assertAlmostEqual(row["ABO"], 0.25)
        self.assertAlmostEqual(row["BestPredIoA"], 0.25)
        self.assertEqual(row["marginal_contributor_count"], 1)


if __name__ == "__main__":
    unittest.main()
