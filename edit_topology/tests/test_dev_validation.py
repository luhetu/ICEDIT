import unittest

from edit_topology.dev_validation import DEV_CASES, run_dev_cases


class DevelopmentValidationTests(unittest.TestCase):
    def test_all_development_cases_pass(self):
        results = run_dev_cases()
        self.assertEqual(len(results), len(DEV_CASES))
        failures = [row for row in results if row["schema"] != "PASS" or row["semantic"] != "PASS"]
        self.assertEqual(failures, [])


if __name__ == "__main__":
    unittest.main()
