"""Check that dashboard charts carry readable data to the browser."""

import unittest
from pathlib import Path

import pyarrow as pa
from streamlit.testing.v1 import AppTest


class DashboardChartPayloadTest(unittest.TestCase):
    def test_every_chart_has_decodable_nonempty_data(self):
        app_path = Path(__file__).resolve().parents[1] / "app.py"
        app = AppTest.from_file(str(app_path)).run(timeout=120)
        self.assertFalse(app.exception)

        charts = app.get("vega_lite_chart")
        self.assertGreaterEqual(len(charts), 45)
        for index, chart in enumerate(charts):
            with self.subTest(chart=index):
                self.assertTrue(chart.proto.datasets)
                for dataset in chart.proto.datasets:
                    table = pa.ipc.open_stream(
                        pa.BufferReader(dataset.data.data)
                    ).read_all()
                    self.assertGreater(table.num_rows, 0)


if __name__ == "__main__":
    unittest.main()
