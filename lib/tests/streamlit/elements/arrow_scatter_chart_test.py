# Copyright (c) Streamlit Inc. (2018-2022) Snowflake Inc. (2022)
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import json

import pandas as pd

import streamlit as st
from streamlit.type_util import bytes_to_data_frame
from tests.delta_generator_test_case import DeltaGeneratorTestCase


class ArrowScatterChartsTest(DeltaGeneratorTestCase):
    """Test Arrow scatter chart."""

    def test_arrow_scatter_chart_with_size_column(self):
        """Test st._arrow_scatter_chart with size set to a column."""
        df = pd.DataFrame([[20, 30, 50, 60]], columns=["a", "b", "c", "d"])
        EXPECTED_DATAFRAME = pd.DataFrame(
            [[0, 60, 20, 30, 50]],
            columns=["index--p5bJXXpQgvPz6yvQMFiy", "d", "a", "b", "c"],
        )

        st._arrow_scatter_chart(df, size="d")

        proto = self.get_delta_from_queue().new_element.arrow_vega_lite_chart
        chart_spec = json.loads(proto.spec)

        self.assertIn(chart_spec["mark"], ["circle", {"type": "circle"}])

        self.assertEqual(chart_spec["encoding"]["size"]["field"], "d")

        pd.testing.assert_frame_equal(
            bytes_to_data_frame(proto.datasets[0].data.data),
            EXPECTED_DATAFRAME,
        )

    def test_arrow_scatter_chart_with_size_value(self):
        """Test st._arrow_scatter_chart with size set to a value."""
        df = pd.DataFrame([[20, 30, 50, 60]], columns=["a", "b", "c", "d"])
        EXPECTED_DATAFRAME = pd.DataFrame(
            [[0, 20, 30, 50, 60]],
            columns=["index--p5bJXXpQgvPz6yvQMFiy", "a", "b", "c", "d"],
        )

        st._arrow_scatter_chart(df, size=42)

        proto = self.get_delta_from_queue().new_element.arrow_vega_lite_chart
        chart_spec = json.loads(proto.spec)

        self.assertIn(chart_spec["mark"], ["circle", {"type": "circle"}])

        self.assertEqual(chart_spec["encoding"]["size"]["value"], 42)

        pd.testing.assert_frame_equal(
            bytes_to_data_frame(proto.datasets[0].data.data),
            EXPECTED_DATAFRAME,
        )
