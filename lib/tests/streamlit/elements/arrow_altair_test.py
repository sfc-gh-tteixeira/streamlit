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
from datetime import date
from functools import reduce
from typing import Callable

import altair as alt
import pandas as pd
import pytest
from parameterized import parameterized

import streamlit as st
from streamlit.elements import arrow_altair as altair
from streamlit.elements.arrow_altair import ChartType
from streamlit.errors import StreamlitAPIException
from streamlit.type_util import bytes_to_data_frame
from tests.delta_generator_test_case import DeltaGeneratorTestCase


def _deep_get(dictionary, *keys):
    return reduce(
        lambda d, key: d.get(key, None) if isinstance(d, dict) else None,
        keys,
        dictionary,
    )


class ArrowAltairTest(DeltaGeneratorTestCase):
    """Test ability to marshall arrow_altair_chart proto."""

    def test_altair_chart(self):
        """Test that it can be called with args."""
        df = pd.DataFrame([["A", "B", "C", "D"], [28, 55, 43, 91]], index=["a", "b"]).T
        chart = alt.Chart(df).mark_bar().encode(x="a", y="b")
        EXPECTED_DATAFRAME = pd.DataFrame(
            {
                "a": ["A", "B", "C", "D"],
                "b": [28, 55, 43, 91],
            }
        )

        st._arrow_altair_chart(chart)

        proto = self.get_delta_from_queue().new_element.arrow_vega_lite_chart

        self.assertEqual(proto.HasField("data"), False)
        self.assertEqual(len(proto.datasets), 1)
        pd.testing.assert_frame_equal(
            bytes_to_data_frame(proto.datasets[0].data.data), EXPECTED_DATAFRAME
        )

        spec_dict = json.loads(proto.spec)
        self.assertEqual(
            spec_dict["encoding"],
            {
                "y": {"field": "b", "type": "quantitative"},
                "x": {"field": "a", "type": "nominal"},
            },
        )
        self.assertEqual(spec_dict["data"], {"name": proto.datasets[0].name})
        self.assertIn(spec_dict["mark"], ["bar", {"type": "bar"}])
        self.assertTrue("encoding" in spec_dict)

    def test_date_column_utc_scale(self):
        """Test that columns with date values have UTC time scale"""
        df = pd.DataFrame(
            {"index": [date(2019, 8, 9), date(2019, 8, 10)], "numbers": [1, 10]}
        ).set_index("index")

        chart, _ = altair._generate_chart(ChartType.LINE, df)
        st._arrow_altair_chart(chart)
        proto = self.get_delta_from_queue().new_element.arrow_vega_lite_chart
        spec_dict = json.loads(proto.spec)

        # The x axis should have scale="utc", because it uses date values.
        x_scale = _deep_get(spec_dict, "encoding", "x", "scale", "type")
        self.assertEqual(x_scale, "utc")

        # The y axis should _not_ have scale="utc", because it doesn't
        # use date values.
        y_scale = _deep_get(spec_dict, "encoding", "y", "scale", "type")
        self.assertNotEqual(y_scale, "utc")

    @parameterized.expand(
        [
            ("streamlit", "streamlit"),
            (None, ""),
        ]
    )
    def test_theme(self, theme_value, proto_value):
        df = pd.DataFrame(
            {"index": [date(2019, 8, 9), date(2019, 8, 10)], "numbers": [1, 10]}
        ).set_index("index")

        chart, _ = altair._generate_chart(ChartType.LINE, df)
        st._arrow_altair_chart(chart, theme=theme_value)

        el = self.get_delta_from_queue().new_element
        self.assertEqual(el.arrow_vega_lite_chart.theme, proto_value)

    def test_bad_theme(self):
        df = pd.DataFrame(
            {"index": [date(2019, 8, 9), date(2019, 8, 10)], "numbers": [1, 10]}
        ).set_index("index")

        chart, _ = altair._generate_chart(ChartType.LINE, df)
        with self.assertRaises(StreamlitAPIException) as exc:
            st._arrow_altair_chart(chart, theme="bad_theme")

        self.assertEqual(
            f'You set theme="bad_theme" while Streamlit charts only support theme=”streamlit” or theme=None to fallback to the default library theme.',
            str(exc.exception),
        )


class ArrowChartsTest(DeltaGeneratorTestCase):
    """Test Arrow charts."""

    def test_arrow_line_chart(self):
        """Test st._arrow_line_chart."""
        df = pd.DataFrame([[20, 30, 50]], columns=["a", "b", "c"])
        EXPECTED_DATAFRAME = pd.DataFrame(
            [[20, 30, 50, 0]], columns=["a", "b", "c", "index-4FLV4aXfCWIrl1KyIeJp"]
        )

        st._arrow_line_chart(df, width=640, height=480)

        proto = self.get_delta_from_queue().new_element.arrow_vega_lite_chart
        chart_spec = json.loads(proto.spec)
        self.assertIn(chart_spec["mark"], ["line", {"type": "line"}])
        self.assertEqual(chart_spec["width"], 640)
        self.assertEqual(chart_spec["height"], 480)
        pd.testing.assert_frame_equal(
            bytes_to_data_frame(proto.datasets[0].data.data),
            EXPECTED_DATAFRAME,
        )

    @parameterized.expand(
        [
            (st._arrow_area_chart, "area"),
            (st._arrow_bar_chart, "bar"),
            (st._arrow_line_chart, "line"),
            (st._arrow_scatter_chart, "circle"),
        ]
    )
    def test_arrow_chart_with_x_y(self, chart_command: Callable, altair_type: str):
        """Test x/y-support for built-in charts."""
        df = pd.DataFrame([[20, 30, 50]], columns=["a", "b", "c"])
        EXPECTED_DATAFRAME = pd.DataFrame([[20, 30]], columns=["a", "b"])

        chart_command(df, x="a", y="b", width=640, height=480)

        proto = self.get_delta_from_queue().new_element.arrow_vega_lite_chart
        chart_spec = json.loads(proto.spec)

        self.assertIn(chart_spec["mark"], [altair_type, {"type": altair_type}])
        self.assertEqual(chart_spec["width"], 640)
        self.assertEqual(chart_spec["height"], 480)
        self.assertEqual(chart_spec["encoding"]["x"]["field"], "a")
        self.assertEqual(chart_spec["encoding"]["y"]["field"], "b")
        pd.testing.assert_frame_equal(
            bytes_to_data_frame(proto.datasets[0].data.data),
            EXPECTED_DATAFRAME,
        )

    @parameterized.expand(
        [
            (st._arrow_area_chart, "area"),
            (st._arrow_bar_chart, "bar"),
            (st._arrow_line_chart, "line"),
            (st._arrow_scatter_chart, "circle"),
        ]
    )
    def test_arrow_chart_with_x_y_sequence(
        self, chart_command: Callable, altair_type: str
    ):
        """Test x/y-sequence support for built-in charts."""
        df = pd.DataFrame([[20, 30, 50]], columns=["a", "b", "c"])
        EXPECTED_DATAFRAME = pd.DataFrame([[20, 30, 50]], columns=["a", "b", "c"])

        chart_command(df, x="a", y=["b", "c"])

        proto = self.get_delta_from_queue().new_element.arrow_vega_lite_chart
        chart_spec = json.loads(proto.spec)

        self.assertIn(chart_spec["mark"], [altair_type, {"type": altair_type}])
        self.assertEqual(chart_spec["encoding"]["x"]["field"], "a")
        self.assertEqual(
            chart_spec["encoding"]["y"]["field"], "values-7hbjwi6ywufr4T3VmvRh"
        )

        pd.testing.assert_frame_equal(
            bytes_to_data_frame(proto.datasets[0].data.data),
            EXPECTED_DATAFRAME,
        )

    @parameterized.expand(
        [
            (st._arrow_area_chart, "area"),
            (st._arrow_bar_chart, "bar"),
            (st._arrow_line_chart, "line"),
            (st._arrow_scatter_chart, "circle"),
        ]
    )
    def test_arrow_chart_with_color_column(
        self, chart_command: Callable, altair_type: str
    ):
        """Test color support for built-in charts."""
        df = pd.DataFrame(
            {
                "x": [0, 1, 2],
                "y": [22, 21, 20],
                "tuple3_int_color": [[255, 0, 0], [0, 255, 0], [0, 0, 255]],
                "tuple4_int_int_color": [
                    [255, 0, 0, 51],
                    [0, 255, 0, 51],
                    [0, 0, 255, 51],
                ],
                "tuple4_int_float_color": [
                    [255, 0, 0, 0.2],
                    [0, 255, 0, 0.2],
                    [0, 0, 255, 0.2],
                ],
                "tuple3_float_color": [
                    [1.0, 0.0, 0.0],
                    [0.0, 1.0, 0.0],
                    [0.0, 0.0, 1.0],
                ],
                "tuple4_float_float_color": [
                    [1.0, 0.0, 0.0, 0.2],
                    [0.0, 1.0, 0.0, 0.2],
                    [0.0, 0.0, 1.0, 0.2],
                ],
                "hex3_color": ["#f00", "#0f0", "#00f"],
                "hex4_color": ["#f008", "#0f08", "#00f8"],
                "hex6_color": ["#ff0000", "#00ff00", "#0000ff"],
                "hex8_color": ["#ff000088", "#00ff0088", "#0000ff88"],
            }
        )

        color_columns = sorted(set(df.columns))
        color_columns.remove("x")
        color_columns.remove("y")

        expected_values = pd.DataFrame(
            {
                "tuple3": ["rgb(255, 0, 0)", "rgb(0, 255, 0)", "rgb(0, 0, 255)"],
                "tuple4": [
                    "rgba(255, 0, 0, 0.2)",
                    "rgba(0, 255, 0, 0.2)",
                    "rgba(0, 0, 255, 0.2)",
                ],
                "hex3": ["#f00", "#0f0", "#00f"],
                "hex6": ["#ff0000", "#00ff00", "#0000ff"],
                "hex4": ["#f008", "#0f08", "#00f8"],
                "hex8": ["#ff000088", "#00ff0088", "#0000ff88"],
            }
        )

        def get_expected_color_values(col_name):
            for prefix, expected_color_values in expected_values.items():
                if col_name.startswith(prefix):
                    return expected_color_values

        for color_column in color_columns:
            expected_color_values = get_expected_color_values(color_column)

            chart_command(df, x="x", y="y", color=color_column)

            proto = self.get_delta_from_queue().new_element.arrow_vega_lite_chart
            chart_spec = json.loads(proto.spec)

            self.assertEqual(chart_spec["encoding"]["color"]["field"], color_column)

            # Manually specified colors should not have a special legend
            self.assertEqual(chart_spec["encoding"]["color"]["legend"], None)

            # Manually specified colors are set via the color scale's range property.
            self.assertTrue(chart_spec["encoding"]["color"]["scale"]["range"])

            proto_df = bytes_to_data_frame(proto.datasets[0].data.data)

            pd.testing.assert_series_equal(
                proto_df[color_column],
                expected_color_values,
                check_names=False,
            )

    @parameterized.expand(
        [
            (st._arrow_area_chart, "a", "foooo"),
            (st._arrow_bar_chart, "not-valid", "b"),
            (st._arrow_line_chart, "foo", "bar"),
            (st._arrow_line_chart, None, "bar"),
            (st._arrow_line_chart, "foo", None),
            (st._arrow_line_chart, "a", ["b", "foo"]),
            (st._arrow_line_chart, None, "variable"),
            (st._arrow_line_chart, "variable", ["a", "b"]),
        ]
    )
    def test_arrow_chart_with_x_y_invalid_input(
        self,
        chart_command: Callable,
        x: str,
        y: str,
    ):
        """Test x/y support for built-in charts with invalid input."""
        df = pd.DataFrame([[20, 30, 50]], columns=["a", "b", "c"])

        with pytest.raises(StreamlitAPIException):
            chart_command(df, x=x, y=y)

    def test_arrow_chart_with_x_y_on_sliced_data(
        self,
    ):
        """Test x/y-support for built-in charts on sliced data."""
        df = pd.DataFrame([[20, 30, 50], [60, 70, 80]], columns=["a", "b", "c"])
        EXPECTED_DATAFRAME = pd.DataFrame([[20, 30], [60, 70]], columns=["a", "b"])[1:]

        # Use all data after first item
        st.line_chart(df[1:], x="a", y="b")

        proto = self.get_delta_from_queue().new_element.arrow_vega_lite_chart
        chart_spec = json.loads(proto.spec)

        self.assertEqual(chart_spec["encoding"]["x"]["field"], "a")
        self.assertEqual(chart_spec["encoding"]["y"]["field"], "b")

        pd.testing.assert_frame_equal(
            bytes_to_data_frame(proto.datasets[0].data.data),
            EXPECTED_DATAFRAME,
        )

    def test_arrow_line_chart_with_generic_index(self):
        """Test st._arrow_line_chart with a generic index."""
        df = pd.DataFrame([[20, 30, 50]], columns=["a", "b", "c"])
        df.set_index("a", inplace=True)

        EXPECTED_DATAFRAME = pd.DataFrame(
            [[30, 50, 20]],
            columns=["b", "c", "index-4FLV4aXfCWIrl1KyIeJp"],
            index=pd.RangeIndex(0, 1, 1),
        )

        st._arrow_line_chart(df)

        proto = self.get_delta_from_queue().new_element.arrow_vega_lite_chart
        chart_spec = json.loads(proto.spec)
        self.assertIn(chart_spec["mark"], ["line", {"type": "line"}])
        pd.testing.assert_frame_equal(
            bytes_to_data_frame(proto.datasets[0].data.data),
            EXPECTED_DATAFRAME,
        )

    def test_arrow_area_chart(self):
        """Test st._arrow_area_chart."""
        df = pd.DataFrame([[20, 30, 50]], columns=["a", "b", "c"])
        EXPECTED_DATAFRAME = pd.DataFrame(
            [[20, 30, 50, 0]],
            columns=["a", "b", "c", "index-4FLV4aXfCWIrl1KyIeJp"],
        )

        st._arrow_area_chart(df)

        proto = self.get_delta_from_queue().new_element.arrow_vega_lite_chart
        chart_spec = json.loads(proto.spec)
        self.assertIn(chart_spec["mark"], ["area", {"type": "area"}])
        pd.testing.assert_frame_equal(
            bytes_to_data_frame(proto.datasets[0].data.data),
            EXPECTED_DATAFRAME,
        )

    def test_arrow_bar_chart(self):
        """Test st._arrow_bar_chart."""
        df = pd.DataFrame([[20, 30, 50]], columns=["a", "b", "c"])
        EXPECTED_DATAFRAME = pd.DataFrame(
            [[20, 30, 50, 0]],
            columns=["a", "b", "c", "index-4FLV4aXfCWIrl1KyIeJp"],
        )

        st._arrow_bar_chart(df, width=640, height=480)

        proto = self.get_delta_from_queue().new_element.arrow_vega_lite_chart
        chart_spec = json.loads(proto.spec)

        self.assertIn(chart_spec["mark"], ["bar", {"type": "bar"}])
        self.assertEqual(chart_spec["width"], 640)
        self.assertEqual(chart_spec["height"], 480)
        pd.testing.assert_frame_equal(
            bytes_to_data_frame(proto.datasets[0].data.data),
            EXPECTED_DATAFRAME,
        )

    def test_unused_columns_are_dropped(self):
        pass
