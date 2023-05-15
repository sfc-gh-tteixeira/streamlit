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

"""A Python wrapper around Altair.
Altair is a Python visualization library based on Vega-Lite,
a nice JSON schema for expressing graphs and charts.
"""

from dataclasses import dataclass
from datetime import date
from enum import Enum
from typing import (
    TYPE_CHECKING,
    Any,
    Dict,
    Hashable,
    List,
    Optional,
    Sequence,
    Tuple,
    TypedDict,
    Union,
    cast,
)

import altair as alt
import pandas as pd
from altair.vegalite.v4.api import Chart
from pandas.api.types import infer_dtype, is_integer_dtype
from typing_extensions import Literal

import streamlit.elements.arrow_vega_lite as arrow_vega_lite
from streamlit import type_util
from streamlit.color_util import Color, is_color_str, to_css_color
from streamlit.elements.arrow import Data
from streamlit.elements.utils import last_index_for_melted_dataframes
from streamlit.errors import Error, StreamlitAPIException
from streamlit.proto.ArrowVegaLiteChart_pb2 import (
    ArrowVegaLiteChart as ArrowVegaLiteChartProto,
)
from streamlit.runtime.metrics_util import gather_metrics

if TYPE_CHECKING:
    from streamlit.delta_generator import DeltaGenerator

# Create and enable streamlit theme
STREAMLIT_THEME = {"embedOptions": {"theme": "streamlit"}}

# This allows to use alt.themes.enable("streamlit") to activate Streamlit theme.
alt.themes.register("streamlit", lambda: {"usermeta": STREAMLIT_THEME})

# no theme applied to charts
alt.themes.enable("none")


class ChartType(Enum):
    AREA = {"mark_type": "area"}
    BAR = {"mark_type": "bar"}
    LINE = {"mark_type": "line"}
    SCATTER = {"mark_type": "circle"}


LEGEND_SETTINGS = dict(titlePadding=0, offset=10, orient="bottom")

# Avoid collision with existing column names.
PROTECTION_SUFFIX = "-p5bJXXpQgvPz6yvQMFiy"
SEPARATED_INDEX_COLUMN_NAME = "index-4FLV4aXfCWIrl1KyIeJp"
SEPARATED_INDEX_COLUMN_TITLE = "index"
MELTED_Y_COLUMN_NAME = "values-7hbjwi6ywufr4T3VmvRh"
MELTED_Y_COLUMN_TITLE = "values"
MELTED_COLOR_COLUMN_NAME = "color-xWSR9VDwhyLw5IHyGvPX"
MELTED_COLOR_COLUMN_TITLE = "color"


class ArrowAltairMixin:
    @gather_metrics("_arrow_line_chart")
    def _arrow_line_chart(
        self,
        data: Data = None,
        *,
        x: Optional[str] = None,
        y: Union[str, Sequence[str], None] = None,
        color: Union[str, Color, List[Color], None] = None,
        width: int = 0,
        height: int = 0,
        use_container_width: bool = True,
    ) -> "DeltaGenerator":
        """Display a line chart.

        This is syntax-sugar around st._arrow_altair_chart. The main difference
        is this command uses the data's own column and indices to figure out
        the chart's spec. As a result this is easier to use for many "just plot
        this" scenarios, while being less customizable.

        If st._arrow_line_chart does not guess the data specification
        correctly, try specifying your desired chart using st._arrow_altair_chart.

        Parameters
        ----------
        data : pandas.DataFrame, pandas.Styler, pyarrow.Table, numpy.ndarray, Iterable, dict or None
            Data to be plotted.

        x : str or None
            Column name to use for the x-axis. If None, uses the data index for
            the x-axis. This argument can only be supplied by keyword.

        y : str, sequence of str, or None
            Column name(s) to use for the y-axis. If a sequence of strings,
            draws several series on the same chart by melting your wide-format
            table into a long-format table behind the scenes. If None, draws
            the data of all remaining columns as data series. This argument
            can only be supplied by keyword.

        color : str, sequence of str, tuple, sequence of tuple, or None
            The color to use for different lines in this chart. Can be:

            **If the chart will only have 1 line:**

            - A hex string like "#ffaa00" or "#ffaa0088".
            - A Matplotlib-compatible color name like "blue". See full list
              at https://matplotlib.org/stable/gallery/color/named_colors.html#css-colors.
            - A color tuple like (255, 255, 128) or (255, 255, 128, 0.5), where
              the RGB components are ints in the interval [0, 255] and the alpha
              component is a float in the interval [0.0, 1.0]. If they aren't
              the right data types or in the right interval, this function tries
              to guess the right thing to do.

            **If the chart will have multiple lines and the dataframe is in
            long format:**

            - The name of the column to use for the color. If the values in
              this column look like real colors, those will be used. If they
              do not, then a different column will be automatically selected
              to represent each value.

            **If the chart will have multiple lines and the dataframe is in
            wide format:**

            - A list of string colors or color tuples to be used for each of
              the lines in the chart. For example, if the chart will have 3
              lines can you can set ``color=["gold", "pink", "blue"]``.

        width : int
            The chart width in pixels. If 0, selects the width automatically.
            This argument can only be supplied by keyword.

        height : int
            The chart height in pixels. If 0, selects the height automatically.
            This argument can only be supplied by keyword.

        use_container_width : bool
            If True, set the chart width to the column width. This takes
            precedence over the width argument.
            This argument can only be supplied by keyword.

        Examples
        --------
        >>> import streamlit as st
        >>> import pandas as pd
        >>> import numpy as np
        >>>
        >>> chart_data = pd.DataFrame(
        ...     np.random.randn(20, 3),
        ...     columns=['a', 'b', 'c'])
        ...
        >>> st._arrow_line_chart(chart_data)

        .. output::
           https://static.streamlit.io/0.50.0-td2L/index.html?id=BdxXG3MmrVBfJyqS2R2ki8
           height: 220px

        You can also choose different columns to use for x and y, as well as set
        the color dynamically based on a 3rd column (assuming your dataframe is in
        long format):

        >>> chart_data = pd.DataFrame(
        ...     np.random.randn(20, 4),
        ...     columns=['col1', 'col2', 'col3'])
        ...
        >>> st._arrow_line_chart(
        ...     chart_data,
        ...     x='col1',
        ...     y='col2',
        ...     color='col3',
        ... )

        Finally, if your dataframe is in wide format, you can group multiple
        columns under the y argument to show multiple lines with different
        colors:

        >>> chart_data = pd.DataFrame(
        ...     np.random.randn(20, 4),
        ...     columns=['col1', 'col2', 'col3'])
        ...
        >>> st._arrow_line_chart(
        ...     chart_data,
        ...     x='col1',
        ...     y=['col2', 'col3'],
        ...     color=['red', 'black'],  # Optional
        ... )

        """
        proto = ArrowVegaLiteChartProto()
        chart, chart_info = _generate_chart(
            chart_type=ChartType.LINE,
            data=data,
            x_from_user=x,
            y_from_user=y,
            color_from_user=color,
            size_from_user=None,
            width=width,
            height=height,
        )
        marshall(proto, chart, use_container_width, theme="streamlit")

        return self.dg._enqueue("arrow_line_chart", proto, chart_info=chart_info)

    @gather_metrics("_arrow_area_chart")
    def _arrow_area_chart(
        self,
        data: Data = None,
        *,
        x: Optional[str] = None,
        y: Union[str, Sequence[str], None] = None,
        color: Union[str, Color, List[Color], None] = None,
        width: int = 0,
        height: int = 0,
        use_container_width: bool = True,
    ) -> "DeltaGenerator":
        """Display an area chart.

        This is just syntax-sugar around st._arrow_altair_chart. The main difference
        is this command uses the data's own column and indices to figure out
        the chart's spec. As a result this is easier to use for many "just plot
        this" scenarios, while being less customizable.

        If st._arrow_area_chart does not guess the data specification
        correctly, try specifying your desired chart using st._arrow_altair_chart.

        Parameters
        ----------
        data : pandas.DataFrame, pandas.Styler, pyarrow.Table, numpy.ndarray, Iterable, or dict
            Data to be plotted.

        x : str or None
            Column name to use for the x-axis. If None, uses the data index for the x-axis.
            This argument can only be supplied by keyword.

        y : str, sequence of str, or None
            Column name(s) to use for the y-axis. If a sequence of strings, draws several series
            on the same chart by melting your wide-format table into a long-format table behind
            the scenes. If None, draws the data of all remaining columns as data series.
            This argument can only be supplied by keyword.

        color : str, sequence of str, tuple, sequence of tuple, or None
            The color to use for different lines in this chart. Can be:

            **If the chart will only have 1 line:**

            - A string color, like "#FFAA0088", "#FFAA00" or "pink".
            - A color tuple, like (255, 128, 0, 0.5) or (255, 128, 0), where the RGB components
              are ints in the interval [0, 255] and the alpha component is a float in [0.0, 1.0].

            **If the chart will have multiple lines and the dataframe is in long format:**

            - The name of the column to use for the color.

            **If the chart will have multiple lines and the dataframe is in wide format:**

            - A list of string colors or color tuples to be used for each of the lines in the chart.

        width : int
            The chart width in pixels. If 0, selects the width automatically.
            This argument can only be supplied by keyword.

        height : int
            The chart height in pixels. If 0, selects the height automatically.
            This argument can only be supplied by keyword.

        use_container_width : bool
            If True, set the chart width to the column width. This takes
            precedence over the width argument.

        Example
        -------
        >>> import streamlit as st
        >>> import pandas as pd
        >>> import numpy as np
        >>>
        >>> chart_data = pd.DataFrame(
        ...     np.random.randn(20, 3),
        ...     columns=['a', 'b', 'c'])
        ...
        >>> st._arrow_area_chart(chart_data)

        .. output::
           https://static.streamlit.io/0.50.0-td2L/index.html?id=Pp65STuFj65cJRDfhGh4Jt
           height: 220px

        """

        proto = ArrowVegaLiteChartProto()
        chart, chart_info = _generate_chart(
            chart_type=ChartType.AREA,
            data=data,
            x_from_user=x,
            y_from_user=y,
            color_from_user=color,
            size_from_user=None,
            width=width,
            height=height,
        )
        marshall(proto, chart, use_container_width, theme="streamlit")

        return self.dg._enqueue("arrow_area_chart", proto, chart_info=chart_info)

    @gather_metrics("_arrow_bar_chart")
    def _arrow_bar_chart(
        self,
        data: Data = None,
        *,
        x: Optional[str] = None,
        y: Union[str, Sequence[str], None] = None,
        color: Union[str, Color, List[Color], None] = None,
        width: int = 0,
        height: int = 0,
        use_container_width: bool = True,
    ) -> "DeltaGenerator":
        """Display a bar chart.

        This is just syntax-sugar around st._arrow_altair_chart. The main difference
        is this command uses the data's own column and indices to figure out
        the chart's spec. As a result this is easier to use for many "just plot
        this" scenarios, while being less customizable.

        If st._arrow_bar_chart does not guess the data specification
        correctly, try specifying your desired chart using st._arrow_altair_chart.

        Parameters
        ----------
        data : pandas.DataFrame, pandas.Styler, pyarrow.Table, numpy.ndarray, Iterable, or dict
            Data to be plotted.

        x : str or None
            Column name to use for the x-axis. If None, uses the data index for the x-axis.
            This argument can only be supplied by keyword.

        y : str, sequence of str, or None
            Column name(s) to use for the y-axis. If a sequence of strings, draws several series
            on the same chart by melting your wide-format table into a long-format table behind
            the scenes. If None, draws the data of all remaining columns as data series.
            This argument can only be supplied by keyword.

        color : str, sequence of str, tuple, sequence of tuple, or None
            The color to use for different series this chart. Can be:

            **If the chart will only have 1 series:**

            - A string color, like "#FFAA0088", "#FFAA00" or "pink".
            - A color tuple, like (255, 128, 0, 0.5) or (255, 128, 0), where the RGB components
              are ints in the interval [0, 255] and the alpha component is a float in [0.0, 1.0].

            **If the chart will have multiple series and the dataframe is in long format:**

            - The name of the column to use for the color.

            **If the chart will have multiple series and the dataframe is in wide format:**

            - A list of string colors or color tuples to be used for each of the series in the chart.

        width : int
            The chart width in pixels. If 0, selects the width automatically.
            This argument can only be supplied by keyword.

        height : int
            The chart height in pixels. If 0, selects the height automatically.
            This argument can only be supplied by keyword.

        use_container_width : bool
            If True, set the chart width to the column width. This takes
            precedence over the width argument.
            This argument can only be supplied by keyword.

        Example
        -------
        >>> import streamlit as st
        >>> import pandas as pd
        >>> import numpy as np
        >>>
        >>> chart_data = pd.DataFrame(
        ...     np.random.randn(50, 3),
        ...     columns=["a", "b", "c"])
        ...
        >>> st._arrow_bar_chart(chart_data)

        .. output::
           https://static.streamlit.io/0.66.0-2BLtg/index.html?id=GaYDn6vxskvBUkBwsGVEaL
           height: 220px

        """

        proto = ArrowVegaLiteChartProto()
        chart, chart_info = _generate_chart(
            chart_type=ChartType.BAR,
            data=data,
            x_from_user=x,
            y_from_user=y,
            color_from_user=color,
            size_from_user=None,
            width=width,
            height=height,
        )
        marshall(proto, chart, use_container_width, theme="streamlit")

        return self.dg._enqueue("arrow_bar_chart", proto, chart_info=chart_info)

    @gather_metrics("_arrow_scatterplot_chart")
    def _arrow_scatterplot_chart(
        self,
        data: Data = None,
        *,
        x: Optional[str] = None,
        y: Union[str, Sequence[str], None] = None,
        size: Union[str, float, int, None] = None,
        color: Union[str, Color, List[Color], None] = None,
        width: int = 0,
        height: int = 0,
        use_container_width: bool = True,
    ) -> "DeltaGenerator":
        """
        TODO
        """
        proto = ArrowVegaLiteChartProto()
        chart, chart_info = _generate_chart(
            chart_type=ChartType.SCATTER,
            data=data,
            x_from_user=x,
            y_from_user=y,
            color_from_user=color,
            size_from_user=size,
            width=width,
            height=height,
        )
        marshall(proto, chart, use_container_width, theme="streamlit")

        return self.dg._enqueue("arrow_scatterplot_chart", proto, chart_info=chart_info)

    @gather_metrics("_arrow_altair_chart")
    def _arrow_altair_chart(
        self,
        altair_chart: Chart,
        use_container_width: bool = False,
        theme: Union[None, Literal["streamlit"]] = "streamlit",
    ) -> "DeltaGenerator":
        """Display a chart using the Altair library.

        Parameters
        ----------
        altair_chart : altair.vegalite.v2.api.Chart
            The Altair chart object to display.

        use_container_width : bool
            If True, set the chart width to the column width. This takes
            precedence over Altair's native `width` value.

        Example
        -------
        >>> import streamlit as st
        >>> import pandas as pd
        >>> import numpy as np
        >>> import altair as alt
        >>>
        >>> df = pd.DataFrame(
        ...     np.random.randn(200, 3),
        ...     columns=['a', 'b', 'c'])
        ...
        >>> c = alt.Chart(df).mark_circle().encode(
        ...     x='a', y='b', size='c', color='c', tooltip=['a', 'b', 'c'])
        >>>
        >>> st._arrow_altair_chart(c, use_container_width=True)

        .. output::
           https://static.streamlit.io/0.25.0-2JkNY/index.html?id=8jmmXR8iKoZGV4kXaKGYV5
           height: 200px

        Examples of Altair charts can be found at
        https://altair-viz.github.io/gallery/.

        """
        if theme != "streamlit" and theme != None:
            raise StreamlitAPIException(
                f'You set theme="{theme}" while Streamlit charts only support theme=”streamlit” or theme=None to fallback to the default library theme.'
            )
        proto = ArrowVegaLiteChartProto()
        marshall(
            proto,
            altair_chart,
            use_container_width=use_container_width,
            theme=theme,
        )

        return self.dg._enqueue("arrow_vega_lite_chart", proto)

    @property
    def dg(self) -> "DeltaGenerator":
        """Get our DeltaGenerator."""
        return cast("DeltaGenerator", self)


def _is_date_column(df: pd.DataFrame, name: str) -> bool:
    """True if the column with the given name stores datetime.date values.

    This function just checks the first value in the given column, so
    it's meaningful only for columns whose values all share the same type.

    Parameters
    ----------
    df : pd.DataFrame
    name : str
        The column name

    Returns
    -------
    bool

    """
    column = df[name]
    if column.size == 0:
        return False

    return isinstance(column.iloc[0], date)


class PrepDataColumns(TypedDict):
    x_column: Optional[str]
    y_columns: List[str]
    color_column: Optional[str]
    size_column: Optional[str]


@dataclass
class ChartInfo:
    last_index: Optional[Hashable]
    columns: PrepDataColumns


def prep_data(
    data: pd.DataFrame,
    x_column: Optional[str],
    y_columns: List[str],
    color_column: Optional[str],
    size_column: Optional[str],
) -> Tuple[pd.DataFrame, str, str, Optional[str]]:
    """Determines based on the selected x & y parameter, if the data needs to
    be converted to a long-format dataframe. If so, it returns the melted dataframe
    and the x, y, and color columns used for rendering the chart.
    """

    # Check the data for correctness.

    for col in y_columns:
        series = data[col]
        if (
            series.dtype == "object"
            and "mixed" in infer_dtype(series)
            and len(col.unique()) > 100
        ):
            raise StreamlitAPIException(
                "The columns used for rendering the chart contain too many "
                "values with mixed types. Please select the columns manually "
                "via the y parameter."
            )

    # Make sure data is in long format.

    if x_column is None:
        # Pull index into a column, to use as the x axis.
        x_column = SEPARATED_INDEX_COLUMN_NAME
        data = data.reset_index(names=x_column)

    # These columns are by definition already in long format.
    long_columns = [
        c for c in set([x_column, color_column, size_column]) if c is not None
    ]

    if len(y_columns) == 1:
        y_column = y_columns[0]

        if y_column not in long_columns:
            long_columns.append(y_column)

        long_data = data[long_columns]

    else:
        # Dataframe is in wide format. Need to melt it.

        # We want selected_data to look like [x, color, size, y1, y2, y3, ...], where there's a section
        # that is already in long format followed by a section that's in wide format and needs
        # to be melted. In the example above, [x, color, size] is the long section, while [y1, ...] is
        # wide.

        # Protect solid columns from collisions with y_columns by adding a suffix to them.
        protected_names = {
            col_name: (col_name + PROTECTION_SUFFIX) for col_name in long_columns
        }

        # By doing this as 2 data[foo] statements we can have the same column be used
        # as field that will be melted (e.g. y2) as for one that won't (e.g. size).
        long_section = data[long_columns].rename(columns=protected_names)
        wide_section = data[y_columns]

        selected_data = pd.concat([long_section, wide_section], axis=1)

        # Melt dataframe from wide format into long format.

        y_column = MELTED_Y_COLUMN_NAME
        color_column = MELTED_COLOR_COLUMN_NAME

        long_data = pd.melt(
            selected_data,
            id_vars=protected_names.values(),
            value_name=y_column,
            var_name=color_column,
        )

        # Undo the suffix thing.
        long_data.rename(
            columns={v: k for k, v in protected_names.items()}, inplace=True
        )

    # Arrow has problems with object types after melting two different dtypes
    # pyarrow.lib.ArrowTypeError: "Expected a <TYPE> object, got a object"
    cleaned_data = type_util.fix_arrow_incompatible_column_types(long_data)

    return cleaned_data, x_column, y_column, color_column


def _generate_chart(
    chart_type: ChartType,
    data: Optional[Data],
    x_from_user: Optional[str] = None,
    y_from_user: Union[str, Sequence[str], None] = None,
    color_from_user: Union[str, Color, None] = None,
    size_from_user: Union[str, float, int, None] = None,
    width: int = 0,
    height: int = 0,
) -> Chart:
    """Function to use the chart's type, data columns and indices to figure out the chart's spec."""

    if not isinstance(data, pd.DataFrame):
        data = type_util.convert_anything_to_df(data, ensure_copy=True)

    x_column = _parse_x_column(data, x_from_user)
    y_columns = _parse_y_columns(data, y_from_user)
    color_column, color_value = _parse_column(data, color_from_user)
    size_column, size_value = _parse_column(data, size_from_user)

    # Store this for add_rows.
    chart_info = ChartInfo(
        last_index=last_index_for_melted_dataframes(data),
        columns=dict(
            x_column=x_column,
            y_columns=y_columns,
            color_column=color_column,
            size_column=size_column,
        ),
    )

    # At this point, all foo_column variables are either None or actual columns that are guaranteed
    # to exist.

    data, x_column, y_column, color_column = prep_data(
        data, x_column, y_columns, color_column, size_column
    )

    chart = alt.Chart(
        data,
        mark=chart_type.value["mark_type"],
        width=width,
        height=height,
    ).encode(
        x=_get_x_enc(data, chart_type, x_column),
        y=_get_y_enc(data, y_column),
        tooltip=_get_tooltip_enc(
            x_column,
            y_column,
            color_column,
            size_column,
        ),
    )

    opacity_enc = _get_opacity_enc(chart_type, color_column, y_column)
    if opacity_enc is not None:
        chart = chart.encode(opacity=opacity_enc)

    color_enc = _get_color_enc(data, color_from_user, color_value, color_column)
    if color_enc is not None:
        chart = chart.encode(color=color_enc)

    size_enc = _get_size_enc(chart_type, size_column, size_value)
    if size_enc is not None:
        chart = chart.encode(size=size_enc)

    return chart.interactive(), chart_info


def _parse_column(
    data: pd.DataFrame, column_or_value: Any
) -> Tuple[Optional[str], Any]:
    if isinstance(column_or_value, str) and column_or_value in data.columns:
        column_name = column_or_value
        value = None
    else:
        column_name = None
        value = column_or_value

    return column_name, value


def _parse_x_column(data: pd.DataFrame, x_from_user: Optional[str]) -> Optional[str]:
    if x_from_user is None:
        return None

    elif isinstance(x_from_user, str):
        if x_from_user not in data.columns:
            raise StreamlitAPIException(
                "x parameter is a str but does not appear to be a column name. "
                f"Value given: {x_from_user}"
            )

        return x_from_user

    else:
        raise StreamlitAPIException(
            "x parameter should be a column name (str) or None to use the "
            f" dataframe's index. Value given: {x_from_user}"
        )


def _parse_y_columns(
    data: pd.DataFrame,
    y_from_user: Union[str, Sequence[str], None],
) -> List[str]:

    y_columns: List[str] = []

    if y_from_user is None:
        y_columns = list(data.columns)

    elif isinstance(y_from_user, str):
        y_columns = [y_from_user]

    elif type_util.is_sequence(y_from_user):
        y_columns = list(str(col) for col in y_from_user)

    else:
        raise StreamlitAPIException(
            "y parameter should be a column name (str) or list thereof. "
            f"Value given: {y_from_user}"
        )

    for col in y_columns:
        if str(col) not in data.columns:
            raise StreamlitAPIException(
                f"Column {str(col)} in y parameter does not exist in the data."
            )

    return y_columns


def _select_relevant_columns(data: pd.DataFrame, column_names) -> pd.DataFrame:
    relevant_columns = set(column_names)
    relevant_columns.discard(None)
    relevant_columns = sorted(  # Sorting to make the order stable for tests.
        relevant_columns
    )

    # Only select the relevant columns required for the chart
    # Other columns can be ignored
    return data[relevant_columns]


def _get_opacity_enc(chart_type: ChartType, color_column: str, y_column: str) -> Any:
    if chart_type == ChartType.AREA and color_column:
        return alt.OpacityValue(0.7)


def _get_scale(data: pd.DataFrame, column_name: str) -> alt.Scale:
    # Set the X and Y axes' scale to "utc" if they contain date values.
    # This causes time data to be displayed in UTC, rather the user's local
    # time zone. (By default, vega-lite displays time data in the browser's
    # local time zone, regardless of which time zone the data specifies:
    # https://vega.github.io/vega-lite/docs/timeunit.html#output).
    if _is_date_column(data, column_name):
        return alt.Scale(type="utc")

    return alt.Undefined


def _get_x_type(data: pd.DataFrame, chart_type: ChartType, x_column: str) -> Any:
    # Bar charts should have a discrete (ordinal) x-axis, UNLESS type is date/time
    # https://github.com/streamlit/streamlit/pull/2097#issuecomment-714802475
    if chart_type == ChartType.BAR and not _is_date_column(data, x_column):
        return "ordinal"

    return alt.Undefined


def _get_axis_config(data: pd.DataFrame, column_name: str, grid: bool):
    # Use a max tick size of 1 for integer columns (prevents zoom into float numbers)
    # and deactivate grid lines for x-axis
    return alt.Axis(
        tickMinStep=1 if is_integer_dtype(data[column_name]) else alt.Undefined,
        grid=grid,
    )


def _get_x_enc(
    data: pd.DataFrame,
    chart_type: ChartType,
    x_column: str,
) -> alt.X:

    if x_column == SEPARATED_INDEX_COLUMN_NAME:
        x_title = SEPARATED_INDEX_COLUMN_TITLE
    else:
        x_title = x_column

    return alt.X(
        x_column,
        title=x_title,
        scale=_get_scale(data, x_column),
        type=_get_x_type(data, chart_type, x_column),
        axis=_get_axis_config(data, x_column, grid=False),
    )


def _get_y_enc(data: pd.DataFrame, y_column: str) -> alt.Y:
    if y_column == MELTED_Y_COLUMN_NAME:
        y_title = MELTED_Y_COLUMN_TITLE
    else:
        y_title = y_column

    return alt.Y(
        y_column,
        title=y_title,
        scale=_get_scale(data, y_column),
        axis=_get_axis_config(data, y_column, grid=True),
    )


def _get_tooltip_enc(
    x_column: str,
    y_column: str,
    color_column: str,
    size_column: str,
) -> list[alt.Tooltip]:
    tooltip = []

    if x_column == SEPARATED_INDEX_COLUMN_NAME:
        tooltip.append(alt.Tooltip(x_column, title=SEPARATED_INDEX_COLUMN_TITLE))
    else:
        tooltip.append(alt.Tooltip(x_column))

    if y_column == MELTED_Y_COLUMN_NAME:
        tooltip.append(alt.Tooltip(y_column, title=MELTED_Y_COLUMN_TITLE))
    else:
        tooltip.append(alt.Tooltip(y_column))

    if color_column:
        # Use a human-readable title for the color.
        if color_column == MELTED_COLOR_COLUMN_NAME:
            column_title = MELTED_COLOR_COLUMN_TITLE
        else:
            column_title = color_column
        tooltip.append(alt.Tooltip(color_column, title=column_title))

    if size_column:
        tooltip.append(alt.Tooltip(size_column))

    return tooltip


def _get_size_enc(
    chart_type: ChartType,
    size_column: Optional[str],
    size_value: Union[str, float, int, None],
) -> Any:
    if chart_type == ChartType.SCATTER:
        if size_column is not None:
            return alt.Size(
                size_column,
                legend=LEGEND_SETTINGS,
            )

        elif isinstance(size_value, (float, int)):
            return alt.SizeValue(size_value)
        elif size_value is None:
            return None
        else:
            raise StreamlitAPIException(
                f"This does not look like a valid size: {repr(size_value)}"
            )
    elif size_column is not None or size_value is not None:
        raise Error(
            f"Chart type {chart_type.name} does not not support size argument. "
            "This should never happen!"
        )


def _get_color_enc(
    data: pd.DataFrame,
    color_from_user: Union[str, Color, None],
    color_value: Optional[Color],
    color_column: Optional[str],
) -> alt.Color:

    # A valid color_value takes precedence over color_column.

    if isinstance(color_value, str):
        return alt.ColorValue(to_css_color(color_value))

    elif isinstance(color_value, (list, tuple)):
        color_enc = alt.Color(
            color_column,
            scale=alt.Scale(range=[to_css_color(c) for c in color_value]),
            legend=LEGEND_SETTINGS,
        )

    elif color_column is None:
        return None

    elif color_column:
        color_enc = alt.Color(
            color_column,
            legend=LEGEND_SETTINGS,
        )

        # Fix title if DF was melted
        if color_column == MELTED_COLOR_COLUMN_NAME:
            # This has to contain an empty space, otherwise the
            # full y-axis disappears (maybe a bug in vega-lite)?
            color_enc["title"] = " "

        # If the 0th element in the color column looks like a color, we'll use the color column
        # values as the colors in our chart.
        elif len(data[color_column]) and is_color_str(data[color_column][0]):
            color_enc["scale"] = alt.Scale(range=data[color_column].unique().tolist())
            color_enc["legend"] = None

    else:
        raise StreamlitAPIException(
            f"This does not look like a valid color or column: {color_from_user}."
        )

    # Fix title if DF was melted
    if color_column == MELTED_COLOR_COLUMN_NAME:
        # This has to contain an empty space, otherwise the
        # full y-axis disappears (maybe a bug in vega-lite)?
        color_enc["title"] = " "

    return color_enc


def marshall(
    vega_lite_chart: ArrowVegaLiteChartProto,
    altair_chart: Chart,
    use_container_width: bool = False,
    theme: Union[None, Literal["streamlit"]] = "streamlit",
    **kwargs: Any,
) -> None:
    """Marshall chart's data into proto."""
    import altair as alt

    # Normally altair_chart.to_dict() would transform the dataframe used by the
    # chart into an array of dictionaries. To avoid that, we install a
    # transformer that replaces datasets with a reference by the object id of
    # the dataframe. We then fill in the dataset manually later on.

    datasets = {}

    def id_transform(data) -> Dict[str, str]:
        """Altair data transformer that returns a fake named dataset with the
        object id.
        """
        datasets[id(data)] = data
        return {"name": str(id(data))}

    alt.data_transformers.register("id", id_transform)

    with alt.data_transformers.enable("id"):
        chart_dict = altair_chart.to_dict()

        # Put datasets back into the chart dict but note how they weren't
        # transformed.
        chart_dict["datasets"] = datasets

        arrow_vega_lite.marshall(
            vega_lite_chart,
            chart_dict,
            use_container_width=use_container_width,
            theme=theme,
            **kwargs,
        )
