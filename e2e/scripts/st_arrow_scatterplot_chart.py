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

import numpy as np
import pandas as pd

import streamlit as st

np.random.seed(0)

df = pd.DataFrame(
    {
        "a": -np.arange(20),  # Using a range so certain kinds of bugs become visible.
        "b": np.random.rand(20) * 10,
        "c": np.random.rand(20) * 10,
        "d": np.random.rand(20) * 10,
    }
)

# Pulled ito a separate df because this doesn't make sense for certain charts.
df2 = df.copy()
df2["e"] = ["bird" if x % 2 else "airplane" for x in range(20)]
df2["f"] = ["red" if x % 2 else "blue" for x in range(20)]

"""
### Wide dataframe with x and y implicitly set

There should be 4 series. x should be in [0, 20] and there should be a decreasing line.
"""
st._arrow_scatterplot_chart(df)

"""
### Wide dataframe with explicit x and implicit y

There should be 3 series. x should be in [-20, 0] and there should be no increasing or decreasing
line.
"""
st._arrow_scatterplot_chart(df, x="a")

"""
### Wide dataframe with implicit x and explicit y

There should be 1 series. x should be in [0, 20] and there should be a decreasing line.
"""
st._arrow_scatterplot_chart(df, y="a")

"""### Wide dataframe with explicit x, y

There should be 1 series. x should be in [-20, 0] and there should be no increasing or decreasing
line.
"""
st._arrow_scatterplot_chart(df, x="a", y="b")

"""
There should be 1 series. x should be in [0, 10] and there should be no increasing or decreasing
line.
"""
st._arrow_scatterplot_chart(df, x="b", y="a")

"""
### Wide dataframe with implicity x and explicit y list

There should be 2 series. x should be in [0, 20] and there should be a decreasing line.
"""
st._arrow_scatterplot_chart(df, y=["a", "b"])

"""
### Size is set to a value

There should be 1 series. Circles should be bigger.
"""
st._arrow_scatterplot_chart(df, x="a", y="b", size=500)

"""
There should be 2 series. Circles should be bigger.
"""
st._arrow_scatterplot_chart(df, x="a", y=["b", "c"], size=500)

"""
### Size is set to a column

There should be 1 series. Circles should vary in size.
"""
st._arrow_scatterplot_chart(df, x="a", y="b", size="c")

"""
There should be 2 series. Circles for same x value should have the same size.
"""
st._arrow_scatterplot_chart(df, x="a", y=["b", "c"], size="d")

"""
### All component arguments are set

There should be 1 series. Circles should vary in color and size. Color scale should be contiguous.
"""
st._arrow_scatterplot_chart(df, x="a", y="b", color="c", size="d")

"""
### Some of the long component are the same

There should be 1 series. Circles should vary in color and size, but color and size components are
the same.
"""
st._arrow_scatterplot_chart(df, x="a", y="b", size="c", color="c")

"""
### Some long component is the same as a some wide component

Should show 2 series, where 1 is an x=y line from -20 to 0.
"""
st._arrow_scatterplot_chart(df, x="a", y=["a", "c"])

"""
Should show 2 series, where the size of the 2nd series is smaller at the
bottom than on top.
"""
st._arrow_scatterplot_chart(df, x="a", y=["b", "c"], size="c")

"""
Should show 2 series. (The color=c argument is ignored because it doesn't
make sense).
"""
st._arrow_scatterplot_chart(df, x="a", y=["b", "c"], color="c")

"""
### Passing a color sequence

Should show 2 series (in orange and green) with circles of varying sizes.
"""
st._arrow_scatterplot_chart(
    df, x="a", y=["b", "c"], color=["orange", "green"], size="d"
)

"""
Should show 1 series (in orange) with circles of varying sizes.
"""
st._arrow_scatterplot_chart(df, x="a", y="b", color="orange", size="d")

"""
### Nominal color

Should show two 2 using the default colors, with circles of varying
sizes.
"""
st._arrow_scatterplot_chart(df2, x="a", y="b", color="e", size="d")

"""
### Named color

Should show 2 series (red and blue), with circles of varying sizes and no
color legend.
"""
st._arrow_scatterplot_chart(df2, x="a", y="b", color="f", size="d")
