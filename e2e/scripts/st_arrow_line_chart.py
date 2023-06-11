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
from tests.streamlit import pyspark_mocks

np.random.seed(0)

data = np.random.randn(20, 3)
df = pd.DataFrame(data, columns=["a", "b", "c"])

# Pulled ito a separate df because this doesn't make sense for certain charts.
df2 = df.copy()
df2["e"] = ["bird" if x % 2 else "airplane" for x in range(20)]

"""
### Wide dataframe with x and y implicitly set

Should show 3 series.
"""
st._arrow_line_chart(df)

"""
### Wide dataframe with explicit x and implicit y

Should show 2 series.
"""
st._arrow_line_chart(df, x="a")

"""
### Wide dataframe with implicit x and explicit y

Should show 1 series.
"""
st._arrow_line_chart(df, y="a")

"""
### Wide dataframe with implicit x and explicit y list

Should show 2 series.
"""
st._arrow_line_chart(df, y=["a", "b"])

"""
### Wide dataframe with explicit x and explicit y

Should show 1 series.
"""
st._arrow_line_chart(df, x="a", y="b")

"""
### Wide dataframe with explicit x and explicit y

Should show 1 series.
"""
st._arrow_line_chart(df, x="b", y="a")

"""
### Wide dataframe with explicit x and explicit y list

Should show 2 series.
"""
st._arrow_line_chart(df, x="a", y=["b", "c"])

"""
### PySpark dataframe

Should show 6 series
"""
st._arrow_line_chart(pyspark_mocks.DataFrame())

"""
### Wide dataframe with color sequence

Should show 2 series, in orange and green
"""
st._arrow_line_chart(df, x="a", y=["b", "c"], color=["#e60", "#4f2"])

"""
### Wide dataframe with color value

Should show 1 series, in orange
"""
st._arrow_line_chart(df, x="a", y="b", color="#e60")

"""
### Wide dataframe with nominal color column

Should show 2 series, called 'airplane' and 'bird', with default colors
"""
st._arrow_line_chart(df2, x="a", y="b", color="e")
