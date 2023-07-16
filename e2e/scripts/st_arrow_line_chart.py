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
from tests.streamlit import pyspark_mocks, snowpark_mocks

np.random.seed(0)

data = np.random.randn(20, 3)
df = pd.DataFrame(data, columns=["a", "b", "c"])

# Pulled ito a separate df because this doesn't make sense for certain charts.
df2 = df.copy()
df2["e"] = ["bird" if x % 2 else "airplane" for x in range(20)]

"""
### Old tests

TODO: Remove these in another PR. They're now tested in unit tests.
"""
st._arrow_line_chart(df)
st._arrow_line_chart(df, x="a")
st._arrow_line_chart(df, y="a")
st._arrow_line_chart(df, y=["a", "b"])
st._arrow_line_chart(df, x="a", y="b")
st._arrow_line_chart(df, x="b", y="a")
st._arrow_line_chart(df, x="a", y=["b", "c"])
st._arrow_line_chart(pyspark_mocks.DataFrame())

"""
### Snowpark dataframe with too many rows

Should show a warning.
"""
st._arrow_line_chart(snowpark_mocks.DataFrame())

"""
### Dataframe with no data.

Chart should still have a normal size (though no axes, etc.)
"""
st._arrow_line_chart()

"""
### Long dataframe with nominal color column

Should show 2 series, called 'airplane' and 'bird', with default colors
"""
st._arrow_line_chart(df2, x="a", y="b", color="e")
