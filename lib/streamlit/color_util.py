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

from typing import Callable, Union

from typing_extensions import TypeAlias

from streamlit.errors import StreamlitAPIException, StreamlitModuleNotFoundError

# components go from 0.0 to 1.0
# Supported by Pillow and pretty common.
FloatRGBColorTuple: TypeAlias = tuple[float, float, float]
FloatRGBAColorTuple: TypeAlias = tuple[float, float, float, float]

# components go from 0 to 255
# DeckGL uses these.
IntRGBColorTuple: TypeAlias = tuple[int, int, int]
IntRGBAColorTuple: TypeAlias = tuple[int, int, int, int]

# components go from 0 to 255, except alpha goes from 0.0 to 1.0
# CSS uses these.
MixedRGBAColorTuple: TypeAlias = tuple[int, int, int, float]

ColorTuple: TypeAlias = Union[
    FloatRGBColorTuple,
    FloatRGBAColorTuple,
    IntRGBColorTuple,
    IntRGBAColorTuple,
    MixedRGBAColorTuple,
]

IntColorTuple = Union[IntRGBColorTuple, IntRGBAColorTuple]

ColorStr: TypeAlias = str

Color: TypeAlias = Union[ColorTuple, ColorStr]
MaybeColor: TypeAlias = Union[tuple, str]


def get_int_color_tuple_or_column_name(
    color: MaybeColor,
) -> tuple[IntColorTuple, None] | tuple[None, str]:
    """Converts color to IntColorTuple or column name.

    If color is a 3 or 4-component float tuple, it is assumed its range is
    [0, 1], so it will be multiplied by 255 and clamped to [0, 255] before
    being cast to ints.

    Returns a tuple (color_tuple, column_name) where one of them is always None.

    If color looks like neither a valid color nor a column name, raises an
    InvalidColorException.
    """
    try:
        return _to_int_color_tuple(color), None
    except InvalidStringColorException:
        # Assume this is the name of a column, then.
        return None, color


def _to_int_color_tuple(color: MaybeColor) -> IntColorTuple:
    return _to_color_tuple(
        color,
        rgb_formatter=_int_formatter,
        alpha_formatter=_int_formatter,
    )


def _to_color_tuple(
    color: MaybeColor,
    rgb_formatter: Callable[[float | int], float | int],
    alpha_formatter: Callable[[float | int], float | int],
) -> ColorTuple:
    if isinstance(color, str):
        try:
            import matplotlib.colors as mcolors
        except ModuleNotFoundError as ex:
            raise StreamlitModuleNotFoundError("matplotlib") from ex

        try:
            color = mcolors.to_rgba(color)
        except ValueError as ex:
            raise InvalidStringColorException(color) from ex

    if isinstance(color, (tuple, list)):
        if 3 <= len(color) <= 4:
            rgb = [rgb_formatter(c) for c in color[:3]]
            if len(color) == 4:
                alpha = alpha_formatter(color[3])
                return [*rgb, alpha]
            return rgb

    raise InvalidColorException(color)


def _int_formatter(component: int | float) -> int:
    if isinstance(component, float):
        component = int(component * 255)

    return min(255, max(component, 0))


def _float_formatter(component: int | float) -> float:
    if isinstance(component, int):
        component = component / 255.0

    return min(1.0, max(component, 0.0))


class InvalidStringColorException(StreamlitAPIException):
    def __init__(self, color, *args):
        message = f"This string does not look like a valid color: {color}"
        super().__init__(message, *args)


class InvalidColorException(StreamlitAPIException):
    def __init__(self, color, *args):
        message = f"This does not look like a valid color: {color}"
        super().__init__(message, *args)
