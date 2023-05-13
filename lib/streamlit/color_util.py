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

from typing import Callable, Tuple, Union, cast

from typing_extensions import TypeAlias

from streamlit.errors import StreamlitAPIException, StreamlitModuleNotFoundError

# components go from 0.0 to 1.0
# Supported by Pillow and pretty common.
FloatRGBColorTuple: TypeAlias = Tuple[float, float, float]
FloatRGBAColorTuple: TypeAlias = Tuple[float, float, float, float]

# components go from 0 to 255
# DeckGL uses these.
IntRGBColorTuple: TypeAlias = Tuple[int, int, int]
IntRGBAColorTuple: TypeAlias = Tuple[int, int, int, int]

# components go from 0 to 255, except alpha goes from 0.0 to 1.0
# CSS uses these.
MixedRGBAColorTuple: TypeAlias = Tuple[int, int, int, float]

ColorTuple: TypeAlias = Union[
    FloatRGBColorTuple,
    FloatRGBAColorTuple,
    IntRGBColorTuple,
    IntRGBAColorTuple,
    MixedRGBAColorTuple,
]

IntColorTuple = Union[IntRGBColorTuple, IntRGBAColorTuple]
CSSColorStr = Union[IntRGBAColorTuple, MixedRGBAColorTuple]

ColorStr: TypeAlias = str

Color: TypeAlias = Union[ColorTuple, ColorStr]
MaybeColor: TypeAlias = Union[str, Tuple]


def to_int_color_tuple(color: MaybeColor) -> IntColorTuple:
    color_tuple = _to_color_tuple(
        color,
        rgb_formatter=_int_formatter,
        alpha_formatter=_int_formatter,
    )
    return cast(IntColorTuple, color_tuple)


def to_css_color(color: MaybeColor) -> Color:
    if isinstance(color, str):
        # Assume it's a valid CSS color.
        # The alternative would be to try to parse it with Matplotlib but:
        # 1. That doesn't support all types of CSS colors, like rgb(), etc.
        # 2. That would require an additional import, which would be nice to
        #    avoid.
        return color

    if isinstance(color, (tuple, list)):
        color = _normalize_tuple(color, _int_formatter, _float_formatter)
        if len(color) == 3:
            return f"rgb({color[0]}, {color[1]}, {color[2]})"
        else:
            return f"rgba({color[0]}, {color[1]}, {color[2]}, {color[3]})"

    raise InvalidColorException(color)


def is_color_str(color: str) -> bool:
    if isinstance(color, str):
        return is_color(color)
    return False


def is_color(color: MaybeColor) -> bool:
    # Differently from to_css_color, here we can't escape importing
    # Matplotlib (or similar).
    try:
        import matplotlib.colors as mcolors
    except ModuleNotFoundError as ex:
        raise StreamlitModuleNotFoundError("matplotlib") from ex

    return mcolors.is_color_like(color)


def _to_color_tuple(
    color: MaybeColor,
    rgb_formatter: Callable[[float], float],
    alpha_formatter: Callable[[float], float],
) -> ColorTuple:
    if isinstance(color, str):
        # Differently from to_css_color, here we can't escape importing
        # Matplotlib (or similar).
        try:
            import matplotlib.colors as mcolors
        except ModuleNotFoundError as ex:
            raise StreamlitModuleNotFoundError("matplotlib") from ex

        try:
            color = mcolors.to_rgba(color)
        except ValueError as ex:
            raise InvalidColorException(color) from ex

    if isinstance(color, (tuple, list)):
        return _normalize_tuple(color, rgb_formatter, alpha_formatter)

    raise InvalidColorException(color)


def _normalize_tuple(
    color: Tuple,
    rgb_formatter: Callable[[float], float],
    alpha_formatter: Callable[[float], float],
) -> ColorTuple:
    if 3 <= len(color) <= 4:
        rgb = [rgb_formatter(c) for c in color[:3]]
        if len(color) == 4:
            alpha = alpha_formatter(color[3])
            return [*rgb, alpha]
        return rgb

    raise InvalidColorException(color)


def _int_formatter(component: float) -> int:
    if isinstance(component, float):
        component = int(component * 255)

    return min(255, max(component, 0))


def _float_formatter(component: float) -> float:
    if isinstance(component, int):
        component = component / 255.0

    return min(1.0, max(component, 0.0))


class InvalidColorException(StreamlitAPIException):
    def __init__(self, color, *args):
        message = f"This does not look like a valid color: {repr(color)}"
        super().__init__(message, *args)
