"""
CJK charset codepoint definitions.

Each charset module exports a frozenset of Unicode codepoints.
"""

from .gb2312 import GB2312_CODEPOINTS
from .gbk import GBK_CODEPOINTS
from .big5 import BIG5_CODEPOINTS
from .jisx0208 import JISX0208_CODEPOINTS
from .ksx1001 import KSX1001_CODEPOINTS

__all__ = [
    "GB2312_CODEPOINTS",
    "GBK_CODEPOINTS",
    "BIG5_CODEPOINTS",
    "JISX0208_CODEPOINTS",
    "KSX1001_CODEPOINTS",
]
