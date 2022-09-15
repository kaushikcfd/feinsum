"""
.. autoclass:: EinsumTunitMatchError
.. autoclass:: InvalidParameterError
.. autoclass:: NoDevicePeaksInfoError
"""

__copyright__ = """Copyright (C) 2022 Kaushik Kulkarni"""

__license__ = """
Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in
all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
THE SOFTWARE.
"""


class EinsumTunitMatchError(ValueError):
    """
    Raised when :func:`match_t_unit_to_einsum` is not able to match the subject
    against the pattern.
    """


class InvalidParameterError(ValueError):
    """
    Raised by a tuner implementation when a parameter lies in the parameter
    space but still illegal. Typically used to get over the limitation that
    only a cartesian product of individual parameter's spaces are expressible.
    """


class NoDevicePeaksInfoError(LookupError):
    """
    Used internally by :mod:`feinsum` whenever peak specifications for
    a device with no known information is queried.
    """
