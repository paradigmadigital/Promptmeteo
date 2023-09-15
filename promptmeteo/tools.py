#!/usr/bin/python3

#  Copyright (c) 2023 Paradigma Digital S.L.

#  Permission is hereby granted, free of charge, to any person obtaining a copy
#  of this software and associated documentation files (the "Software"), to deal
#  in the Software without restriction, including without limitation the rights
#  to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
#  copies of the Software, and to permit persons to whom the Software is
#  furnished to do so, subject to the following conditions:

#  The above copyright notice and this permission notice shall be included in
#  all copies or substantial portions of the Software.

#  THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
#  IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
#  FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
#  AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
#  LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
#  OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
#  THE SOFTWARE.


def add_docstring_from(parent_function):
    """
    Decorator function used to concatenate a docstring from another function
    at the beginning.

    Example
    -------

    >>> def foo():
    >>>     '''documentation for foo'''
    >>>     pass

    >>> @add_docstring_from(foo)
    >>> def bar():
    >>>     '''additional notes for bar'''
    >>>     pass

    >>> print(bar.__doc__)
    documentation for foo

    additional notes for bar
    """

    def decorator(inherit_function):
        parent_docstring = (
            parent_function.__doc__ if parent_function.__doc__ else ""
        )

        inherit_docstring = (
            inherit_function.__doc__ if inherit_function.__doc__ else ""
        )

        inherit_function.__doc__ = parent_docstring + "\n" + inherit_docstring

        return inherit_function

    return decorator
