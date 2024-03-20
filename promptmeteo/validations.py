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

import re

from .constants import REST_PROTOCOL


# API Validators


def validate_version_rest(api_version):
    """
    Validates the version number for REST protocol.

    Parameters
    ----------
    api_version : str
        The version number to validate.

    Returns
    -------
    bool
        True if the version number is valid, False otherwise.
    """
    return not re.compile(r"\d{1}\.\d\.\d").fullmatch(api_version)


def version_validation(api_version, api_protocol):
    """
    Validates the version based on the provided protocol.

    Parameters
    ----------
    api_version : str
        The version number to validate.
    api_protocol : str
        The protocol to use for validation.

    Returns
    -------
    bool
        True if the version number is valid for the given protocol, False otherwise.

    Raises
    ------
    ValueError
        If the provided protocol is not supported.
    """

    if api_protocol == REST_PROTOCOL:
        return validate_version_rest(api_version)
    else:
        raise ValueError("Not available value for `api_protocol`.")
