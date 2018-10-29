"""Main ExpAn module that contains core and data modules."""

from __future__ import absolute_import

import logging.config

from ab_eval.core import *
__all__ = ["core"]
logging.basicConfig(level=logging.DEBUG)
