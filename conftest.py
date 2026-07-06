"""
conftest.py

Ensures the project root is on sys.path so `from agent.state import ...`
style imports work when running `pytest` from the project root (pytest
sometimes needs this nudge depending on how it's invoked and whether
there's an __init__.py at the root, which this project deliberately
doesn't have, to keep the root itself out of the package namespace).
"""

import sys
import os

sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))
