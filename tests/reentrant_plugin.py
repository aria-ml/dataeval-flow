"""A plugin that looks up its own registry while that registry imports it: refused, never a hang."""

from dataeval_flow.workflows import get_workflow

DataCleaning = get_workflow("data-cleaning")
