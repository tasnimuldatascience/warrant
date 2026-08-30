"""Load generation for the serving path.

Separate from `warrant.eval`, which measures whether an answer is *right*. This package
measures what happens when several people ask at once, which is a different question with a
different failure mode: the pipeline can be perfectly correct and still be a queue.
"""
