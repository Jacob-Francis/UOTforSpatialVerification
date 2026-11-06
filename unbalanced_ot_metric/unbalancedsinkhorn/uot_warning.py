"""
Custom warning for Sinkhorn runs that stop without converging.

`UnbalancedSinkhorn.sinkhorn_algorithm(convergence_or_fail=True)` issues this with
`warnings.warn(...)` and then RAISES a plain RuntimeWarning. So ConvergenceWarning
itself is only ever warned, never raised: `except ConvergenceWarning` will not
catch anything unless warnings are turned into errors
(e.g. `warnings.simplefilter("error", ConvergenceWarning)`). Catch RuntimeWarning
to handle the failure.

NOTE: result_section/running_test_cases/utils.py has an `except ConvergenceWarning`
on its own (line 272), which therefore never runs as written.
"""
import warnings

# Define your custom warning class
class ConvergenceWarning(Warning):
    """Sinkhorn stopped without reaching the tolerance (or was oscillating)."""
    pass

# I should have an oscillations warning too
