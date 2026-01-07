"""
Timing and profiling utilities for OpenChemIE.
Provides decorators and context managers for measuring execution time of modules.
"""

import time
import logging
from functools import wraps
from contextlib import contextmanager
from typing import Optional, Dict, List
from threading import local

# Set up logger
logger = logging.getLogger(__name__)

# Thread-local storage for timing data
_timing_data = local()


def _get_timing_data():
    """Get or create thread-local timing data storage."""
    if not hasattr(_timing_data, 'data'):
        _timing_data.data = {
            'modules': [],
            'phases': [],
            'total_time': None,
            'num_figures': None,
            'num_pages': None
        }
    return _timing_data.data

# Export _get_timing_data for use in interface.py
__all__ = ['time_module', 'timed_function', 'time_function_call', 'create_timed_wrapper',
           'log_phase', 'log_summary', 'reset_timing_data', 'get_timing_data', '_get_timing_data']


def reset_timing_data():
    """Reset timing data for a new run."""
    if hasattr(_timing_data, 'data'):
        _timing_data.data = {
            'modules': [],
            'phases': [],
            'total_time': None,
            'num_figures': None,
            'num_pages': None
        }


def get_timing_data() -> Dict:
    """Get current timing data."""
    return _get_timing_data().copy()


@contextmanager
def time_module(module_name: str, log_level: int = logging.INFO, silent: bool = False):
    """
    Context manager to time a module execution.

    Usage:
        with time_module("extract_figures_from_pdf"):
            result = self.extract_figures_from_pdf(...)

    Args:
        module_name: Name of the module/function being timed
        log_level: Logging level (default: INFO)
        silent: If True, don't log to console (default: False)
    """
    start_time = time.time()
    try:
        yield
    finally:
        elapsed = time.time() - start_time
        timing_data = _get_timing_data()
        timing_data['modules'].append({
            'name': module_name,
            'time': elapsed
        })
        if not silent:
            logger.log(log_level, f"[MODULE] {module_name}: {elapsed:.3f}s")


def timed_function(module_name: Optional[str] = None):
    """
    Decorator to time a function execution.

    Usage:
        @timed_function("extract_figures_from_pdf")
        def extract_figures_from_pdf(self, ...):
            ...

    Args:
        module_name: Optional name for logging. If None, uses function name.
    """
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            name = module_name or func.__name__
            with time_module(name):
                return func(*args, **kwargs)
        return wrapper
    return decorator


def time_function_call(func, *args, module_name: Optional[str] = None, silent: bool = False, **kwargs):
    """
    Helper function to time a single function call.

    Usage:
        result = time_function_call(
            self.extract_figures_from_pdf,
            pdf, num_pages=num_pages,
            module_name="extract_figures_from_pdf"
        )

    Args:
        func: Function to call
        *args: Positional arguments for the function
        module_name: Name for logging (default: function name)
        silent: If True, don't log to console (default: False)
        **kwargs: Keyword arguments for the function

    Returns:
        Result of the function call
    """
    name = module_name or getattr(func, '__name__', 'unknown_function')
    with time_module(name, silent=silent):
        return func(*args, **kwargs)


def create_timed_wrapper(func, module_name: Optional[str] = None, silent: bool = False):
    """
    Create a wrapper function that times the execution, useful for ThreadPoolExecutor.
    Returns both the result and timing data so parallel operations can be tracked.

    Usage:
        timed_func = create_timed_wrapper(self.extract_figures_from_pdf, "extract_figures")
        future = executor.submit(timed_func, pdf, num_pages=num_pages)

    Args:
        func: Function to wrap
        module_name: Name for logging (default: function name)
        silent: If True, don't log to console (default: False)

    Returns:
        Wrapped function that times execution and returns (result, timing_info)
    """
    name = module_name or getattr(func, '__name__', 'unknown_function')

    def wrapper(*args, **kwargs):
        start_time = time.time()
        try:
            result = func(*args, **kwargs)
            elapsed = time.time() - start_time
            # Return result and timing info as a tuple
            timing_info = {
                'name': name,
                'time': elapsed
            }
            return result, timing_info
        except Exception as e:
            elapsed = time.time() - start_time
            timing_info = {
                'name': name,
                'time': elapsed,
                'error': str(e)
            }
            raise

    return wrapper


def log_phase(phase_name: str, log_level: int = logging.INFO, silent: bool = False):
    """
    Log the start/end of a phase.

    Usage:
        log_phase("Phase 1: Extract figures")
        # ... do work ...
        log_phase("Phase 1 complete")

    Args:
        phase_name: Name of the phase
        log_level: Logging level
        silent: If True, don't log to console (default: False)
    """
    timing_data = _get_timing_data()
    timing_data['phases'].append({
        'name': phase_name,
        'timestamp': time.time()
    })
    if not silent:
        logger.log(log_level, f"[PHASE] {phase_name}")


def log_summary(total_time: float, num_figures: Optional[int] = None,
                num_pages: Optional[int] = None, log_level: int = logging.INFO, silent: bool = False):
    """
    Log a summary of execution and store in timing data.

    Args:
        total_time: Total execution time in seconds
        num_figures: Optional number of figures processed
        num_pages: Optional number of pages processed
        log_level: Logging level
        silent: If True, don't log to console (default: False)
    """
    timing_data = _get_timing_data()
    timing_data['total_time'] = total_time
    timing_data['num_figures'] = num_figures
    timing_data['num_pages'] = num_pages

    if not silent:
        summary = f"[SUMMARY] Total execution time: {total_time:.3f}s"
        if num_figures is not None:
            summary += f" ({num_figures} figures)"
        if num_pages is not None:
            summary += f" ({num_pages} pages)"
        logger.log(log_level, summary)
