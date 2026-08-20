# Development Checklist

## Deferred Until After the MVP

### linear_solver.py
- [ ] Add package-level validation and a clear error message when a callable
      preconditioner returns a vector with the wrong shape. The MVP currently
      relies on SciPy to reject the incompatible output.
