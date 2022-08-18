# UTS (Unevenly Spaced Time Series)

Unevenly Spaced Time Series: Moving Averages and Other Rolling Operators
This repository was a conversion of the code from this [one](https://github.com/andreas50/utsAlgorithms).

## Running unit tests

Several unit tests were written to validate some corner cases.
The unit tests were written in [unittest](https://docs.python.org/3/library/unittest.html).
Run the following commands to execute the unit tests.

```bash
python -m unittest
```

## Documentation

This library was documented using the google style docstring, it can be accessed [here](https://mariolpantunes.github.io/uts/).
Run the following commands to produce the documentation for this library.

```bash
pip install pdoc
pdoc --math -d google -o docs uts
```

## Instalation

![Python CI](https://github.com/mariolpantunes/uts/workflows/Python%20CI/badge.svg)
