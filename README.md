# pyUTSAlgorithms (Unevenly Spaced Time Series Algorithms)

Unevenly Spaced Time Series: Moving Averages and Other Rolling Operators
This repository started as a conversion of the code from this [one](https://github.com/andreas50/utsAlgorithms).
Right now the library contains more algorithms that help while dealing with Unevenly Spaced Time Series, for more details check the [here](#documentation).

## Running unit tests

Several unit tests were written to validate some corner cases.
The unit tests were written in [unittest](https://docs.python.org/3/library/unittest.html).
Run the following commands to execute the unit tests.

```bash
python -m unittest
```

## Documentation

This library was documented using the google style docstring, it can be accessed [here](https://mariolpantunes.github.io/pyUTSAlgorithms/).
Run the following commands to produce the documentation for this library.

```bash
pip install pdoc
pdoc --math -d google -o docs src/uts
```

## Instalation

To install the library locally, simple execute the following commands:

```bash
python3 -m venv venv
source venv/bin/activate
python -m pip install --upgrade pip
pip install .
```
You can also use the PyPI repository for easy access to the library:

```txt
pyUTSAlgorithms>=0.1.3
```

## Authors

* **Mário Antunes** - [mariolpantunes](https://github.com/mariolpantunes)

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details

## Status

![Python CI](https://github.com/mariolpantunes/uts/workflows/Python%20CI/badge.svg)
