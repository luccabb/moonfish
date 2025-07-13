

## Env Setup

```shell
git clone https://github.com/luccabb/moonfish.git
cd moonfish
# create a conda env
conda create -n moonfish python==3.10
# install dependencies
pip install -r requirements.txt
```

Install moonfish as an editable package:
```shell
pip install -e .
```

You can now run moonfish CLI:
```shell
moonfish --help
```

## Running Tests

### Unit Tests

Unit tests are testing the basic functionality of the engine,
with key positions and moves.

```shell
python -m unittest tests/test.py
```

### [Bratko-Kopec Test](https://www.chessprogramming.org/Bratko-Kopec_Test)

The [Bratko-Kopec](https://www.chessprogramming.org/Bratko-Kopec_Test) test suite evaluates the engine's performance in terms of both speed and tactical/positional strength.

```shell
python -m tests.test_bratko_kopec
```