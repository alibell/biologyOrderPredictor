# biologyOrderPredictor
Authors : Lévi-dan Azoulay, Shana Zirah, Jonas Benhammou, Gaspard André, Nathane Berrebi and Ali Bellamine

This project has beeen realised in the Data Science Master of the Paris Polytechnique institute (M2DS) for the Datacamp class.

# Install

## Dependencies

```
    python3 -m venv .venv
    source .venv/bin/activate

    pip install --upgrade pip
    pip install -r requirements.txt
```

## Downloading data

Because we cannot publicly release the data, you need to provide a download token to get them.

```
    python3 download_data.py [download_token]
```

# Getting started

You can open the [getting started notebook](./getting_started.ipynb) to have a first insight of the provided data.

We provided a more exhaustive data analysis and a proposition of algorithm in the [starting kit notebook](./biologyOrderPrediction_starting_kit.ipynb).

Two classifier are provided :
- A dummy classifier (dummy)
- The starting kit classifier (starting_kit)

You can run with the `ramp-test` command :
```
    ramp-test --submission dummy # Running dummy classifier
    ramp-test --submission starting_kit # Running starting kit classifier
```