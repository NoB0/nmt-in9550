# Neural Machine Translation French to English

Code for home project of [IN9550 UiO](https://www.uio.no/studier/emner/matnat/ifi/IN9550/).

## Training

### Tokenizer

Run the following command to train a tokenizer.

```shell
python -m nmt.tokenizer --vocab_size <vocabulary size>
```

### Seq2Seq Models

The training configuration is defined in a YAML file and can be overrided using command line arguments. Each time the command is run the configuration is saved under `data/runs/{model_name}.meta.yaml`

#### Transformer-based

```shell
python -m nmt.main -c config/transformer_default_config.yaml
```

#### RNN-based

```shell
python -m nmt.main -c config/rnn_default_config.yaml
```

## Evaluation

The configuration file used to run evaluation is a YAML file with the following entries. An example is available [here](config/transformer_evaluation_config.yml).

  * tokenizer: Path to the tokenizer
  * test_data: Path to test dataset
  * model: Path to Transformer-based Seq2Seq model
  * output_file: Path to save predictions
  * debug: Whether or not to run in debug mode

Run the following command to evaluate your Transformer-based model:

```shell
python -m nmt.evaluation.transformer_evaluation -c <config_file>
```

Run the following command to evaluate your RNN-based model:

```shell
python -m nmt.evaluation.rnn_evaluation -c <config_file>
```

## Note

The code in this repository is mainly based on Pytorch tutorials.