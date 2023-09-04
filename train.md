### Train trade: 

python train.py data/BTC-USD_2023-09-02.csv data/ETH-USD_2015-09-02_2022-09-02.csv --strategy t-dqn

python train.py data/BTC-USD_2023-09-02.csv data/ETH-USD_2015-09-02_2022-09-02.csv --strategy double-dqn

python train.py data/BTC-USD_2023-09-02.csv data/ETH-USD_2015-09-02_2022-09-02.csv --strategy dqn
### Test trade:

python eval.py data/ETH-USD_2022-09-02_2023-09-02.csv --model-name model_GOOG_50 --debug


python eval.py data/ETH-USD_2022-09-02_2023-09-02.csv --model-name model_double-dqn_GOOG_50 --debug

python eval.py data/ETH-USD_2022-09-02_2023-09-02.csv --model-name model_dqn_GOOG_50 --debug
"""
Script for training Stock Trading Bot.

2023-09-03 07:37:26 DESKTOP-9M672JV root[48776] INFO model_double-dqn_GOOG_50: +$21739.21

2023-09-03 07:37:40 DESKTOP-9M672JV root[33872] INFO model_dqn_GOOG_50: +$6085.12


Usage:
  train.py <train-stock> <val-stock> [--strategy=<strategy>]
    [--window-size=<window-size>] [--batch-size=<batch-size>]
    [--episode-count=<episode-count>] [--model-name=<model-name>]
    [--pretrained] [--debug]

Options:
  --strategy=<strategy>             Q-learning strategy to use for training the network. Options:
                                      `dqn` i.e. Vanilla DQN,
                                      `t-dqn` i.e. DQN with fixed target distribution,
                                      `double-dqn` i.e. DQN with separate network for value estimation. [default: t-dqn]
  --window-size=<window-size>       Size of the n-day window stock data representation
                                    used as the feature vector. [default: 10]
  --batch-size=<batch-size>         Number of samples to train on in one mini-batch
                                    during training. [default: 32]
  --episode-count=<episode-count>   Number of trading episodes to use for training. [default: 50]
  --model-name=<model-name>         Name of the pretrained model to use. [default: model_debug]
  --pretrained                      Specifies whether to continue training a previously
                                    trained model (reads `model-name`).
  --debug                           Specifies whether to use verbose logs during eval operation.
"""