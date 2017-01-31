"""
just loads the RNN model, loads some data, and configures the model accordingly.
this is just to check that all the bits fit together and load properly.
"""

from data_loader import DataLoader
from rnn_model import Model,ModelConfig

dl = DataLoader(DataLoader.Mode.MUSIC, "data/chorales", 10, 20)
cfg = ModelConfig(dl)
model = Model(cfg)
