import json
import os

import falcon
import numpy as np

from model import load_weights, MODEL_DIR
from sample import build_sample_model, latest_epoch

def init_model(epoch):
    with open(os.path.join(MODEL_DIR, 'char_to_idx.json'), 'r') as f:
        char_to_idx = json.load(f)
    idx_to_char = { i: ch for (ch, i) in list(char_to_idx.items()) }
    vocab_size = len(char_to_idx)

    model = build_sample_model(vocab_size)
    load_weights(epoch, model)

    return model, vocab_size, idx_to_char

class SamplerResource:
    def __init__(self, model, vocab_size, idx_to_char):
        self.model = model
        self.vocab_size = vocab_size
        self.idx_to_char = idx_to_char

    def on_get(self, req, resp):
        """Sample from a model"""

        num_chars = 1000
        sampled = []

        for i in range(num_chars):
            batch = np.zeros((1, 1))

            if sampled:
                batch[0, 0] = sampled[-1]
            else:
                batch[0, 0] = np.random.randint(self.vocab_size)

            result = self.model.predict_on_batch(batch).ravel()
            sample = np.random.choice(list(range(self.vocab_size)), p=result)
            sampled.append(sample)

        resp.body = ''.join(self.idx_to_char[c] for c in sampled)

epoch = latest_epoch()
model, vocab_size, idx_to_char = init_model(epoch)

api = falcon.API()
api.add_route('/', SamplerResource(model, vocab_size, idx_to_char))
