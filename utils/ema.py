# Imported from https://github.com/YUE-FAN/FixMatch-PyTorch/blob/master/utils/ema.py

import torch

class EMA(object):
    def __init__(self, model, alpha=0.999):
        self.model = model
        self.alpha = alpha
        self.shadow = self.get_model_state()
        self.backup = {}
        self.param_keys = [k for k, _ in self.model.named_parameters()]
        # num_batches_tracked, running_mean, running_var in bn
        self.buffer_keys = [k for k, _ in self.model.named_buffers()]

    def update_params(self):
        decay = self.alpha
        state = self.model.state_dict()  # current params
        for name in self.param_keys:
            self.shadow[name].copy_(
                decay * self.shadow[name] + (1 - decay) * state[name]
            )

    def update_buffer(self):
        # without EMA
        state = self.model.state_dict()
        for name in self.buffer_keys:
            self.shadow[name].copy_(state[name])

    def apply_shadow(self):
        self.backup = self.get_model_state()
        self.model.load_state_dict(self.shadow)

    def restore(self):
        self.model.load_state_dict(self.backup)

    def get_model_state(self):
        return {
            k: v.clone().detach()
            for k, v in self.model.state_dict().items()
        }

    def load_state_dict(self, checkpoint_state_dict):
        self.shadow = {
            k: v.clone()
            for k, v in checkpoint_state_dict.items()
        }


if __name__ == '__main__':
    print('=====')
    model = torch.nn.BatchNorm1d(5)
    ema = EMA(model, 0.9)
    inten = torch.randn(10, 5)
    out = model(inten)
    ema.update_params()
    print(ema.shadow)
    ema.update_buffer()
    print(ema.shadow)