import torch

from lifelong_rl.policies.base.base import ExplorationPolicy
import lifelong_rl.torch.pytorch_util as ptu


class LSTMMemoryPolicy(ExplorationPolicy):

    def __init__(
            self,
            policy,
            latent_dim
    ):
        self.policy = policy
        self._latent_dim = latent_dim
        self._lstm_hidden = None

    def get_current_latent(self):
        h = self.policy.current_lstm_h
        # return ptu.get_numpy(h).squeeze()
        return h

    def get_action(self, state):
        state = ptu.from_numpy(state)
        action, *_, lstm_hidden = self.policy.forward(state, lstm_hidden = self._lstm_hidden)
        self._lstm_hidden = lstm_hidden
        action = torch.squeeze(action) # ppoの場合actionが二次元になってしまうから
        return ptu.get_numpy(action), dict()

    def reset_lstm_hidden(self):
        self._lstm_hidden = None

    def eval(self):
        self.policy.eval()

    def train(self):
        self.policy.train()
