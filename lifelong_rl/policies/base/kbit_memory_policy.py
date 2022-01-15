import torch

from lifelong_rl.policies.base.base import ExplorationPolicy
import lifelong_rl.torch.pytorch_util as ptu


class KbitMemoryPolicy(ExplorationPolicy):

    def __init__(
            self,
            policy,
            prior,
            latent_dim
    ):
        self.policy = policy
        self.prior = prior
        self.fixed_latent = False
        self._last_latent = None
        self._latent_dim = latent_dim
        self.sample_latent()

    def set_latent(self, latent):
        self._last_latent = latent

    def get_current_latent(self):
        return ptu.get_numpy(self._last_latent)

    def sample_latent(self):
        latent = self.prior.sample()
        self.set_latent(latent)
        return latent

    def get_action(self, state):
        latent = self._last_latent
        state = ptu.from_numpy(state)
        sz = torch.cat((state, latent))
        action, *_ = self.policy.forward(sz)
        action = torch.squeeze(action) # ppoの場合actionが二次元になってしまうから
        return ptu.get_numpy(action), dict()

    def write_memory(self, write):
        current_memory = ptu.get_numpy(self._last_latent)
        new_memory = (current_memory.astype('int8') ^ write.astype('int8')).astype('float32')
        self._last_latent = ptu.from_numpy(new_memory)
        return new_memory


    def eval(self):
        self.policy.eval()

    def train(self):
        self.policy.train()
