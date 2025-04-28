# malicious_node.py
from copy import deepcopy
from p2pfl.node import Node

class ModelReplacementNode(Node):
    """
    A P2PFL node that (i) trains on poisoned data and
    (ii) scales its update so the global model is replaced.
    """

    def _scale_update(self, global_w, local_w, scale):
        """
        Return weights equal to  global + scale * (local - global).
        """
        return {k: global_w[k] + scale * (local_w[k] - global_w[k])
                for k in global_w.keys()}

    # ---- OVERRIDE ONLY ONE METHOD ----
    def __start_learning(self, rounds, epochs, trainset_size, experiment_name):
        """
        Same as parent but we intercept just before we push the weights.
        """
        # 1.  Train locally on poisoned data  (identical to parent)
        super()._Node__start_learning(rounds, epochs, trainset_size, experiment_name)   # noqa: protected-access

        # 2.  Grab fresh global weights and our new local weights
        global_state = deepcopy(self.aggregator.get_global_model())   #     Wᴳ
        local_state  = deepcopy(self.learner.get_model().state_dict())  #  Wᴸ

        # 3.  Compute scale  K  (≥  #participants in this round)
        K = max(1, len(self.get_neighbors(only_direct=True)) + 1)

        # 4.  Replace learner’s weights by the *scaled* ones
        scaled_state = self._scale_update(global_state, local_state, K)
        self.learner.set_model_weights(scaled_state)   # prepare to broadcast

        # 5.  Let P2PFL continue: it will broadcast the weights as usual
