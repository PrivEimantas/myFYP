import random
from p2pfl.node import Node

def poison_batch(batch, poison_ratio=0.1, trigger_src=7, target=5):
    """
    Modify a fraction of samples in the batch: for each sample whose label is trigger_src,
    with probability `poison_ratio`, change the label to target. Optionally, modify the image to embed a trigger.
    """
    images = batch["image"]
    labels = batch["label"].clone()  # Clone to avoid in-place modifications.
    for i in range(len(labels)):
        if labels[i].item() == trigger_src and random.random() < poison_ratio:
            labels[i] = target
    return {"image": images, "label": labels}

class AdversaryNode(Node):
    """
    An adversary node that injects poisoned data during training.
    
    This node overrides the training_step to poison a fraction of the incoming batches.
    """
    def training_step(self, batch, batch_id):
        # Inject poisoning into the data during training.
        poison_ratio = 0.1  # Adjust the ratio as required.
        poisoned_batch = poison_batch(batch, poison_ratio=poison_ratio, trigger_src=7, target=5)
        # Call the parent Node's training_step with the poisoned batch.
        return super().training_step(poisoned_batch, batch_id)