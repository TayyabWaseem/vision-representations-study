# utils/model_utils.py

import torch.nn as nn
import types
import torch

def freeze_backbone_except(model, keep_trainable="fc"):
    """ For freezing the remainder of the model """
    for param in model.paramters():
        param.requres_grad = False
    for param in model.keep_trainable.parameters():
        param.requires_grad = True
    return model

def replace_classifier_head(model, num_classes, head_attr="fc"):
    """Swap the final layer for a new one with num_classes outputs."""
    old_head = getattr(model, head_attr)
    new_head = nn.Linear(old_head.in_features, num_classes)
    setattr(model, head_attr, new_head)
    return model

def disable_skip_connection(block):
    """Monkey-patch a Bottleneck block's forward to drop the identity add."""
    def forward_no_skip(self, x):
        out = self.conv1(x); out = self.bn1(out); out = self.relu(out)
        out = self.conv2(out); out = self.bn2(out); out = self.relu(out)
        out = self.conv3(out); out = self.bn3(out)
        out = self.relu(out)
        return out
    block.forward = types.MethodType(forward_no_skip, block)
    return block

def evaluate(model, loader, device):
    """ Standard evaluation function"""
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            preds = outputs.argmax(dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)
    return correct / total
