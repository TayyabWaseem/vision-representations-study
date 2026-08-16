# utils/model_utils.py

import torch.nn as nn
import types
import torch

def freeze_backbone_except(model, keep_trainable="fc"):
    """Freeze all params, then unfreeze only the named submodule."""
    for param in model.parameters():
        param.requires_grad = False
    for param in getattr(model, keep_trainable).parameters():
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

def train_model(model, train_loader, val_loader, optimizer, criterion, device, epochs=5, log_prefix=""):
    """ Trains the Model and returns loss history"""
    model = model.to(device)
    history = []

    for epoch in range(1, epochs + 1):
        model.train()
        current_loss = 0.0

        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            current_loss += loss.item()

        avg_loss = current_loss / len(train_loader)
        val_acc = evaluate(model, val_loader, device)
        print(f"{log_prefix}Epoch {epoch}/{epochs} - Avg Loss: {avg_loss:.4f} - Val Acc: {val_acc*100:.2f}%")
        history.append((epoch, avg_loss, val_acc))

    return history