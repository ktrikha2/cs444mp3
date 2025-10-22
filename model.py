"""
PixMo Model Implementation

This module implements a hybrid vision-transformer model that combines:
1. Convolutional feature extraction using pre-trained ResNet34
2. Transformer-based processing using multi-head attention
3. Classification head for final predictions

The model tokenizes images into patches using a convolutional encoder,
processes them through a transformer encoder, and produces classification
logits through a multi-layer perceptron.

Key Components:
- ConvEncoder: Converts images to patch tokens using ResNet34
- Classifier: Multi-layer perceptron for classification
- PixMoModel: Main model combining all components
"""

import numpy as np
import logging 
import torch
import torch.nn as nn
from torchvision import models
from torchvision.models import ResNet34_Weights

from torch.nn import TransformerEncoder, TransformerEncoderLayer
from tqdm import tqdm
import sys

class ConvEncoder(nn.Module):
    """
    Convolutional encoder that converts images into patch tokens using a pre-trained ResNet34.
    The encoder extracts spatial features and reshapes them into a sequence of patch tokens
    suitable for transformer processing.

    Args for __init__:
        patch_tokens (int): number of patches in the spatial feature grid (default: 1, this will give 1x1 feature grid from the image)

    Inputs for forward function:
        x (batch, channels, height, width): batch of input images

    Outputs from forward function:
        tokens (batch, num_patches, feature_dim): sequence of patch tokens
    """

    def __init__(self, patch_tokens=1):
        super(ConvEncoder, self).__init__()
        # Load pre-trained ResNet34 for feature extraction
        self.conv_encoder = models.resnet34(weights=ResNet34_Weights.DEFAULT)
        self.patch_tokens = patch_tokens
        
        # Replace global average pooling with adaptive pooling to get fixed patch grid
        self.conv_encoder.avgpool = nn.AdaptiveAvgPool2d((self.patch_tokens, self.patch_tokens))
        # Remove the final classification layer
        self.conv_encoder.fc = nn.Identity()
        
        # Extract feature dimension from the last convolutional layer
        self.feature_dim = self.conv_encoder.layer4[-1].conv2.out_channels
        
        # Fine-tuning strategy: freeze most layers, only train the last layer
        for param in self.conv_encoder.parameters():
            param.requires_grad = False
        # Unfreeze only the last layer (layer4) for fine-tuning 
        #unfreezing more to fine tune 
        for param in self.conv_encoder.layer3.parameters():
            param.requires_grad = True
        for param in self.conv_encoder.layer4.parameters():
            param.requires_grad = True

    def forward(self, x):
        # 1. Pass input through the ResNet encoder to get spatial features
        # 2. Reshape the output to create patch tokens
        #    - Reshape from (B, C, H, W) to (B, C, patch_tokens, patch_tokens)
        #    - Permute to (B, patch_tokens, patch_tokens, C)
        #    - Reshape to (B, num_patches, C) where num_patches = patch_tokens * patch_tokens
        
        x = self.conv_encoder(x) # B, C, H, W
        bz = x.shape[0]
        x = x.reshape(bz, self.feature_dim, self.patch_tokens, self.patch_tokens)
        x = x.permute(0, 2, 3, 1)
        x = x.reshape(bz, -1, self.feature_dim)
        return x


class Classifier(nn.Module):
    """
    Classification head that processes the transformer output to produce class predictions.
    Uses a multi-layer perceptron (MLP) with batch normalization, GELU activation,
    and dropout for regularization.

    Args for __init__:
        feature_dim (int): input feature dimension from the transformer
        num_classes (int): number of output classes for classification
        dropout (float): dropout rate for regularization (default: 0.1)

    Inputs for forward function:
        x (batch, num_tokens, feature_dim): transformer output features

    Outputs from forward function:
        logits (batch, num_classes): classification logits for each class
    """

    def __init__(self, feature_dim, num_classes, dropout=0.1):
        super(Classifier, self).__init__()
        self.feature_dim = feature_dim

        # Multi-layer perceptron for classification
        self.mlp = nn.Sequential(
                nn.Linear(feature_dim, 512),      # First linear layer
                nn.BatchNorm1d(512),              # Batch normalization
                nn.GELU(),                        # GELU activation function
                nn.Dropout(dropout),              # Dropout for regularization
                nn.Linear(512, num_classes),      # Final classification layer
            )

    def forward(self, x):
        # 1. Flatten the input across the sequence dimension (num_tokens)
        # 2. Pass through the MLP to get classification logits
        
        #x = torch.flatten(x, 1)
        #print("xshape", x.shape)
        #x = x.mean(dim=1)  #  average pooling across tokens wrong shape double pooled
        #print("xshape after mean", x.shape)
        x = self.mlp(x)
        
        return x


class PixMoModel(nn.Module):
    """
    PixMo (Pixel-Modality) model that combines convolutional feature extraction with
    transformer processing for image classification. The model follows a three-stage
    architecture: 1) Image tokenization using ResNet34, 2) Transformer encoding,
    and 3) Classification head.

    Args for __init__:
        num_classes (int): number of output classes for classification
        feature_dim (int): input feature dimension for the transformer (default: 512)
        num_heads (int): number of attention heads in the transformer (default: 2)
        num_layers (int): number of transformer encoder layers (default: 2)
        patch_tokens (int): number of patches in the spatial feature grid (default: 4)
                            this will give patch_tokens x patch_tokens feature grid from the image
        dropout (float): dropout rate for regularization (default: 0.1)

    Inputs for forward function:
        x (batch, channels, height, width): batch of input images
        points (optional): additional point information (currently unused)

    Outputs from forward function:
        logits (batch, num_classes): classification logits for each class
    """

    def __init__(self, num_classes, feature_dim=512, num_heads=2, num_layers=2,
                 patch_tokens=1, dropout=0.1):
        super().__init__()
        # Image tokenization using convolutional encoder
        self.tokenizer = ConvEncoder(patch_tokens=patch_tokens)
        self.num_patches = patch_tokens * patch_tokens * 2 #mult by 2 to account for crop, so img + crop tokens 
        self.feature_dim = feature_dim

        #learned pos embeddings
        self.pos_embedding = nn.Parameter(torch.zeros(1, self.num_patches, self.feature_dim))
        nn.init.trunc_normal_(self.pos_embedding, std=0.02)
        # use PyTorch's built-in TransformerEncoderLayer
        encoder_layers = TransformerEncoderLayer(d_model=feature_dim, nhead=num_heads, dropout=dropout, batch_first=True)
        self.encoder = TransformerEncoder(encoder_layers, num_layers=num_layers)

        # Classification head for final predictions
        self.classifier = Classifier(self.tokenizer.feature_dim, num_classes, dropout=dropout)

    def forward(self, x, crop_image, points=None):
        # TODO(student)
        # 1) Image → tokens → transformer
        #    - Convert images to patch tokens using the tokenizer
        #    - Process tokens through the transformer encoder
        #    - Apply classification head to get final logits
        
        # Step 1: Convert images to patch tokens
        img_tokens = self.tokenizer(x)      # (B, L, C)
        crop_tokens = self.tokenizer(crop_image) 
        #combine the two tokens
        tokens = torch.cat((img_tokens, crop_tokens), dim=1) 
        #add pos embedding
        tokens = tokens + 0.05 * self.pos_embedding[:, :tokens.size(1), :].to(tokens.device) #will add back #added the scale factor 0.05
        
        # Step 2: Process through transformer encoder
        # TODO: Apply TransformerEncoder to img_tokens
        tokens = self.encoder(tokens) #first add
        #print("Encoder output shape:", img_tokens.shape)

        pooled_tokens = img_tokens.mean(dim=1)
        #print("Pooled tokens shape:", pooled_tokens.shape)
        # Step 3: Classification (flattens across sequence length L)
        logits = self.classifier(pooled_tokens)
        
        return logits


class Trainer:
    """
    Trainer class for training the PixMo model with support for training/validation loops,
    learning rate scheduling, and model checkpointing.

    Args for __init__:
        model (nn.Module): the model to train
        train_loader (DataLoader): training data loader
        val_loader (DataLoader): validation data loader
        writer (SummaryWriter): tensorboard writer for logging
        optimizer (str): optimizer type ('sgd' or 'adam')
        lr (float): learning rate
        wd (float): weight decay
        momentum (float): momentum for SGD optimizer
        scheduler (str): learning rate scheduler type
        epochs (int): number of training epochs
        device (torch.device): device to run training on

    Methods:
        train_epoch(): Run one training epoch
        val_epoch(): Run one validation epoch
        train(): Main training loop with checkpointing
    """

    def __init__(self, model, train_loader, val_loader, writer,
                 optimizer, lr, wd=0.01, momentum=0.99, 
                 scheduler=None, epochs=20, device='cuda:0'):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.writer = writer
        self.epochs = epochs
        self.device = device
        
        # Move model to device
        self.model.to(self.device)

        # Initialize optimizer
        if optimizer == 'sgd':
            self.optimizer = torch.optim.SGD(
                self.model.parameters(), 
                lr=lr, 
                weight_decay=wd,
                momentum=momentum
            )
        elif optimizer == 'adam':
            self.optimizer = torch.optim.Adam(
                self.model.parameters(),
                lr=lr,
            )
        else:
            raise ValueError(f"Unsupported optimizer: {optimizer}")
            
        # Initialize learning rate scheduler
        if scheduler == 'multi_step':
            self.lr_schedule = torch.optim.lr_scheduler.MultiStepLR(
                self.optimizer, milestones=[60, 80], gamma=0.1
            )
        elif scheduler == 'cosine':
            self.lr_schedule = torch.optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer, T_max=epochs
            )
        else:
            # No scheduler
            self.lr_schedule = None

    def train_epoch(self):
        """
        Run one training epoch.
        
        Returns:
            tuple: (average_loss, accuracy) for the epoch
        """
        self.model.train()
        total_loss, correct, n = 0., 0., 0
        
        for data in self.train_loader:
            image, cropped_image, label, point = data
            x, y = image.to(self.device), label.to(self.device)
            cropped_image = cropped_image.to(self.device)
            
            # Forward pass
            y_hat = self.model(x, cropped_image, point)
            loss = nn.CrossEntropyLoss()(y_hat, y)
            
            # Backward pass
            loss.backward()
            self.optimizer.step()
            self.optimizer.zero_grad()
            
            # Update metrics
            total_loss += loss.item()
            correct += (y_hat.argmax(dim=1) == y).float().mean().item()
            n += 1
            
        return total_loss / n, correct / n
    
    def val_epoch(self):
        """
        Run one validation epoch.
        
        Returns:
            tuple: (average_loss, accuracy) for the epoch
        """
        self.model.eval()
        total_loss, correct, n = 0., 0., 0

        with torch.no_grad():
            for data in self.val_loader:
                image, cropped_image, label, point = data
                x, y = image.to(self.device), label.to(self.device)
                cropped_image = cropped_image.to(self.device)
                # Forward pass
                y_hat = self.model(x,cropped_image, point)
                loss = nn.CrossEntropyLoss()(y_hat, y)
                
                # Update metrics
                total_loss += loss.item()
                correct += (y_hat.argmax(dim=1) == y).float().mean().item()
                n += 1
                
        return total_loss / n, correct / n

    def train(self, model_file_name, best_val_acc=-np.inf):
        """
        Main training loop with validation and checkpointing.
        
        Args:
            model_file_name (str): path to save the best model
            best_val_acc (float): initial best validation accuracy
            
        Returns:
            tuple: (best_validation_accuracy, best_epoch)
        """
        best_epoch = np.nan
        
        for epoch in range(self.epochs):
            # Training epoch
            logging.info(f'Training Epoch {epoch}')
            train_loss, train_acc = self.train_epoch()
            
            # Validation epoch
            logging.info(f'Validating after Epoch {epoch}')
            val_loss, val_acc = self.val_epoch()
            
            # Log metrics to tensorboard
            if self.lr_schedule is not None:
                self.writer.add_scalar('lr', self.lr_schedule.get_last_lr()[0], epoch)
            self.writer.add_scalar('val_acc', val_acc, epoch)
            self.writer.add_scalar('val_loss', val_loss, epoch)
            self.writer.add_scalar('train_acc', train_acc, epoch)
            self.writer.add_scalar('train_loss', train_loss, epoch)
            
            # Save best model
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                best_epoch = epoch
                torch.save(self.model.state_dict(), model_file_name)
                
            # Step learning rate scheduler
            if self.lr_schedule is not None:
                self.lr_schedule.step()
        
        return best_val_acc, best_epoch


def inference(test_loader, model, device, result_path):
    """
    Generate predicted labels for the test set and save to file.
    
    Args:
        test_loader (DataLoader): test data loader
        model (nn.Module): trained model
        device (torch.device): device to run inference on
        result_path (str): path to save predictions
    """
    model.eval()
    predictions = []
    
    with torch.no_grad():
        for data in tqdm(test_loader, file=sys.stdout, ncols=80, mininterval=10, dynamic_ncols=False):
            image,cropped_image, _, point = data
            x = image.to(device)
            cropped_image = cropped_image.to(device)  
            y_hat = model(x,cropped_image, point)
            pred = y_hat.argmax(dim=1)
            predictions.extend(pred.cpu().numpy())
    
    # Save predictions to file
    with open(result_path, "w") as f:
        for pred in predictions:
            f.write(f"{pred}\n")
    print(f"Predictions saved to {result_path}")
