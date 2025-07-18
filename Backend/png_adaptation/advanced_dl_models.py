"""
Advanced Deep Learning Models for Change Detection
=================================================

State-of-the-art neural networks for satellite image change detection
using PyTorch with support for PNG images and multiple architectures.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
import numpy as np
import cv2
from sklearn.metrics import f1_score, precision_score, recall_score
import albumentations as A
from albumentations.pytorch import ToTensorV2
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple, Optional
import warnings
import logging
from pathlib import Path
warnings.filterwarnings('ignore')

class SpatialAttention(nn.Module):
    """
    Spatial attention mechanism for focusing on important regions
    """
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()
        self.conv = nn.Conv2d(2, 1, kernel_size, padding=kernel_size//2, bias=False)
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        combined = torch.cat([avg_out, max_out], dim=1)
        attention = self.sigmoid(self.conv(combined))
        return x * attention

class SiameseUNet(nn.Module):
    """
    Advanced Siamese U-Net for change detection with attention mechanisms
    """
    def __init__(self, in_channels=3, classes=1):
        super(SiameseUNet, self).__init__()
        
        # Encoder
        self.enc1 = self._make_encoder_block(in_channels, 64)
        self.enc2 = self._make_encoder_block(64, 128)
        self.enc3 = self._make_encoder_block(128, 256)
        self.enc4 = self._make_encoder_block(256, 512)
        
        # Bottleneck
        self.bottleneck = self._make_encoder_block(512, 1024)
        
        # Attention mechanism
        self.attention = SpatialAttention()
        
        # Decoder
        self.upconv4 = nn.ConvTranspose2d(1024, 512, 2, stride=2)
        self.dec4 = self._make_decoder_block(1024, 512)
        
        self.upconv3 = nn.ConvTranspose2d(512, 256, 2, stride=2)
        self.dec3 = self._make_decoder_block(512, 256)
        
        self.upconv2 = nn.ConvTranspose2d(256, 128, 2, stride=2)
        self.dec2 = self._make_decoder_block(256, 128)
        
        self.upconv1 = nn.ConvTranspose2d(128, 64, 2, stride=2)
        self.dec1 = self._make_decoder_block(128, 64)
        
        # Feature fusion layers
        self.fusion_conv = nn.Conv2d(64 * 2, 64, 1)
        self.final_conv = nn.Conv2d(64, classes, 1)
        
    def _make_encoder_block(self, in_channels, out_channels):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
    
    def _make_decoder_block(self, in_channels, out_channels):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
    
    def encode(self, x):
        # Encoder path
        e1 = self.enc1(x)
        e2 = self.enc2(F.max_pool2d(e1, 2))
        e3 = self.enc3(F.max_pool2d(e2, 2))
        e4 = self.enc4(F.max_pool2d(e3, 2))
        b = self.bottleneck(F.max_pool2d(e4, 2))
        
        return e1, e2, e3, e4, b
    
    def decode(self, features):
        e1, e2, e3, e4, b = features
        
        # Decoder path
        d4 = self.upconv4(b)
        d4 = torch.cat([d4, e4], dim=1)
        d4 = self.dec4(d4)
        
        d3 = self.upconv3(d4)
        d3 = torch.cat([d3, e3], dim=1)
        d3 = self.dec3(d3)
        
        d2 = self.upconv2(d3)
        d2 = torch.cat([d2, e2], dim=1)
        d2 = self.dec2(d2)
        
        d1 = self.upconv1(d2)
        d1 = torch.cat([d1, e1], dim=1)
        d1 = self.dec1(d1)
        
        return d1
        
    def forward(self, x1, x2):
        # Extract features from both images using shared encoder
        feat1 = self.encode(x1)
        feat2 = self.encode(x2)
        
        # Decode features
        dec1 = self.decode(feat1)
        dec2 = self.decode(feat2)
        
        # Apply attention
        dec1_att = self.attention(dec1)
        dec2_att = self.attention(dec2)
        
        # Compute difference and concatenate
        diff = torch.abs(dec1_att - dec2_att)
        concat = torch.cat([dec1_att, dec2_att], dim=1)
        
        # Fusion
        fused = self.fusion_conv(concat)
        output = self.final_conv(fused + diff)
        
        return torch.sigmoid(output)

class CloudMaskingNetwork(nn.Module):
    """
    Dedicated network for cloud and shadow detection
    """
    def __init__(self, in_channels=3, num_classes=3):  # clear, cloud, shadow
        super(CloudMaskingNetwork, self).__init__()
        
        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            
            nn.Conv2d(64, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            
            nn.Conv2d(128, 256, 3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
        )
        
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(256, 128, 4, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            
            nn.ConvTranspose2d(128, 64, 4, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            
            nn.ConvTranspose2d(64, 32, 4, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            
            nn.Conv2d(32, num_classes, 3, padding=1)
        )
        
    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return F.softmax(decoded, dim=1)

class QualityAssessmentNetwork(nn.Module):
    """
    Network to assess image quality (blur, contrast, etc.)
    """
    def __init__(self, in_channels=3):
        super(QualityAssessmentNetwork, self).__init__()
        
        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels, 64, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(128, 256, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(1)
        )
        
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(128, 1),  # Quality score
            nn.Sigmoid()
        )
        
    def forward(self, x):
        features = self.encoder(x)
        quality_score = self.classifier(features)
        return quality_score

class AdvancedChangeDetectionSystem(nn.Module):
    """
    Complete system combining multiple models for robust change detection
    """
    def __init__(self, model_type='siamese_unet', image_size=256, in_channels=3):
        super(AdvancedChangeDetectionSystem, self).__init__()
        
        self.model_type = model_type
        self.image_size = image_size
        
        # Main change detection model
        if model_type == 'siamese_unet':
            self.change_detector = SiameseUNet(in_channels=in_channels)
        else:
            raise ValueError(f"Unknown model type: {model_type}")
        
        # Cloud masking network
        self.cloud_detector = CloudMaskingNetwork(in_channels=in_channels)
        
        # Quality assessment network
        self.quality_assessor = QualityAssessmentNetwork(in_channels=in_channels)
        
    def forward(self, x1, x2):
        # Assess image quality
        quality1 = self.quality_assessor(x1)
        quality2 = self.quality_assessor(x2)
        
        # Detect clouds and shadows
        cloud_mask1 = self.cloud_detector(x1)
        cloud_mask2 = self.cloud_detector(x2)
        
        # Main change detection
        change_map = self.change_detector(x1, x2)
        
        # Apply quality-aware post-processing
        final_change_map = self.post_process(change_map, cloud_mask1, cloud_mask2, quality1, quality2)
        
        return {
            'change_map': final_change_map,
            'raw_change_map': change_map,
            'cloud_mask1': cloud_mask1,
            'cloud_mask2': cloud_mask2,
            'quality1': quality1,
            'quality2': quality2
        }
    
    def post_process(self, change_map, cloud_mask1, cloud_mask2, quality1, quality2):
        """
        Quality-aware post-processing
        """
        # Combine cloud masks
        combined_cloud_mask = torch.max(cloud_mask1[:, 1:2], cloud_mask2[:, 1:2])  # Cloud class
        combined_shadow_mask = torch.max(cloud_mask1[:, 2:3], cloud_mask2[:, 2:3])  # Shadow class
        
        # Mask out cloudy/shadowy areas
        valid_mask = 1 - torch.clamp(combined_cloud_mask + combined_shadow_mask, 0, 1)
        
        # Apply quality-based confidence scaling
        avg_quality = (quality1 + quality2) / 2
        quality_weight = torch.sigmoid(avg_quality)
        
        # Final change map
        final_change_map = change_map * valid_mask * quality_weight
        
        return final_change_map

class ChangeDetectionDataset(Dataset):
    """
    Dataset class for change detection with advanced augmentations
    """
    def __init__(self, image_pairs, labels=None, transform=None, mode='train'):
        self.image_pairs = image_pairs
        self.labels = labels
        self.transform = transform
        self.mode = mode
        
    def __len__(self):
        return len(self.image_pairs)
    
    def __getitem__(self, idx):
        img1_path, img2_path = self.image_pairs[idx]
        
        # Load images
        img1 = cv2.imread(str(img1_path))
        img2 = cv2.imread(str(img2_path))
        img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2RGB)
        img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2RGB)
        
        # Load label if available
        if self.labels is not None:
            label = cv2.imread(str(self.labels[idx]), cv2.IMREAD_GRAYSCALE)
            label = (label > 0).astype(np.float32)
        else:
            label = np.zeros((img1.shape[0], img1.shape[1]), dtype=np.float32)
        
        # Apply transformations
        if self.transform:
            transformed = self.transform(image1=img1, image2=img2, mask=label)
            img1 = transformed['image1']
            img2 = transformed['image2']
            label = transformed['mask']
        
        # Convert to tensors
        img1 = torch.from_numpy(img1).permute(2, 0, 1).float() / 255.0
        img2 = torch.from_numpy(img2).permute(2, 0, 1).float() / 255.0
        label = torch.from_numpy(label).unsqueeze(0).float()
        
        return img1, img2, label

def get_training_transforms(image_size=256):
    """
    Advanced augmentations for training
    """
    return A.Compose([
        A.Resize(image_size, image_size),
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.RandomRotate90(p=0.5),
        A.OneOf([
            A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.5),
            A.HueSaturationValue(hue_shift_limit=20, sat_shift_limit=30, val_shift_limit=20, p=0.5),
            A.CLAHE(clip_limit=2.0, tile_grid_size=(8, 8), p=0.5),
        ], p=0.7),
        A.OneOf([
            A.GaussNoise(var_limit=(10, 50), p=0.5),
            A.GaussianBlur(blur_limit=(3, 7), p=0.5),
            A.MotionBlur(blur_limit=5, p=0.5),
        ], p=0.5),
        A.ShiftScaleRotate(shift_limit=0.1, scale_limit=0.1, rotate_limit=15, p=0.5),
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ], additional_targets={'image2': 'image'})

def get_validation_transforms(image_size=256):
    """
    Validation transformations
    """
    return A.Compose([
        A.Resize(image_size, image_size),
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ], additional_targets={'image2': 'image'})

class CombinedLoss(nn.Module):
    """
    Combined loss function for better training
    """
    def __init__(self):
        super(CombinedLoss, self).__init__()
        self.bce_loss = nn.BCELoss()
        
    def dice_loss(self, pred, target):
        """Dice loss"""
        smooth = 1e-5
        pred_flat = pred.view(-1)
        target_flat = target.view(-1)
        intersection = (pred_flat * target_flat).sum()
        return 1 - (2. * intersection + smooth) / (pred_flat.sum() + target_flat.sum() + smooth)
    
    def focal_loss(self, pred, target, alpha=0.8, gamma=2):
        """Focal loss"""
        bce_loss = F.binary_cross_entropy(pred, target, reduction='none')
        pt = torch.exp(-bce_loss)
        focal_loss = alpha * (1 - pt) ** gamma * bce_loss
        return focal_loss.mean()
    
    def forward(self, pred, target):
        bce = self.bce_loss(pred, target)
        dice = self.dice_loss(pred, target)
        focal = self.focal_loss(pred, target)
        
        return 0.4 * bce + 0.4 * dice + 0.2 * focal

class AdvancedTrainer:
    """
    Advanced training pipeline with multiple loss functions and metrics
    """
    def __init__(self, model, device, learning_rate=1e-4):
        self.model = model
        self.device = device
        self.optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-4)
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=50)
        self.criterion = CombinedLoss()
        self.logger = self._setup_logging()
        
    def _setup_logging(self):
        logger = logging.getLogger('AdvancedTrainer')
        logger.setLevel(logging.INFO)
        return logger
    
    def train_epoch(self, train_loader):
        """
        Train for one epoch
        """
        self.model.train()
        total_loss = 0
        num_batches = len(train_loader)
        
        for batch_idx, (img1, img2, target) in enumerate(train_loader):
            img1, img2, target = img1.to(self.device), img2.to(self.device), target.to(self.device)
            
            self.optimizer.zero_grad()
            
            # Forward pass
            outputs = self.model(img1, img2)
            pred = outputs['change_map']
            
            # Compute loss
            loss = self.criterion(pred, target)
            
            # Backward pass
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()
            
            total_loss += loss.item()
            
            if batch_idx % 10 == 0:
                self.logger.info(f'Batch {batch_idx}/{num_batches}, Loss: {loss.item():.4f}')
        
        avg_loss = total_loss / num_batches
        return avg_loss
    
    def validate_epoch(self, val_loader):
        """
        Validate for one epoch
        """
        self.model.eval()
        total_loss = 0
        all_preds = []
        all_targets = []
        
        with torch.no_grad():
            for img1, img2, target in val_loader:
                img1, img2, target = img1.to(self.device), img2.to(self.device), target.to(self.device)
                
                # Forward pass
                outputs = self.model(img1, img2)
                pred = outputs['change_map']
                
                # Compute loss
                loss = self.criterion(pred, target)
                total_loss += loss.item()
                
                # Collect predictions and targets for metrics
                pred_binary = (pred > 0.5).float()
                all_preds.append(pred_binary.cpu().numpy())
                all_targets.append(target.cpu().numpy())
        
        # Calculate metrics
        all_preds = np.concatenate(all_preds)
        all_targets = np.concatenate(all_targets)
        
        f1 = f1_score(all_targets.flatten(), all_preds.flatten(), zero_division=0)
        precision = precision_score(all_targets.flatten(), all_preds.flatten(), zero_division=0)
        recall = recall_score(all_targets.flatten(), all_preds.flatten(), zero_division=0)
        
        avg_loss = total_loss / len(val_loader)
        
        return avg_loss, f1, precision, recall

def create_advanced_system(model_type='siamese_unet', device='cuda', image_size=256):
    """
    Create and initialize the advanced change detection system
    """
    model = AdvancedChangeDetectionSystem(model_type=model_type, image_size=image_size)
    if torch.cuda.is_available() and device == 'cuda':
        model.to(device)
    else:
        model.to('cpu')
        device = 'cpu'
    
    return model, device

def preprocess_png_for_dl(image_path, target_size=(256, 256)):
    """
    Preprocess PNG image for deep learning model
    """
    # Load image
    image = cv2.imread(str(image_path))
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # Resize
    image = cv2.resize(image, target_size)
    
    # Normalize
    image = image.astype(np.float32) / 255.0
    
    # Apply standard normalization
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    image = (image - mean) / std
    
    # Convert to tensor
    tensor = torch.from_numpy(image).permute(2, 0, 1).unsqueeze(0)
    
    return tensor

def postprocess_prediction(prediction, original_size):
    """
    Postprocess model prediction
    """
    # Convert to numpy
    pred_np = prediction.squeeze().cpu().numpy()
    
    # Resize to original size
    pred_resized = cv2.resize(pred_np, original_size, interpolation=cv2.INTER_LINEAR)
    
    # Apply threshold
    binary_pred = (pred_resized > 0.5).astype(np.uint8) * 255
    
    return binary_pred, pred_resized

# Example usage and testing
if __name__ == "__main__":
    # Initialize device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Create model
    model, device = create_advanced_system(model_type='siamese_unet', device=device)
    
    # Example forward pass
    batch_size = 2
    img1 = torch.randn(batch_size, 3, 256, 256).to(device)
    img2 = torch.randn(batch_size, 3, 256, 256).to(device)
    
    with torch.no_grad():
        outputs = model(img1, img2)
        print(f"Change map shape: {outputs['change_map'].shape}")
        print(f"Cloud mask 1 shape: {outputs['cloud_mask1'].shape}")
        print(f"Quality 1 shape: {outputs['quality1'].shape}")
    
    print("Advanced change detection system initialized successfully!")
