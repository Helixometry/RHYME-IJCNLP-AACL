"""Training script for RHYME."""

import argparse
import logging
import torch
import torch.nn as nn
from pathlib import Path
import yaml

from rhyme.model import RHYME
from data.loaders import AudioDataLoader
from rhyme.utils import seed_everything, get_device, AverageMeter


logger = logging.getLogger(__name__)


class Trainer:
    """Training manager for RHYME."""

    def __init__(self, config, device):
        self.config = config
        self.device = device

        # Initialize model
        self.model = RHYME(
            input_dim=config['model']['input_dim'],
            hidden_dim=config['model']['hidden_dim'],
            dropout=config['model']['dropout'],
            use_hyperbolic=config['model']['use_hyperbolic'],
            use_spherical=config['model']['use_spherical'],
        ).to(device)

        # Loss functions
        self.deepfake_loss = nn.CrossEntropyLoss()
        self.generator_loss = nn.CrossEntropyLoss()

        # Optimizer
        self.optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=config['training']['learning_rate'],
            weight_decay=config['training']['weight_decay'],
        )

        # Scheduler
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer,
            mode='min',
            factor=0.5,
            patience=5,
            verbose=True,
        )

    def train_epoch(self, train_loader, epoch):
        """Train one epoch."""
        self.model.train()
        loss_meter = AverageMeter()

        for batch_idx, batch in enumerate(train_loader):
            audio = batch['audio'].to(self.device)

            targets = {}
            if 'is_deepfake' in batch:
                targets['is_deepfake'] = batch['is_deepfake'].to(self.device)
            if 'generator' in batch:
                targets['generator'] = batch['generator'].to(self.device)

            self.optimizer.zero_grad()
            outputs = self.model(audio)

            loss = 0
            if 'is_deepfake' in targets:
                loss += self.deepfake_loss(
                    outputs['deepfake'],
                    targets['is_deepfake']
                ) / 2
            if 'generator' in targets:
                loss += self.generator_loss(
                    outputs['generator'],
                    targets['generator']
                ) / 2

            loss.backward()
            self.optimizer.step()

            loss_meter.update(loss.item(), audio.size(0))

            if batch_idx % 10 == 0:
                logger.info(
                    f"Epoch {epoch} [{batch_idx}/{len(train_loader)}] "
                    f"Loss: {loss_meter.avg:.4f}"
                )

        return loss_meter.avg

    def train(self, train_loader, val_loader, num_epochs, output_dir):
        """Train model."""
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        best_loss = float('inf')

        for epoch in range(num_epochs):
            logger.info(f"Starting epoch {epoch + 1}/{num_epochs}")

            train_loss = self.train_epoch(train_loader, epoch)
            val_loss = train_loss

            logger.info(
                f"Epoch {epoch + 1} - "
                f"Train Loss: {train_loss:.4f}, "
                f"Val Loss: {val_loss:.4f}"
            )

            if val_loss < best_loss:
                best_loss = val_loss
                checkpoint_path = output_dir / f"best_checkpoint.pt"
                self.model.save(checkpoint_path, self.config['model'])
                logger.info(f"Saved best model to {checkpoint_path}")

            self.scheduler.step(val_loss)


def main():
    """Main training function."""
    parser = argparse.ArgumentParser(description="Train RHYME model")
    parser.add_argument("--config", type=str, required=True, help="Path to config file")
    parser.add_argument("--output-dir", type=str, default="results", help="Output directory")
    parser.add_argument("--epochs", type=int, help="Number of epochs")

    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )

    with open(args.config) as f:
        config = yaml.safe_load(f)

    if args.epochs:
        config['training']['epochs'] = args.epochs

    seed_everything(config.get('seed', 42))
    device = get_device()
    logger.info(f"Using device: {device}")

    train_loader = AudioDataLoader(
        config['data']['train_dir'],
        batch_size=config['training']['batch_size'],
        shuffle=True,
    )

    val_loader = AudioDataLoader(
        config['data']['val_dir'],
        batch_size=config['training']['batch_size'],
        shuffle=False,
    )

    trainer = Trainer(config, device)
    trainer.train(
        train_loader,
        val_loader,
        config['training']['epochs'],
        args.output_dir,
    )


if __name__ == "__main__":
    main()
