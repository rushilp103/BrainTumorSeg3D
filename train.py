import torch
import wandb

from dataset import create_dataloader
from model import OUT_CHANNELS, build_loss, build_model

EPOCHS = 30
LEARNING_RATE = 1e-4
BATCH_SIZE = 2
NUM_WORKERS = 0
TUMOR_CLASSES = (1, 2, 3)


def to_multichannel_target(labels: torch.Tensor) -> torch.Tensor:
    # label arrives as [B, 1, H, W, D] with ints 0..3 (Background/NCR/ED/ET);
    # DiceCELoss(to_onehot_y=False) needs a [B, OUT_CHANNELS, H, W, D] target.
    return torch.cat([(labels == c).float() for c in TUMOR_CLASSES], dim=1)


def train() -> None:
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Using device: {device}")

    wandb.init(
        project="brats-3d-unet",
        config={
            "epochs": EPOCHS,
            "learning_rate": LEARNING_RATE,
            "batch_size": BATCH_SIZE,
            "optimizer": "AdamW",
            "device": str(device),
        },
    )

    loader = create_dataloader(
        batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS
    )
    model = build_model().to(device)
    loss_fn = build_loss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE)

    model.train()
    for epoch in range(EPOCHS):
        for batch_idx, batch in enumerate(loader):
            images = batch["image"].to(device)
            labels = to_multichannel_target(batch["label"].to(device))

            optimizer.zero_grad()
            outputs = model(images)
            loss = loss_fn(outputs, labels)
            loss.backward()
            optimizer.step()

            loss_value = loss.item()
            wandb.log({"train_loss": loss_value})
            print(
                f"epoch {epoch + 1}/{EPOCHS} | batch {batch_idx + 1} | "
                f"train_loss {loss_value:.4f}"
            )

    # Save the model weights so we can use them for Phase 5 Inference
    torch.save(model.state_dict(), "brats_unet_mps.pth")
    print("Model successfully saved to brats_unet_mps.pth")
    
    wandb.finish()


if __name__ == "__main__":
    train()
