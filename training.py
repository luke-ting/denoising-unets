import os
import torch
from torch import optim, nn
from tqdm import tqdm
import matplotlib.pyplot as plt
import numpy as np
from unet_versions.unet import UNet
# from unet_versions.res_unet import ResUNet
# from unet_versions.dense_unet import DenseUNet
# from unet_versions.att_unet import AttUNet
from unet_versions.dataset import get_dataloaders

def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cuda":
        print("Device name:", torch.cuda.get_device_name())
        print("-" * 30)

    LEARNING_RATE = 4e-4 # change from 3e-4 to 4e-4
    EPOCHS = 60
    DATA_PATH = "/path/to/data/training_data.npz"
    MODELS_PATH = "/path/to/models/unet_model"
    FIGURES_PATH = "/path/to/figures/unet_figures"
    os.makedirs(MODELS_PATH, exist_ok=True)
    os.makedirs(FIGURES_PATH, exist_ok=True)
    
    # 0.9 | 0.1 split
    train_loader, val_loader = get_dataloaders(DATA_PATH, validation_split=0.1, batch_size=8)

    model = UNet(in_channels=1).to(device)
    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE)
    criterion = nn.L1Loss()
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", patience=5, factor=0.5
    )

    best_val_loss = float('inf')
    train_losses = []
    val_losses = []
    for epoch in range(EPOCHS):
        model.train()
        train_running_loss = 0.0
        for noisy, clean in tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS} - Training"):
            noisy, clean = noisy.to(device), clean.to(device)
            optimizer.zero_grad()
            output = model(noisy)
            loss = criterion(output, clean)
            loss.backward()
            optimizer.step()
            train_running_loss += loss.item() * noisy.size(0)
        train_loss = train_running_loss / len(train_loader.dataset)
        train_losses.append(train_loss)

        model.eval()
        val_running_loss = 0.0
        with torch.no_grad():
            for noisy, clean in tqdm(val_loader, desc=f"Epoch {epoch+1}/{EPOCHS} - Validation"):
                noisy, clean = noisy.to(device), clean.to(device)
                output = model(noisy)
                loss = criterion(output, clean)
                val_running_loss += loss.item() * noisy.size(0)
        val_loss = val_running_loss / len(val_loader.dataset)
        val_losses.append(val_loss)

        scheduler.step(val_loss)

        print(f"Epoch {epoch+1}/{EPOCHS}")
        print(f"Train Loss: {train_loss:.6f}")
        print(f"Valid Loss: {val_loss:.6f}")
        print("-" * 30)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_path = os.path.join(MODELS_PATH, 'weights_best.pth')
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': val_loss,
            }, best_path)
            print(f"Saved best weights to {best_path}")
            print("-" * 30)

    last_path = os.path.join(MODELS_PATH, 'weights_last.pth')
    torch.save(model.state_dict(), last_path)
    print(f"Saved last weights to {last_path}")
    print("-" * 30)

    epochs = list(range(1, EPOCHS + 1))
    plt.figure()
    plt.plot(epochs, train_losses, label='Training MAE')
    plt.plot(epochs, val_losses, label='Validation MAE')
    plt.xlabel('Epoch')
    plt.ylabel('Mean Absolute Error')
    plt.title('Training vs. Validation MAE')
    plt.legend()
    figure_path = os.path.join(FIGURES_PATH, 'mae_curve.png')
    plt.savefig(figure_path)
    print(f"Saved MAE curve to {figure_path}")
    print("-" * 30)

    # Save MAE arrays to .npz
    losses_path = os.path.join(FIGURES_PATH, 'mae_history.npz')
    np.savez(losses_path,
             mae=np.array(train_losses),
             val_mae=np.array(val_losses)
    )
    print(f"Saved MAE history to {losses_path}")
    print("-" * 30)

if __name__ == "__main__":
    main()
