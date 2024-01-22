import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset
import numpy as np
import cv2
import glob
import os
from sklearn.model_selection import train_test_split
import torch.optim as optim
import torch
import typer
from torch.utils.data import DataLoader
import tqdm
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score


class HomoDataset(Dataset):
    def __init__(self, input_paths, labels_list, image_size, batch_size):
        self.input_paths = input_paths
        self.labels_list = labels_list
        self.batch_size = batch_size
        self.image_size = image_size
        self.indices = np.arange(len(labels_list))

    def _read_homography(self, file_path: str):
        return np.load(file_path).reshape((9,))[:-1]

    def __len__(self):
        return len(self.labels_list)

    def __getitem__(self, index):
        image = (
            cv2.resize(
                cv2.cvtColor(cv2.imread(self.input_paths[index]), cv2.COLOR_BGR2RGB),
                self.image_size,
            )
            / 255.0
        )

        image = np.reshape(image, (3, 540, 540))

        y = self._read_homography(self.labels_list[index])

        return torch.tensor(image), torch.tensor(np.array(y))


class HomographyNet(nn.Module):
    def __init__(self, input_channels, output_dim):
        super(HomographyNet, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(input_channels, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        self.layer2 = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        self.layer3 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        self.layer4 = nn.Sequential(
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
        )
        self.avgpool = nn.AdaptiveAvgPool2d((16, 16))
        self.fc = nn.Sequential(
            nn.Dropout(0.1),
            nn.Linear(16 * 16 * 3, 1024),
            nn.ReLU(True),
            nn.Dropout(0.1),
            nn.Linear(1024, output_dim),
        )

    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        x = self.avgpool(x)
        out = torch.flatten(x, 1)
        out = self.fc(out)
        return out


def main():
    LR = 0.001
    BATCH_SIZE = 4

    DATA_DIR = (
        "/home/fer/Escritorio/futstatistics/datasets/narya/homography_dataset/dataset"
    )

    train_x = glob.glob(os.path.join(DATA_DIR, "train_img") + "/*.jpg")
    train_y = glob.glob(os.path.join(DATA_DIR, "train_homo") + "/*.npy")

    val_x = glob.glob(os.path.join(DATA_DIR, "test_img") + "/*.jpg")
    val_y = glob.glob(os.path.join(DATA_DIR, "test_homo") + "/*.npy")

    # resplit train and val
    x = train_x + val_x
    y = train_y + val_y

    total_iteration = 90000
    num_samples = len(x)
    steps_per_epoch = num_samples // BATCH_SIZE
    EPOCHS = int(total_iteration / steps_per_epoch)

    # use split train test from sklearn
    train_x, val_x, train_y, val_y = train_test_split(
        x, y, test_size=0.15, shuffle=True
    )

    print(f"{len(train_x)} training images found")
    assert len(train_x) == len(train_y)

    print(f"{len(val_x)} validation images found")
    assert len(val_x) == len(val_y)

    train_dataset = HomoDataset(
        train_x,
        train_y,
        image_size=(540, 540),
        batch_size=BATCH_SIZE,
    )

    test_dataset = HomoDataset(
        val_x,
        val_y,
        image_size=(540, 540),
        batch_size=BATCH_SIZE,
    )

    train_loader = DataLoader(
        dataset=train_dataset, batch_size=BATCH_SIZE, shuffle=True
    )
    test_loader = DataLoader(dataset=test_dataset, batch_size=BATCH_SIZE, shuffle=True)

    model = HomographyNet(3, 8)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=LR)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.1, patience=4, verbose=True
    )

    if torch.cuda.is_available():
        model = model.cuda()

    print("Training!....")
    best_loss = np.inf

    for epoch in range(EPOCHS):
        model.train()
        running_loss = 0.0

        print(
            f"Epoch [{epoch + 1}/{EPOCHS}],",
            f"Learning rate: {optimizer.param_groups[0]['lr']}",
        )

        for batch_idx, (inputs, targets) in tqdm.tqdm(
            enumerate(train_loader),
            total=len(train_loader),
            desc="Training...",
            colour="green",
        ):
            inputs, targets = inputs.to("cuda", dtype=torch.float), targets.to(
                "cuda", dtype=torch.float
            )

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        print(f"Train Loss: {running_loss/len(train_loader):.6f}")

        model.eval()
        val_loss = 0.0
        predicted_values = []
        true_values = []
        with torch.no_grad():
            for val_inputs, val_targets in tqdm.tqdm(
                test_loader,
                total=len(test_loader),
                desc="Evaluating...",
                colour="blue",
            ):
                val_inputs, val_targets = val_inputs.to(
                    "cuda", dtype=torch.float
                ), val_targets.to("cuda", dtype=torch.float)

                outputs = model(val_inputs)
                val_loss += criterion(outputs, val_targets).item()
                # For evaluation metrics calculation
                predicted_values.extend(outputs.cpu().numpy())
                true_values.extend(val_targets.cpu().numpy())

        curr_val_loss = val_loss / len(test_loader)
        print(f"Validation Loss: {curr_val_loss:.6f}")

        # Calculate Evaluation Metrics for Regression
        mse = mean_squared_error(true_values, predicted_values)
        mae = mean_absolute_error(true_values, predicted_values)
        r2 = r2_score(true_values, predicted_values)
        print(f"MSE: {mse:.4f}, MAE: {mae:.4f}, R²: {r2:.4f}")

        if curr_val_loss < best_loss:
            print(
                f"Best model checkpoint found! Went from {best_loss:.6f} to {curr_val_loss:.6f}"
            )
            best_loss = curr_val_loss
            best_model_weights = model.state_dict()
            model.load_state_dict(best_model_weights)
            torch.save(model.state_dict(), "best_model_homographynet.pth")

        print("--------------------------------------------------------")

        scheduler.step(curr_val_loss)

    print("Training finished")


if __name__ == "__main__":
    typer.run(main)
