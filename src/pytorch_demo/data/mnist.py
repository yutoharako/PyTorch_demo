import os
import torch
from torch import optim, nn, utils, Tensor
from torchvision.datasets import MNIST
from torchvision.transforms import ToTensor
import lightning as L
from typing import Tuple, Any

# define any number of nn.Modules (or use your current ones)
encoder: nn.Sequential = nn.Sequential(
    nn.Linear(28 * 28, 64), nn.ReLU(), nn.Linear(64, 3)
)
decoder: nn.Sequential = nn.Sequential(
    nn.Linear(3, 64), nn.ReLU(), nn.Linear(64, 28 * 28)
)


# define the LightningModule
class LitAutoEncoder(L.LightningModule):
    def __init__(self, encoder: nn.Sequential, decoder: nn.Sequential) -> None:
        super().__init__()
        self.encoder: nn.Sequential = encoder
        self.decoder: nn.Sequential = decoder

    def training_step(self, batch: Tuple[Tensor, Any], batch_idx: int) -> Tensor:
        # training_step defines the train loop.
        # it is independent of forward
        x, _ = batch
        x = x.view(x.size(0), -1)
        z: Tensor = self.encoder(x)
        x_hat: Tensor = self.decoder(z)
        loss: Tensor = nn.functional.mse_loss(x_hat, x)
        # Logging to TensorBoard (if installed) by default
        self.log("train_loss", loss)
        return loss

    def configure_optimizers(self) -> optim.Adam:
        optimizer: optim.Adam = optim.Adam(self.parameters(), lr=1e-3)
        return optimizer


# init the autoencoder
autoencoder: LitAutoEncoder = LitAutoEncoder(encoder, decoder)

# setup data
dataset: MNIST = MNIST(os.getcwd(), download=True, transform=ToTensor())
train_loader: utils.data.DataLoader = utils.data.DataLoader(dataset)

# train the model (hint: here are some helpful Trainer arguments for rapid idea iteration)
trainer: L.Trainer = L.Trainer(limit_train_batches=100, max_epochs=1)
trainer.fit(model=autoencoder, train_dataloaders=train_loader)

# load checkpoint
checkpoint: str = "./lightning_logs/version_0/checkpoints/epoch=0-step=100.ckpt"
autoencoder = LitAutoEncoder.load_from_checkpoint(
    checkpoint, encoder=encoder, decoder=decoder
)

# choose your trained nn.Module
encoder = autoencoder.encoder
encoder.eval()

# embed 4 fake images!
fake_image_batch: Tensor = torch.rand(4, 28 * 28, device=autoencoder.device)
embeddings: Tensor = encoder(fake_image_batch)
print("⚡" * 20, "\nPredictions (4 image embeddings):\n", embeddings, "\n", "⚡" * 20)

# train on 4 GPUs
# trainer = L.Trainer(
#     devices=4,
#     accelerator="gpu",
# )

# train 1TB+ parameter models with Deepspeed/fsdp
# trainer = L.Trainer(
#     devices=4, accelerator="gpu", strategy="deepspeed_stage_2", precision=16
# )

# 20+ helpful flags for rapid idea iteration
trainer = L.Trainer(max_epochs=10, min_epochs=5, overfit_batches=1)  # type: L.Trainer

# access the latest state of the art techniques
# trainer = L.Trainer(callbacks=[WeightAveraging(...)])
