import os

import torch
import torchvision
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, random_split
from torchvision.datasets import MNIST

import matplotlib.pyplot as plt
import pytorch_lightning as pl


random_seed = 42
torch.manual_seed(random_seed)

BATCH_SIZE = 128
AVAIL_GPUS = min(1, torch.cuda.device_count())
NUM_WORKERS = int(os.cpu_count() / 2)

class MNISTDataModule(pl.LightningDataModule):
    def __init__(self, data_dir="./data", batch_size=BATCH_SIZE, num_workers=NUM_WORKERS):
        super().__init__()
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.num_workers = num_workers

        self.transform = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize((0.1307,), (0.3081,)),
            ]
        )

    def prepare_data(self):
        MNIST(self.data_dir, train=True, download=True)
        MNIST(self.data_dir, train=False, download=True)

    def setup(self, stage=None):
        # Assign train/val datasets
        if stage == "fit" or stage is None:
            mnist_full = MNIST(self.data_dir, train=True, transform=self.transform)
            self.mnist_train, self.mnist_val = random_split(mnist_full, [55000, 5000])

        # Assign test dataset
        if stage == "test" or stage is None:
            self.mnist_test = MNIST(self.data_dir, train=False, transform=self.transform)

    def train_dataloader(self):
        return DataLoader(self.mnist_train, batch_size=self.batch_size, num_workers=self.num_workers)

    def val_dataloader(self):
        return DataLoader(self.mnist_val, batch_size=self.batch_size, num_workers=self.num_workers)

    def test_dataloader(self):
        return DataLoader(self.mnist_test, batch_size=self.batch_size, num_workers=self.num_workers)

# Detective: fake or not fake -> 1 output [0, 1]
class Discriminator(nn.Module):
    def __init__(self) -> None:
        super().__init__()

        # Simple CNN
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.conv2_drop = nn.Dropout2d()  # Dropout layer
        self.fc1 = nn.Linear(320, 50)     # This layer have 320 inputs and 50 outputs 
        self.fc2 = nn.Linear(50, 1)       # Last layer with 50 inputs and one output (0, or 1)

    # Applying all layers
    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2)) # First convolutional layer with max pooling and RELU activation
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2)) # Same as first one put with Dropout

        # Flatten the tensor so it can be fed into the FC layers
        x = x.view(-1, 320)

        # Apply the FC layers
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = self.fc2(x)
        
        # Sigmoid will take care of the output so it will be between 0 and 1
        return torch.sigmoid(x)

# Generate Fake Data: output like-real data (it does the opposite of Descriminator)
class Generator(nn.Module):
    def __init__(self, latent_dim):
        super().__init__()
        self.lin1 = nn.Linear(latent_dim, 7 * 7 * 64)
        self.ct1 = nn.ConvTranspose2d(64, 32, 4, stride=2)
        self.ct2 = nn.ConvTranspose2d(32, 16, 4, stride=2)
        self.conv = nn.Conv2d(16, 1, kernel_size=7)

    def forward(self, x):
        # Pass latent space input into linear layer and reshape
        x = self.lin1(x)
        x = F.relu(x)
        x = x.view(-1, 64, 7, 7) 

        # Upsample (transposed conv) 16x16 (64 feature maps)
        x = self.ct1(x)
        x = F.relu(x)

        # Upsample to 34x34 (16 feature maps)
        x = self.ct2(x)
        x = F.relu(x)

        # Convolution to 28x28 (1 feature map)
        return self.conv(x)

# Here we put the Descriminator and Generator in one class "GAN"
class GAN(pl.LightningModule):
    def __init__(self, latent_dim=100, lr=0.0002):
        super().__init__()
        self.save_hyperparameters()

        self.generator = Generator(latent_dim=self.hparams.latent_dim)
        self.discriminator = Discriminator()

        # random nois
        self.validation_z = torch.randn(6, self.hparams.latent_dim)

    def forward(self, input_tensor):
        return self.generator(input_tensor)
    
    def adversarial_loss(self, y_hat, y):
        return F.binary_cross_entropy(y_hat, y)

    def training_step(self, batch, batch_idx, optimizer_idx):
        real_imgs, _ = batch # unpacking the batch

        z = torch.randn(real_imgs.shape[0], self.hparams.latent_dim) # sample noise data
        z = z.type_as(real_imgs) # move to GPU if availabe

        # Train Genertator: max log(D(G(z)))
        if optimizer_idx == 0:
            fake_imgs = self(z) # Execute the generator
            y_hat = self.discriminator(fake_imgs) # Running the Discriminator on generated images

            y = torch.ones(real_imgs.size(0), 1)
            y = y.type_as(real_imgs) # move to GPU if availabe

            generator_loss = self.adversarial_loss(y_hat, y) # Evaluating how good the gemerator is discriminator
            
            log_dict = {"generator_loss": generator_loss}
            return {"loss": generator_loss, "progress_bar": log_dict, "log": log_dict}


        # Train the Discriminator: max log(D(x)) + log(1 - D(G(z)))
        if optimizer_idx == 1:

          # How well can it label real images as real
          y_hat_real = self.discriminator(real_imgs)
          y_real = torch.ones(real_imgs.size(0), 1)
          y_real = y_real.type_as(real_imgs)

          real_loss = self.adversarial_loss(y_hat_real, y_real)

          # How well can it label generated images as fake
          y_hat_fake = self.discriminator(self(z).detach())
          y_fake = torch.zeros(real_imgs.size(0), 1)
          y_fake = y_fake.type_as(real_imgs)

          fake_loss = self.adversarial_loss(y_hat_fake, y_fake)

          discriminator_loss = (real_loss + fake_loss) / 2

          log_dict = {"discriminator_loss": discriminator_loss}
          return {"loss": discriminator_loss, "progress_bar": log_dict, "log": log_dict}

    def configure_optimizers(self):
        lr = self.hparams.lr

        # Creating two optimizers (One for the Generator, and one for the Discriminator)
        opt_generator = torch.optim.Adam(self.generator.parameters(), lr=lr)
        opt_discriminator = torch.optim.Adam(self.discriminator.parameters(), lr=lr)

        return [opt_generator, opt_discriminator], []

    def plot_imgs(self):
        z = self.validation_z.type_as(self.generator.lin1.weight) # 6 images with random noise
        sample_imgs = self(z).cpu() # This will execute the generator

        print('epoch', self.current_epoch)
        fig = plt.figure()

        # Plotting images as a grid
        for i in range(sample_imgs.size(0)):
            plt.subplot(2, 3, i + 1)
            plt.tight_layout()
            plt.imshow(sample_imgs.detach()[i, 0, :, :], cmap='gray_r', interpolation='none')
            plt.title("Generated Data")
            plt.xticks([])
            plt.yticks([])
            plt.axis('off')
        plt.show()
        
    # This function plots some fake (generated) images after each epoch
    def on_epoch_end(self):
        self.plot_imgs()

data_module = MNISTDataModule()
model = GAN()

model.plot_imgs() # Plotting generated images before training

trainer = pl.Trainer(max_epochs=30, gpus=AVAIL_GPUS)
trainer.fit(model, data_module)
