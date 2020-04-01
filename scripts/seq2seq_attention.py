import torch
import torch.nn as nn
import torch.optim as optim
from torch.nn import functional as F
import numpy as np
import torchvision
from torchvision import transforms, datasets, utils
import os
import matplotlib.pyplot as plt

# https://github.com/AntixK/PyTorch-VAE

transform = transforms.Compose([
                               transforms.ToTensor(),
                           ])

kmnist = torchvision.datasets.KMNIST("data/kmnist", train=True, transform=transform, download=True)


dataloader = torch.utils.data.DataLoader(kmnist, batch_size=128,
                                         shuffle=True, num_workers=4)

device = torch.device("cuda:0" if torch.cuda.is_available()  else "cpu")

image_size = 28

class View(nn.Module):
    def __init__(self, shape):
        super(View, self).__init__()
        self.shape = shape
    def forward(self, x):
        return x.view(*self.shape)

class VAE(nn.Module):
    def __init__(self, latent_dim):
        super(VAE, self).__init__()
        self.latent_dim = latent_dim
        self.encoder = nn.Sequential(
            nn.Conv2d(1, image_size, kernel_size= 3, stride= 2, padding  = 1),
            nn.BatchNorm2d(image_size),
            nn.LeakyReLU(),
            nn.Conv2d(image_size, image_size * 2, kernel_size= 3, stride= 2, padding  = 1),
            nn.BatchNorm2d(image_size * 2),
            nn.LeakyReLU(),
            nn.Conv2d(image_size * 2, image_size * 4, kernel_size= 3, stride= 2, padding  = 1),
            nn.BatchNorm2d(image_size * 4),
            nn.LeakyReLU(),
            nn.Conv2d(image_size * 4, image_size * 8, kernel_size= 3, stride= 2, padding  = 1),
            nn.BatchNorm2d(image_size * 8),
            nn.LeakyReLU()
        )
        self.latent_mean = nn.Linear(896, latent_dim)
        self.latent_log_var = nn.Linear(896, latent_dim)
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, image_size * 8),
            View((-1, image_size * 8, 1, 1)),
            nn.ConvTranspose2d(image_size * 8, image_size * 4, kernel_size = 4, stride = 1, padding = 0, bias = False),
            nn.BatchNorm2d(image_size * 4),
            nn.LeakyReLU(),
            # 8 * 8
            nn.ConvTranspose2d(image_size * 4, image_size * 2, kernel_size = 4, stride = 2, padding = 1, bias = False),
            nn.BatchNorm2d(image_size * 2),
            nn.LeakyReLU(),
            # 16 x 16
            nn.ConvTranspose2d(image_size * 2, image_size, kernel_size = 4, stride = 2, padding = 2, bias = False),
            nn.BatchNorm2d(image_size),
            nn.LeakyReLU(),
            # 28 x 28
            nn.ConvTranspose2d(image_size, 1, kernel_size = 4, stride = 2, padding = 1, bias = False),
            nn.Sigmoid()
        )
    def encode(self, input): 
        result = self.encoder(input)
        result = torch.flatten(result, start_dim = 1)
        mean = self.latent_mean(result)
        log_var = self.latent_log_var(result)
        return [mean, log_var]
    def decode(self, z):
        result = self.decoder(z)
        return result
    def reparameterize(self, mean, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return eps * std + mean
    def forward(self, input):
        mean, log_var = self.encode(input)
        z = self.reparameterize(mean, log_var)
        return [self.decode(z), input, mean, log_var]
    def loss_function(self, reconstruction, input, mean, log_var):
        reconstruction_loss = F.binary_cross_entropy(reconstruction, input, reduction='sum')
        kl_loss = -0.5 * torch.sum(1 + log_var - mean ** 2 - log_var.exp())
        loss = reconstruction_loss + kl_loss
        return loss, reconstruction_loss, kl_loss
    def sample(self, num_samples, current_device):
        z = torch.randn(num_samples, self.latent_dim)
        z = z.to(current_device)
        samples = self.decode(z)
        return samples

model = VAE(latent_dim = 2).to(device)
optimizer = optim.Adam(model.parameters(), lr = 1e-3)

num_epochs = 5

img_list  = []

for epoch in range(num_epochs):
    for i, data in enumerate(dataloader, 0):
        data = data[0].to(device)
        optimizer.zero_grad()
        reconstruction, input, mean, log_var = model(data)
        loss, reconstruction_loss, kl_loss = model.loss_function(reconstruction, input, mean, log_var)
        loss.backward()
        optimizer.step()
        # Output training stats
        if i % 50 == 0:
            print('[%d/%d][%d/%d]\tLoss: %.2f\tReconstruction: %.2f\tKL: %.2f'
                  % (epoch, num_epochs, i, len(dataloader),
                     loss.item(), reconstruction_loss.item(), kl_loss.item()))
            with torch.no_grad():
              generated = model.sample(64, device)
              img_list.append(utils.make_grid(generated, padding=2, normalize=True))


fig = plt.figure(figsize=(8,8))
plt.axis("off")
img1 = img_list[-1].cpu()
plt.imshow(np.transpose(img1,(1,2,0)))
plt.show()


real_batch = next(iter(dataloader))
plt.figure(figsize=(15,15))
plt.subplot(1,2,1)
plt.axis("off")
plt.title("Real Images")
plt.imshow(np.transpose(utils.make_grid(real_batch[0].to(device)[:64], padding=5, normalize=True).cpu(),(1,2,0)))
plt.show()
# Plot the fake images from the last epoch
plt.subplot(1,2,2)
plt.axis("off")
plt.title("Fake Images")
plt.imshow(np.transpose(img_list[-1].cpu(),(1,2,0)))
plt.show()




