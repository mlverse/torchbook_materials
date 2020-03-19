import torch
import torch.nn as nn
import torch.optim as optim
from torch.nn import functional as F
import numpy as np
import torchvision
from torchvision import transforms, datasets, utils
import os
import matplotlib.pyplot as plt


transform = transforms.Compose([
                               transforms.ToTensor(),
                           ])

kmnist = torchvision.datasets.KMNIST("data/kmnist", train=True, transform=transform, download=True)


dataloader = torch.utils.data.DataLoader(kmnist, batch_size=128,
                                         shuffle=True, num_workers=4)

# Decide which device we want to run on
device = torch.device("cuda:0" if torch.cuda.is_available()  else "cpu")

latent_dim <- 

class VAE(nn.Module):
    def __init__(self):
        super(VAE, self).__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size= 3, stride= 2, padding  = 1),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(),
            nn.Conv2d(32, 64, kernel_size= 3, stride= 2, padding  = 1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(),
            nn.Conv2d(64, 128, kernel_size= 3, stride= 2, padding  = 1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(),
            nn.Conv2d(128, 256, kernel_size= 3, stride= 2, padding  = 1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU()
        )
        self.latent_mu = nn.Linear(1024, latent_dim)
        self.latent_var = nn.Linear(1024, latent_dim)
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 1024)
            nn.Conv2dTranspose(1024, 256, kernel_size= 3, stride= 2, padding  = 1, output_padding = 1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(),
            nn.Conv2d(256, 128, kernel_size= 3, stride= 2, padding  = 1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(),
            nn.Conv2d(128, 64, kernel_size= 3, stride= 2, padding  = 1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(),
            nn.Conv2d(64, 32, kernel_size= 3, stride= 2, padding  = 1),
            nn.BatchNorm2d(32),
            nn.LeakyReLU()
        )



model = VAE().to(device)
optimizer = optim.Adam(model.parameters(), lr=1e-3)

# Reconstruction + KL divergence losses summed over all elements and batch
def loss_function(recon_x, x, mu, logvar):
    BCE = F.binary_cross_entropy(recon_x, x.view(-1, 784), reduction='sum')
    # see Appendix B from VAE paper:
    # Kingma and Welling. Auto-Encoding Variational Bayes. ICLR, 2014
    # https://arxiv.org/abs/1312.6114
    # 0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    return BCE + KLD

num_epochs = 1

for epoch in range(num_epochs):
    train_loss = 0
    for i, data in enumerate(dataloader, 0):
        data = data[0].to(device)
        optimizer.zero_grad()
        recon_batch, mu, logvar = model(data)
        loss = loss_function(recon_batch, data, mu, logvar)
        loss.backward()
        train_loss += loss.item()
        optimizer.step()
        if i % 50 == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, i * len(data), len(dataloader.dataset),
                100. * i / len(dataloader),
                loss.item() / len(data)))
            print('====> Epoch: {} Average loss: {:.4f}'.format(
                epoch, train_loss / len(dataloader.dataset)))
        # if i % 50 == 0:
        #    with torch.no_grad():
        #       fake = generator(fixed_noise).detach().cpu()
        #       img_list.append(utils.make_grid(fake, padding=2, normalize=True))


plt.figure(figsize=(10,5))
plt.title("Generator and Discriminator Loss During Training")
plt.plot(generator_losses,label="G")
plt.plot(discriminator_losses,label="D")
plt.xlabel("iterations")
plt.ylabel("Loss")
plt.legend()
plt.show()

fig = plt.figure(figsize=(8,8))
plt.axis("off")
img1 = img_list[0]
plt.imshow(np.transpose(img1,(1,2,0)))
plt.show()

# Grab a batch of real images from the dataloader
real_batch = next(iter(dataloader))

# Plot the real images
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
plt.imshow(np.transpose(img_list[-1],(1,2,0)))
plt.show()




