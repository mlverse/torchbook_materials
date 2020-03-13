import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import torchvision
from torchvision import transforms, datasets, utils
import os
import matplotlib.pyplot as plt

# Size of z latent vector (i.e. size of generator input)
latent_input_size = 100

# Size of feature maps in generator
image_size = 28

# Number of training epochs
num_epochs = 5

# Learning rate for optimizers
lr = 0.0002

# Beta1 hyperparam for Adam optimizers
beta1 = 0.5


transform = transforms.Compose([
                               transforms.ToTensor(),
                           ])

kmnist = torchvision.datasets.KMNIST("data/kmnist", train=True, transform=transform, download=True)


dataloader = torch.utils.data.DataLoader(kmnist, batch_size=128,
                                         shuffle=True, num_workers=4)

# Decide which device we want to run on
device = torch.device("cuda:0" if torch.cuda.is_available()  else "cpu")

# Plot some training images
real_batch = next(iter(dataloader))
real_batch[0].shape
plt.figure(figsize=(8,8))
plt.axis("off")
plt.title("Training Images")
plt.imshow(np.transpose(utils.make_grid(real_batch[0].to(device)[:64], padding=2, normalize=True).cpu(),(1,2,0)))
plt.show()

def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)
        
# Generator Code

class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
        self.main = nn.Sequential(
            # torch.nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride=1, padding=0, output_padding=0, groups=1, bias=True, dilation=1, padding_mode='zeros')
            # h_out = (h_in - 1) * stride - 2 * padding + dilation * (kernel_size - 1) + output_padding + 1
            # (1 - 1) * 1 - 2 * 0 + 1 * (4 -1 ) + 0 + 1
            # 4 x 4
            nn.ConvTranspose2d(latent_input_size, image_size * 4, 4, 1, 0, bias=False),
            nn.BatchNorm2d(image_size * 4),
            nn.ReLU(True),
            # 8 * 8
            nn.ConvTranspose2d(image_size * 4, image_size * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(image_size * 2),
            nn.ReLU(True),
            # 16 x 16
            nn.ConvTranspose2d(image_size * 2, image_size, 4, 2, 2, bias=False),
            nn.BatchNorm2d(image_size),
            nn.ReLU(True),
            # 28 x 28
            nn.ConvTranspose2d(image_size, 1, 4, 2, 1, bias=False),
            nn.Tanh()
        )
    def forward(self, input):
        return self.main(input)

generator = Generator().to(device)

# Apply the weights_init function to randomly initialize all weights
#  to mean=0, stdev=0.2.
generator.apply(weights_init)

class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.main = nn.Sequential(
            # 14 x 14
            nn.Conv2d(1, image_size, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            # 7 x 7
            nn.Conv2d(image_size, image_size * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(image_size * 2),
            nn.LeakyReLU(0.2, inplace=True),
            # 3 x 3
            nn.Conv2d(image_size * 2, image_size * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(image_size * 4),
            nn.LeakyReLU(0.2, inplace=True),
            # 1 x 1
            nn.Conv2d(image_size * 4, 1, 4, 2, 1, bias=False),
            nn.Sigmoid()
        )
    def forward(self, input):
        return self.main(input)


discriminator = Discriminator().to(device)

# Apply the weights_init function to randomly initialize all weights
#  to mean=0, stdev=0.2.
discriminator.apply(weights_init)


# Initialize BCELoss function
criterion = nn.BCELoss()

# Create batch of latent vectors that we will use to visualize
#  the progression of the generator
fixed_noise = torch.randn(64, latent_input_size, 1, 1, device=device)

# Establish convention for real and fake labels during training
real_label = 1
fake_label = 0

# Setup Adam optimizers for both G and D
disc_optimizer = optim.Adam(discriminator.parameters(), lr=lr, betas=(beta1, 0.999))
gen_optimizer = optim.Adam(generator.parameters(), lr=lr, betas=(beta1, 0.999))

# Training Loop

# Lists to keep track of progress
img_list = []
generator_losses = []
discriminator_losses = []

print("Starting Training Loop...")
# For each epoch
for epoch in range(num_epochs):
    # For each batch in the dataloader
    for i, data in enumerate(dataloader, 0):
        ############################
        # (1) Update D network: maximize log(D(x)) + log(1 - D(G(z)))
        ###########################
        ## Train with all-real batch
        discriminator.zero_grad()
        # Format batch
        real_cpu = data[0].to(device)
        b_size = real_cpu.size(0)
        label = torch.full((b_size,), real_label, device=device)
        # Forward pass real batch through D
        output = discriminator(real_cpu).view(-1)
        # Calculate loss on all-real batch
        discriminator_loss_real = criterion(output, label)
        # Calculate gradients for D in backward pass
        discriminator_loss_real.backward()
        discriminator_real_verdict = output.mean().item()
        ## Train with all-fake batch
        # Generate batch of latent vectors
        noise = torch.randn(b_size, latent_input_size, 1, 1, device=device)
        # Generate fake image batch with G
        fake = generator(noise)
        label.fill_(fake_label)
        # Classify all fake batch with D
        output = discriminator(fake.detach()).view(-1)
        # Calculate D's loss on the all-fake batch
        discriminator_loss_fake = criterion(output, label)
        # Calculate the gradients for this batch
        discriminator_loss_fake.backward()
        # Add the gradients from the all-real and all-fake batches
        discriminator_loss = discriminator_loss_real + discriminator_loss_fake
        # Update D
        disc_optimizer.step()
        ############################
        # (2) Update G network: maximize log(D(G(z)))
        ###########################
        generator.zero_grad()
        label.fill_(real_label)  # fake labels are real for generator cost
        # Since we just updated D, perform another forward pass of all-fake batch through D
        output = discriminator(fake).view(-1)
        # Calculate G's loss based on this output
        generator_loss = criterion(output, label)
        # Calculate gradients for G
        generator_loss.backward()
        discriminator_fake_verdict = output.mean().item()
        # Update G
        gen_optimizer.step()
        # Output training stats
        if i % 50 == 0:
            print('[%d/%d][%d/%d]\tLoss_D: %.4f\tLoss_G: %.4f\tD(x): %.4f\tD(G(z)): %.4f '
                  % (epoch, num_epochs, i, len(dataloader),
                     discriminator_loss.item(), generator_loss.item(), discriminator_real_verdict, discriminator_fake_verdict))
        # Save Losses for plotting later
        generator_losses.append(generator_loss.item())
        discriminator_losses.append(discriminator_loss.item())
        # Check how the generator is doing by saving G's output on fixed_noise
        with torch.no_grad():
          fake = generator(fixed_noise).detach().cpu()
          img_list.append(utils.make_grid(fake, padding=2, normalize=True))

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
ims = [[plt.imshow(np.transpose(i,(1,2,0)))] for i in img_list]
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




