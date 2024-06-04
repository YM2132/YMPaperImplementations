import torch
import torchvision

import torch.nn as nn

import torchvision.transforms as transforms

from torch.utils.data import TensorDataset, ConcatDataset, random_split, DataLoader, Dataset

import numpy as np
import matplotlib.pyplot as plt
def imshow(img):
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()

# Check if MPS is available, if not check if CUDA is available
if torch.backends.mps.is_available():
	device = torch.device("mps")
	x = torch.ones(1, device=device)
	print(x)

elif torch.backends.cuda.is_built():
	device = torch.device("cuda")
	x = torch.ones(1, device=device)
	print(x)

else:
	print("MPS device not found.")

# Load the MNIST dataset
batch_size = 32
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,)),
])

mnist_train = torchvision.datasets.MNIST(
    root='./Data',
    train=True,
    download=True,
    transform=transform,
)
mnist_train.targets = torch.ones_like(mnist_train.targets, dtype=torch.float32)

train_loader = torch.utils.data.DataLoader(
    mnist_train,
    shuffle=True,
    batch_size=batch_size,
)

mnist_test = torchvision.datasets.MNIST(
    root='./Data',
    train=False,
    download=True,
    transform=transform
)
mnist_test.targets = torch.ones_like(mnist_train.targets, dtype=torch.float32)

test_loader = torch.utils.data.DataLoader(
    mnist_test,
    shuffle=True,
    batch_size=batch_size,
)

# Define the discriminator model
class Discriminator(nn.Module):
	def __init__(self):
		super().__init__()

		self.linear1 = nn.Linear(784, 1024)
		self.relu1 = nn.ReLU()
		self.dropout1 = nn.Dropout(0.3)

		self.linear2 = nn.Linear(1024, 512)
		self.relu2 = nn.ReLU()
		self.dropout2 = nn.Dropout(0.3)

		self.linear3 = nn.Linear(512, 256)
		self.relu3 = nn.ReLU()
		self.dropout3 = nn.Dropout(0.3)

		self.linear4 = nn.Linear(256, 1)

		# Use sigmoid to ensure output is a probability
		self.sigmoid = nn.Sigmoid()

	def forward(self, x):
		x = x.view(x.size(0), 784)

		x = self.linear1(x)
		x = self.relu1(x)
		x = self.dropout1(x)

		x = self.linear2(x)
		x = self.relu2(x)
		x = self.dropout2(x)

		x = self.linear3(x)
		x = self.relu3(x)
		x = self.dropout3(x)

		x = self.linear4(x)

		x = self.sigmoid(x)

		return x


discriminator = Discriminator().to(device)

# Define the generator model
class Generator(nn.Module):
	def __init__(self):
		super().__init__()

		self.linear1 = nn.Linear(100, 256)
		self.relu1 = nn.ReLU()

		self.linear2 = nn.Linear(256, 512)
		self.relu2 = nn.ReLU()

		self.linear3 = nn.Linear(512, 1024)
		self.relu3 = nn.ReLU()

		self.linear4 = nn.Linear(1024, 784)
		self.tanh = nn.Tanh()

	def forward(self, x):
		x = self.linear1(x)
		x = self.relu1(x)

		x = self.linear2(x)
		x = self.relu2(x)

		x = self.linear3(x)
		x = self.relu3(x)

		x = self.linear4(x)

		out = self.tanh(x)

		# Reshape the output to a 28x28 matrix
		out = out.view(x.size(0), 1, 28, 28)

		return out


generator = Generator().to(device)

# Init two optimizers, one for G and one for D
criterion = nn.BCELoss()

optimizer_D = torch.optim.Adam(discriminator.parameters(), lr=0.0001)
optimizer_G = torch.optim.Adam(generator.parameters(), lr=0.0001)

# Train the GAN
for epoch in range(50):
	for i, data in enumerate(train_loader):
		real_images, _ = data  #  We dont care about the MNIST labels we can just generate all 1s to sim them
		real_images = real_images.to(device)

		# Sample from noise and generate the fake images
		noise_tensor = torch.randn((batch_size, 100)).to(device)  #  Increase amount of noise it was 1 before
		with torch.no_grad():
			gen_images = generator(noise_tensor)

		# Create the real and fake labels
		gen_labels = torch.zeros((batch_size, 1)).to(device)
		real_labels = torch.ones((batch_size, 1)).to(device)

		# Concat fake and real images
		combined_images = torch.cat((real_images, gen_images))  # Change from [inp, gen] to ()
		combined_labels = torch.cat((real_labels, gen_labels))

		# Optional: shuffle the combined batch to prevent the model from learning order
		indices = torch.randperm(combined_images.size(0))
		combined_images = combined_images[indices]
		combined_labels = combined_labels[indices]

		#  First update the D model
		discriminator.zero_grad()
		d_outputs_combined = discriminator(combined_images)
		loss_d = criterion(d_outputs_combined, combined_labels)
		loss_d.backward()
		optimizer_D.step()

		# Generate new images for updating G
		noise_tensor = torch.randn((batch_size, 100)).to(device)

		# Next update the G model,
		generator.zero_grad()
		gen_images = generator(noise_tensor)  #  Gen new images for training G
		# For generator we need to switch the label from fake 0s, to real 1s
		# Note we use the D model, the equation in the paper is max log(D(G(z))) and we already have G(z)
		d_outputs_generated = discriminator(gen_images)
		loss_g = criterion(d_outputs_generated, real_labels)
		loss_g.backward()
		optimizer_G.step()

		if i == batch_size - 1:
			print(f'Epoch {epoch}: Loss_D: {loss_d.item()}, Loss_G: {loss_g.item()}')

	imshow(torchvision.utils.make_grid(gen_images.cpu()))

print("Training complete")
