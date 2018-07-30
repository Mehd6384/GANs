import torch 
import torch.nn as nn 
import torch.nn.functional as F 
import torch.optim as optim 

from torchvision.utils import save_image 

import pickle 
import numpy as np 
import matplotlib.pyplot as plt 
plt.style.use('dark_background')

def make_grid(x, rows = 4, decal = 2): 

	# x is a (batch_size, nb_channels, height, width) nd-array

	col = int(x.shape[0]/rows)
	image = np.zeros(((x.shape[2] + decal)*rows, (decal + x.shape[3])*col, 3))
	ligne = 0 
	column = 0 
	for i in range(x.shape[0]): 
		current = x[i,:,:,:]
		current = np.transpose(current, [1,2,0])

		
		image[decal + ligne*(x.shape[2]):(ligne+1)*(x.shape[2]) + decal, decal + column*(x.shape[3]):(column+1)*(x.shape[3]) + decal,:] = current

		column = (column + 1)%col
		if(column == 0): 
			ligne += 1

	# input(image.shape)
	return image 


def read_data(path): 

	return pickle.load(open(path, 'rb'))

def normal_init(model, mean = 0., std = 0.02):

	for m in model._modules: 

		if isinstance(model._modules[m], nn.Linear): 
			model._modules[m].weight.data.normal_(mean, std)
			model._modules[m].bias.data.zero_()

		elif isinstance(model._modules[m], nn.ModuleList): 
			for i in range(len(model._modules[m])): 

				model._modules[m][i].weight.data.normal_(mean, std)
				model._modules[m][i].bias.data.zero_()


class Loader: 

	def __init__(self, path, max_el): 

		self.path = path
		self.max_el = max_el

	def sample(self, batch_size): 

		x_ = np.zeros((batch_size, 784))
		y_ = np.zeros((batch_size, 10))

		inds = np.random.randint(0,self.max_el, (batch_size))

		for i in range(batch_size): 

			x = read_data(self.path + str(inds[i]))
			label = read_data(self.path + 'labels')[inds[i]]

			x_[i,:] = x
			y_[i,label] = 1 


		x_ /= 255.
		x_ = (x_ - 0.5)/0.5

		return torch.tensor(x_).float(),torch.tensor(y_).float()


noise_size = 50 
class Generator(nn.Module): 

	def __init__(self): 

		nn.Module.__init__(self)

		self.z = nn.Linear(50,64)
		self.cond = nn.Linear(10,64)

		self.layers = nn.ModuleList([nn.Linear(128,256), 
									 nn.Linear(256,512)
									])

		self.head = nn.Linear(512,784)

	def forward(self, x,y): 

		z = F.relu(self.z(x))
		cond = F.relu(self.cond(y))

		latent = torch.cat([z, cond], 1)

		for l in self.layers: 
			latent = F.relu(l(latent))

		return self.head(latent)

class Discriminator(nn.Module): 

	def __init__(self): 

		nn.Module.__init__(self)

		self.obs = nn.Linear(784,400)
		self.cond = nn.Linear(10,100)

		self.layers = nn.ModuleList([nn.Linear(500,256), 
									 nn.Linear(256,128)
									])

		self.head = nn.Linear(128,1)

	def forward(self, x,y): 

		z = F.relu(self.obs(x))
		cond = F.relu(self.cond(y))

		latent = torch.cat([z, cond], 1)

		for l in self.layers: 
			latent = F.relu(l(latent))

		return self.head(latent)



loader = Loader('/home/mehdi/Codes/MNIST/', 60000)

g = Generator()
d = Discriminator()

normal_init(g)
normal_init(d)

adam_d = optim.Adam(d.parameters(), 1e-4)
adam_g = optim.Adam(g.parameters(), 1e-4)

epochs = 10000
batch_size = 64

recap_g = []
recap_d = []

f,ax = plt.subplots(2,1)

for epoch in range(1,epochs+1): 

	# train discriminator 
	
	for i in range(np.random.randint(1,3)): 
		x,y = loader.sample(batch_size)


		z = torch.randn(batch_size, noise_size)

		fakes = g(z,y)

		d_fakes = d(fakes, y)
		d_real = d(x,y)

		soft_labels_real = torch.ones((batch_size, 1)).uniform_(0.8,1.2)
		soft_labels_fake = torch.ones((batch_size, 1)).uniform_(0.0,0.25)


		d_loss_fake = torch.mean(torch.pow(soft_labels_fake - d_fakes, 2))
		d_loss_real = torch.mean(torch.pow(soft_labels_real - d_real, 2))

		d_loss = d_loss_fake + d_loss_real

		adam_d.zero_grad()
		d_loss.backward()
		adam_d.step()


	# train generator 
	for i in range(np.random.randint(1,4)): 

		z = torch.randn(batch_size, noise_size)
		fakes = g(z,y)
		d_fakes = d(fakes, y)

		soft_labels_real = torch.ones((batch_size, 1)).uniform_(0.8,1.2)
		g_loss = torch.mean(torch.pow(d_fakes - soft_labels_real,2))

		adam_g.zero_grad()
		g_loss.backward()
		adam_g.step()


	recap_g.append(g_loss.item())
	recap_d.append(d_loss.item())

	if epoch%20 == 0: 
		print('\n \t\t\t === Epoch {} ===\nLoss D: {} \nLoss G: {}'.format(epoch, np.mean(recap_d[-20:]), np.mean(recap_g[-20:])))

	if epoch% 25 == 0: 
		for a in ax: 
			a.clear()
		ax[0].plot(recap_g, label = 'Generator')
		ax[0].plot(recap_d, label = 'Discriminator')

		ax[0].legend()

		x,y = loader.sample(32)

		y = torch.zeros_like(y)
		for i in range(32): 
			y[i,i%y.shape[1]] = 1 

		noise = torch.randn(32, noise_size)
		prod = g(noise, y)

		grid = make_grid(prod.detach().numpy().reshape(32,1,28,28))
		ax[1].matshow(grid[:,:,0], cmap=plt.cm.gray_r)


		plt.pause(0.1)

torch.save(g.state_dict(), 'g_conditional_lsgan')
torch.save(d.state_dict(), 'd_conditional_lsgan')

