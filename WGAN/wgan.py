import torch 
import torch.nn as nn 
import torch.nn.functional as F 
import torch.optim as optim 

import numpy as np 
import matplotlib.pyplot as plt 
import pickle 

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

def normal_init(models, mean = 0., std = 0.02):

	for model in models: 
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
		# y_ = np.zeros((batch_size, 10))

		inds = np.random.randint(0,self.max_el, (batch_size))

		for i in range(batch_size): 

			x = read_data(self.path + str(inds[i]))
			# label = read_data(self.path + 'labels')[inds[i]]

			x_[i,:] = x
			# y_[i,label] = 1 


		x_ /= 255.
		x_ = (x_ - 0.5)/0.5

		return torch.tensor(x_).float() #,torch.tensor(y_).float()

noise_size = 50


class Generator(nn.Module): 

	def __init__(self): 

		nn.Module.__init__(self)

		self.l1 = nn.Linear(noise_size, 128)
		self.l2 = nn.Linear(128,384)
		self.l3 = nn.Linear(384,512)
		self.head = nn.Linear(512,784)


	def forward(self, z): 

		layers = [self.l1, self.l2, self.l3]

		for l in layers: 
			z = F.relu(l(z))

		return self.head(z)

class Discriminator(nn.Module): 

	def __init__(self): 

		nn.Module.__init__(self)

		self.l1 = nn.Linear(784, 512)
		self.l2 = nn.Linear(512,384)
		self.l3 = nn.Linear(384,128)
		self.head = nn.Linear(128,1)


	def forward(self, z): 

		layers = [self.l1, self.l2, self.l3]

		for l in layers: 
			z = F.relu(l(z))

		return self.head(z)
	

d = Discriminator()
g = Generator()

normal_init([d,g], 0.,0.02)

loader = Loader('/home/mehdi/Codes/MNIST/', 60000)
epochs = 10000
batch_size = 64

optim_g = optim.RMSprop(g.parameters(), 1e-4)
optim_d = optim.RMSprop(d.parameters(), 1e-4)

recap_g = []
recap_d = []

_, ax = plt.subplots(2,1)

for epoch in range(epochs): 

	# train D 
	for i in range(np.random.randint(1,5)): 

		x = loader.sample(batch_size)
		z = torch.randn(batch_size, noise_size)

		fakes = g(z)
		d_real = d(x)
		d_fake = d(fakes)

		d_loss = -(torch.mean(d_real) - torch.mean(d_fake))
		optim_d.zero_grad()
		d_loss.backward()
		optim_d.step()

		for p in d.parameters(): 
			p.data.clamp_(-0.01,0.01)


	# train G 

	z = torch.randn(batch_size, noise_size)
	fakes = g(z)

	d_fake = d(fakes)
	g_loss = -torch.mean(d_fake)
	optim_g.zero_grad()
	g_loss.backward()
	optim_g.step()

	recap_g.append(g_loss.item())
	recap_d.append(d_loss.item())

	if epoch%50 == 0: 
		print('\n \t\t\t === Epoch {} ===\nLoss D: {:.6f} \nLoss G: {:.6f}\n'.format(epoch, 
			np.mean(recap_d[-20:]), np.mean(recap_g[-20:])))

	if epoch% 25 == 0: 
		for a in ax: 
			a.clear()
		ax[0].plot(recap_g, label = 'Generator')
		ax[0].plot(recap_d, label = 'Discriminator')

		ax[0].legend()

		x = loader.sample(batch_size)

		z = torch.randn(batch_size, noise_size)
		
		prod = g(z)

		grid = make_grid(prod.detach().numpy().reshape(batch_size,1,28,28))
		ax[1].matshow(grid[:,:,0])


		plt.pause(0.1)

torch.save(g.state_dict(), 'w_g')
torch.save(d.state_dict(), 'w_d')

plt.show()