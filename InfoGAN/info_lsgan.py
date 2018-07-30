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
nb_classes = 10 


def sample_c(batch_size): 

	classes = np.random.multinomial(1, nb_classes*[1./nb_classes], size = batch_size)
	return torch.tensor(classes).float()


class Q(nn.Module): 

	def __init__(self): 

		nn.Module.__init__(self)

		self.layers = nn.ModuleList([nn.Linear(784,384),
									 nn.Linear(384,128)
									]) 

		self.head = nn.Linear(128,nb_classes)

	def forward(self, x): 

		for l in self.layers: 
			x = F.relu(l(x))

		return F.softmax(self.head(x), dim = 1)

class Generator(nn.Module): 

	def __init__(self): 

		nn.Module.__init__(self)

		# self.l1 = nn.Linear(noise_size,64)
		# self.l1b = nn.Linear(nb_classes, 32)

		self.layers = nn.ModuleList([nn.Linear(60, 128), 
									 nn.Linear(128, 384),
									 nn.Linear(384, 512)
									]) 

		self.head = nn.Linear(512,784)

	def forward(self, z, c): 


		# z_b = F.relu(self.l1(z))
		# c_b = F.relu(self.l1b(c))

		x = torch.cat([z, c], 1)

		for l in self.layers: 
			x = F.relu(l(x))

		return self.head(x)

class Discriminator(nn.Module): 

	def __init__(self): 

		nn.Module.__init__(self)

		self.layers = nn.ModuleList([nn.Linear(784,512),
									 nn.Linear(512,384), 
									 nn.Linear(384,128)
									]) 

		self.head = nn.Linear(128,1)

	def forward(self, x): 

		for l in self.layers: 
			x = F.relu(l(x))

		return self.head(x)


def update(opt, l): 

	opt.zero_grad()
	l.backward()
	opt.step()


q = Q()
g = Generator()
d = Discriminator()

loader = Loader('/home/mehdi/Codes/MNIST/', 60000)

for model in [g,d,q]: 
	normal_init(model)


adam_g = optim.Adam(g.parameters(), 2e-4)
adam_d = optim.Adam(d.parameters(), 2e-4)
adam_q = optim.Adam(q.parameters(), 2e-4)


epochs = 10000
batch_size = 32

recap_d = []
recap_g = []
recap_m = []

f, ax = plt.subplots(2,1)


for epoch in range(epochs) :

	# training D 

	z = torch.randn(batch_size, noise_size)
	latent = sample_c(batch_size)

	fakes = g(z,latent)
	reals = loader.sample(batch_size)

	d_real = d(reals)
	d_fake = d(fakes)

	soft_labels_real = torch.ones((batch_size, 1)).uniform_(0.8,1.2)
	soft_labels_fake = torch.ones((batch_size, 1)).uniform_(0.0,0.25)

	d_loss_fake = torch.mean(torch.pow(soft_labels_fake - d_fake, 2))
	d_loss_real = torch.mean(torch.pow(soft_labels_real - d_real, 2))

	d_loss = d_loss_fake + d_loss_real

	update(adam_d, d_loss)

	# training g

	for i in range(np.random.randint(1,3)): 

		z = torch.randn(batch_size, noise_size)
		latent = sample_c(batch_size)

		fakes = g(z, latent)

		soft_labels_real = torch.ones((batch_size, 1)).uniform_(0.8,1.2)

		g_loss = torch.mean(torch.pow(d(fakes) - soft_labels_real,2))

		q_given_x = q(fakes)


		cond_e = torch.mean(-torch.sum(torch.log(q_given_x + 1e-10)*latent, 1))
		e = torch.mean(-torch.sum(torch.log(latent + 1e-10)*latent, 1))
		mutual_info_loss = cond_e + e
		full_loss = mutual_info_loss + g_loss

		adam_g.zero_grad()
		adam_q.zero_grad()
		full_loss.backward()
		adam_g.step()
		adam_q.step()


	recap_g.append(g_loss.item())
	recap_d.append(d_loss.item())
	recap_m.append(mutual_info_loss.item())

	if epoch%50 == 0: 
		print('\n \t\t\t === Epoch {} ===\nLoss D: {:.6f} \nLoss G: {:.6f}\nMutual info loss: {:.6f}'.format(epoch, 
			np.mean(recap_d[-20:]), np.mean(recap_g[-20:]), np.mean(recap_m[-20:])))

	if epoch% 25 == 0: 
		for a in ax: 
			a.clear()
		ax[0].plot(recap_g, label = 'Generator')
		ax[0].plot(recap_d, label = 'Discriminator')
		ax[0].plot(recap_m, label = 'Mutual Info')

		ax[0].legend()

		x = loader.sample(batch_size)

		z = torch.randn(batch_size, noise_size)
		latent = sample_c(batch_size)
		
		prod = g(z, latent)

		grid = make_grid(prod.detach().numpy().reshape(batch_size,1,28,28))
		ax[1].matshow(grid[:,:,0])


		plt.pause(0.1)

torch.save(g.state_dict(), 'info_g')
torch.save(d.state_dict(), 'info_d')

plt.show()

