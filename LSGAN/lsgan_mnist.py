import torch 
import torch.nn as nn 
import torch.nn.functional as F 
import torch.optim as optim 

import pickle
import numpy as np 
import matplotlib.pyplot as plt 
plt.style.use('ggplot')

from torchvision.utils import save_image
from tensorboardX import SummaryWriter 
from PIL import Image 


def make_grid(grid_data, rows = 4): 

	if isinstance(grid_data, torch.Tensor): 
		# print('Is a tensor')
		grid_data = grid_data.detach().numpy()

	im_size = [grid_data.shape[2], grid_data.shape[3]]
	nb_im = grid_data.shape[0]

	col = int(nb_im/rows)

	x = np.zeros((grid_data.shape[1], int(im_size[0]*4), int(im_size[1]*col)))

	counter = 0

	for i in range(rows): 
		for j in range(col): 

			x[:,i*im_size[0]:(i+1)*im_size[0],j*im_size[1]:(j+1)*im_size[1]] = grid_data[counter,:,:,:]
			counter += 1


	if grid_data.shape[1] == 1: 
		x = np.stack([x for i in range(3)], 1)
		x = np.squeeze(x,0)

	return np.transpose(x, [1,2,0]) 

def update(opt, l): 

	opt.zero_grad()
	l.backward()
	opt.step()

class D(nn.Module): 

	def __init__(self) :

		nn.Module.__init__(self)

		self.l1 = nn.Linear(784,400)
		self.l2 = nn.Linear(400,128)
		self.l3 = nn.Linear(128,1)

	def forward(self, x): 

		x = F.leaky_relu(self.l1(x),0.2)
		x = F.leaky_relu(self.l2(x),0.2)
		pred = self.l3(x)

		return pred 

class G(nn.Module): 

	def __init__(self, noise_size) :

		nn.Module.__init__(self)

		self.l1 = nn.Linear(noise_size,128)
		self.l2 = nn.Linear(128,400)
		self.l3 = nn.Linear(400,784)

	def forward(self, x): 

		x = F.leaky_relu(self.l1(x),0.2)
		x = F.leaky_relu(self.l2(x),0.2)
		# pred = F.sigmoid(self.l3(x),0.2)
		pred = self.l3(x)

		return pred 

class Loader: 

	def __init__(self, path, max_points): 

		self.path = path 
		self.max_points = max_points

	def sample(self, batch): 

		data = np.zeros((batch,784))
		index = np.random.randint(0,self.max_points,(batch))
		for i,ind in enumerate(index): 
			im = pickle.load(open(self.path+'{}'.format(ind), 'rb')) 
			data[i,:] = im.reshape(-1)

		data /= 255.
		data = (data - 0.5)/0.5
		return torch.tensor(data).float()


def normal_init(model, mean, std): 

	for m in model._modules: 
		if isinstance(model._modules[m], nn.Linear): 
			model._modules[m].weight.data.normal_(mean, std)
			model._modules[m].bias.data.zero_()

# def get_images_from_images(image): 

# 	# save_image(prod.reshape(64,1,size,size), '{}.png'.format(e))
# 	prod = image.reshape(64,1,28,28).expand_as(torch.zeros(64,3,28,28))
# 	return make_grid(prod)

# def get_images(model): 

# 	prod = model(torch.randn(64,noise_size))
# 	size = int(np.sqrt(prod.shape[1]))
# 	# save_image(prod.reshape(64,1,size,size), '{}.png'.format(e))
# 	prod = prod.reshape(64,1,size,size).expand_as(torch.zeros(64,3,size,size))
# 	return make_grid(prod)

# def get_images_loader(loader): 

# 	im = loader.sample(64) 
# 	size = int(np.sqrt(im.shape[1]))
# 	# save_image(prod.reshape(64,1,size,size), '{}.png'.format(e))
# 	im = im.reshape(64,1,size,size).expand_as(torch.zeros(64,3,size,size))
# 	return make_grid(im)

noise_size = 10 
epochs = 10000
batch_size = 128

g = G(noise_size)
d = D()

normal_init(d,0.,0.1)
normal_init(g,0.,0.1)

loader =Loader('/home/mehdi/Codes/MNIST/', 60000)
path_to_result = '/home/mehdi/Codes/ML2/Generative Modeling/GAN/LSGAN/results/'

opt_g = optim.Adam(g.parameters(), 2e-4)
opt_d = optim.Adam(d.parameters(), 2e-4)

mean_d, mean_g = 0.,0.
recap_d, recap_g = [],[]

first = True
writer =SummaryWriter('run')


for epoch in range(1,epochs+1): 

	nb_times_gen = np.random.randint(1,3)
	nb_times_dis = np.random.randint(1,3)

	for i in range(nb_times_dis): 
		z = torch.rand(batch_size, noise_size)

		# if np.random.random() < 0.05: 
		# 	fakes = loader.sample(batch_size)
		# 	x = g(z).detach()
		# else:
		# 	fakes = g(z).detach() 
		# 	x = loader.sample(batch_size)

		fakes = g(z).detach()
		x = loader.sample(batch_size)

		real_label = np.random.uniform(0.8,1.2)
		fake_label = np.random.uniform(0.0,0.3)

		d_loss_real = 0.5*torch.mean((d(x) - real_label).pow(2))
		d_loss_fake = 0.5*torch.mean((d(fakes) - fake_label)**2)
		update(opt_d,d_loss_fake)
		update(opt_d,d_loss_real)

	for i in range(nb_times_gen): 
		z = z = torch.rand(batch_size, noise_size)
		fakes = g(z)

		g_loss = 0.5*torch.mean((d(fakes)-1.).pow(2))
		update(opt_g,g_loss)

	writer.add_scalars('Losses/Relative', {'D':d_loss_real.item() + d_loss_fake.item(), 'G':g_loss.item()}, epoch)


	mean_g += g_loss
	mean_d += (d_loss_real + d_loss_fake)


	if epoch%400 == 0: 
		# recap_g.append(mean_g/300.)
		# recap_d.append(mean_d/300.)
		print('\n\n\t\t === Epoch: {} ===\nLoss d: {}\nLoss g: {}'.format(epoch, mean_d/300, mean_g/300))
		# plt.cla()
		# plt.plot(recap_g, label = 'Loss generator')
		# plt.plot(recap_d, label = 'Loss discriminator')
		# plt.legend()
		# plt.pause(0.1)
		z = torch.rand(64,10)
		images = g(z)
		save_image(images.reshape(64,1,28,28), 'test/{}.png'.format(epoch))

		images = images.reshape(64,1,28,28)
		im = make_grid(images)
		# im_fakes = get_images_from_images(images)
		im_real = make_grid(loader.sample(64).reshape(64,1,28,28))
		current = np.array(Image.open('test/{}.png'.format(epoch)))

		writer.add_image('Data/Fakes',current, epoch)
		# writer.add_image('Data/Fakes', im, epoch)
		# writer.add_image('Data/Fakes2', im_fakes, epoch)
		if first: 
			first = False
			save_image(loader.sample(64).reshape(64,1,28,28), 'test/real.png')
			current = np.array(Image.open('test/real.png'))
			writer.add_image('Data/Real',current , epoch)


		mean_g = 0. 
		mean_d = 0.
