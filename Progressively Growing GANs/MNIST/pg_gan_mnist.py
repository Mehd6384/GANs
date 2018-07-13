import torch 
import torch.nn.functional as F 
import torch.nn as nn
import torch.optim as optim

import pickle
import numpy as np 
import matplotlib.pyplot as plt 
from PIL import Image 

from progressively_growing_gan import G,D
from tensorboardX import SummaryWriter
from torchvision.utils import make_grid, save_image 
import os 

def count_param(opt): 

	params = opt.param_groups[0]['params']
	somme = 0 
	for a in params: 
		c = 1 
		for s in a.shape: 
			c *= s
		somme += c
	return somme 

def write_image(grid_data, writer, name = '', epoch = 0): 

	save_image(grid_data, 'current_image.png')
	im =np.array(Image.open('current_image.png'))

	writer.add_image(name, im, epoch)
	os.remove('current_image.png')


class Loader: 

	def __init__(self, path, max_points): 

		self.path = path 
		self.max_points = max_points

	def sample(self, batch, divisions = 1): 

		size = int(784/divisions)

		data = np.zeros((batch,size))
		index = np.random.randint(0,self.max_points,(batch))
		for i,ind in enumerate(index): 
			im = pickle.load(open(self.path+'{}'.format(ind), 'rb')) 
			if divisions == 1: 
				data[i,:] = im.reshape(-1)
			else: 
				data[i,:] = im[0::divisions].reshape(-1)
			# data[i,:] = im.reshape(-1)



		data /= 255.
		data = (data - 0.5)/0.5
		return torch.tensor(data).float()

def run_d(d, g, x, z, penultimate = False): 

	real_label = np.random.uniform(0.8,1.2)
	fake_label = np.random.uniform(0.0,0.2)

	if penultimate: 
		d_real = d.alpha_forward(x)
		d_fake = d.alpha_forward(g.alpha_forward(z))
	else: 
		d_real = d(x)
		d_fake = d(g(z))

	d_loss = 0.5*torch.mean((real_label - d_real).pow(2) + (fake_label - d_fake).pow(2))

	return d_loss

def run_g(d,g,z, penultimate = False): 

	if penultimate: 
		g_loss = 0.5*torch.mean((1. - d.alpha_forward(g.alpha_forward(z))).pow(2))
	else: 
		g_loss = 0.5*torch.mean((1. - d(g(z))).pow(2))
	return g_loss

def update(opt, l): 

	opt.zero_grad()
	l.backward()
	opt.step()

def normal_init(model, mean = 0., std = 0.02): 

	for m in model._modules: 
		if isinstance(model._modules[m], nn.Linear): 
			model._modules[m].weight.data.normal_(mean, std)
			model._modules[m].bias.data.zero_()



possible_divisions = [16,4,1]
path = '/home/mehdi/Codes/MNIST/'

loader = Loader(path, 60000)

noise_size = 50
start_size = 49 # 7x7

def get_images(model): 

	prod = model(torch.randn(64,noise_size))
	size = int(np.sqrt(prod.shape[1]))
	# save_image(prod.reshape(64,1,size,size), '{}.png'.format(e))
	prod = prod.reshape(64,1,size,size).expand_as(torch.zeros(64,3,size,size))
	return make_grid(prod)
def get_images_loader(loader, divisions): 

	im = loader.sample(64,divisions) 
	size = int(np.sqrt(im.shape[1]))
	# save_image(prod.reshape(64,1,size,size), '{}.png'.format(e))
	im = im.reshape(64,1,size,size).expand_as(torch.zeros(64,3,size,size))
	return make_grid(im)

g = G(noise_size, start_size)
d = D(start_size)

ad = optim.Adam(g.parameters())


normal_init(g)
normal_init(d)

def set_adam(model): 

	adam = optim.Adam(model.parameters(), 2e-4)
	return adam 

def add_layer(): 
	g.add_layer()
	d.add_layer()

	return set_adam(g), set_adam(d)


adam_g = set_adam(g)
adam_d = set_adam(d)

epochs = 100
batch_size = 128

beta = 1. 
alpha = 0. 

beta_factor = 0.1
alpha_factor = 0.001

steps_per_size = [50.,100.,250.]
it_beta = int(1./beta_factor)



layer_counter = 0
max_layer = len(possible_divisions)

writer = SummaryWriter('run')
writer_counter = 0 
alpha_writer = 0
first = True

for epoch in range(1,epochs+1): 

	writer.add_scalar('Parameters/D', count_param(adam_d), epoch)
	writer.add_scalar('Parameters/G', count_param(adam_g), epoch)
	beta_factor = 1/steps_per_size[layer_counter]

	# print('Entering beta phase ')
	beta = 1.
	beta_loss_g = 0. 
	beta_loss_d = 0. 
	while beta > 0.: 

		beta -= beta_factor
		for i in range(np.random.randint(1,3)): 	

			x = loader.sample(batch_size, divisions = possible_divisions[layer_counter])
			z = torch.randn(batch_size, noise_size)

			loss_d = run_d(d,g,x,z, penultimate = False)

			update(adam_d, loss_d)

		# x = loader.sample(batch_size, divisions = possible_divisions[layer_counter])
		for i in range(np.random.randint(1,3)): 
			z = torch.randn(batch_size, noise_size)

			loss_g = run_g(d,g,z, penultimate = False)
			update(adam_g, loss_g)

		
		beta_loss_g += loss_g.item()
		beta_loss_d += loss_d.item()

		
		# print('Beta: {:.3f} - x_sample size: {}'.format(beta, x.size()))
		# writer.add_scalar('Losses/G', loss_g.item(), writer_counter)
		# writer.add_scalar('Losses/D', loss_d.item(), writer_counter)
		writer.add_scalars('Losses/Phases', {'Alpha':alpha, 'Beta':beta}, writer_counter)
		writer.add_scalars('Losses/Relative', {'D':loss_d.item(), 'G':loss_g.item()}, writer_counter)
		writer_counter += 1 

	im = get_images(g)
	im_real = get_images_loader(loader, possible_divisions[layer_counter])

	prod = g(torch.rand(64, noise_size)).reshape(64,1,g.last_layer_q,g.last_layer_q)
	write_image(prod, writer, 'Prod/Fakes', epoch)


	if first: 
		first = False 
		real = loader.sample(64).reshape(64,1,28,28)
		write_image(real, writer, 'Prod/Real', 0)

	# writer.add_image('Data/Fakes', im, epoch)
	# writer.add_image('Data/Real', im_real, epoch)

	print('\n\n \t\t\t === Epoch : {} ====\nLoss D: {:.6f}\nLoss G: {:.6f}'.format(epoch, beta_loss_d/it_beta, beta_loss_g/it_beta))

	if(layer_counter < max_layer-1 and epoch%30 == 0): 
		# print('Entering alpha phase ')
		alpha = 1.
		layer_counter += 1
		adam_g, adam_d = add_layer()

		while alpha > 0.: 

			alpha -= alpha_factor

			# Discriminator 

			x_big = loader.sample(batch_size, divisions = possible_divisions[layer_counter])
			z = torch.randn(batch_size, noise_size)
			loss_d_big = run_d(d,g,x_big,z,penultimate = False)

			x_small = loader.sample(batch_size, divisions = possible_divisions[layer_counter-1])
			z = torch.randn(batch_size, noise_size)
			loss_d_small = run_d(d,g,x_small,z,penultimate = True)

			loss_d = loss_d_small*alpha + (1.-alpha)*loss_d_big

			update(adam_d, loss_d)

			# Generator 

			z = torch.randn(batch_size, noise_size)
			loss_g_big = run_g(d,g,z,penultimate = False)

			z = torch.randn(batch_size, noise_size)
			loss_g_small = run_g(d,g,z,penultimate = True)

			loss_g = loss_g_small*alpha + (1.-alpha)*loss_g_big

			update(adam_g, loss_g)


			# writer.add_scalar('Losses/G', loss_g.item(), writer_counter)
			# writer.add_scalar('Losses/D', loss_d.item(), writer_counter)
			writer.add_scalars('Losses/Phases', {'Alpha':alpha, 'Beta':beta}, writer_counter)
			writer.add_scalars('Losses/Relative', {'D':loss_d.item(), 'G':loss_g.item()}, writer_counter)
			writer_counter += 1 

			# print('Alpha: {:.3f} - x big size: {} x small size: {}'.format(alpha, x_big.size(), x_small.size()))

	# input('round')

# print('\n\n\t\t\t === Final model shape ===\nG: {}\nD: {}'.format(g,d))



