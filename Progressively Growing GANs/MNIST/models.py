import torch 
import torch.nn as nn 
import torch.nn.functional as F 
import numpy as np 

class G(nn.Module): 

	def __init__(self, noise_size, start_size): 

		nn.Module.__init__(self)
		self.layers = nn.ModuleList()

		self.last_layer_size = start_size
		self.layers.append(nn.Linear(noise_size,self.last_layer_size))

	def add_layer(self, mean = 0., std = 0.02):

		l = nn.Linear(self.last_layer_size, self.last_layer_size*4)
		l.weight.data.normal_(mean, std)
		l.bias.data.zero_()


		self.layers.append(l)
		self.last_layer_size *= 4

	def alpha_forward(self, x): 

		nb_layers = len(self.layers)
		for i in range(nb_layers-1): 
			if i == nb_layers-2: 
				x = self.layers[i](x)
			else:
				x = F.leaky_relu(self.layers[i](x),0.02)
		return x 

	def forward(self, x): 

		for l in self.layers:

			if l == self.layers[-1]: 
				x = l(x)
			else: 
				x = F.leaky_relu(l(x), 0.02)
		return x 

	@property
	def last_layer(self):
		return self.last_layer_size

	@property
	def last_layer_q(self):
		return int(np.sqrt(self.last_layer_size))
	


class D(nn.Module): 

	def __init__(self, start_size): 

		nn.Module.__init__(self)

		self.layers = nn.ModuleList()

		self.last_layer_size = start_size 
		self.layers.append(nn.Linear(self.last_layer_size, 1))

	def add_layer(self, mean = 0., std = 0.02):

		l = nn.Linear(self.last_layer_size*4, self.last_layer_size)
		l.weight.data.normal_(mean, std)
		l.bias.data.zero_()


		self.layers.append(l)
		self.last_layer_size *= 4 

	def forward(self,x): 

		for l in reversed(self.layers): 
			if l == self.layers[0]: 
				# print('last layer, using lin')
				x = l(x)
			else: 
				# print('using non lin')
				x = F.leaky_relu(l(x), 0.02)
			# x = F.sigmoid(l(x))

		return x 

	def alpha_forward(self, x): 

		nb_l = len(self.layers)

		for i in reversed(range(nb_l-1)): 
			if i == 0: 
				# print('last layer, using lin')
				x = self.layers[i](x)
			else: 
				# print('using non lin')
				x = F.leaky_relu(self.layers[i](x), 0.02)

			# x = F.sigmoid(self.layers[i](x))

		return x 

# g = G(10,20)
# g.add_layer()
# g.add_layer()

# g(torch.rand(1,10))
# g.alpha_forward(torch.rand(1,10))

# d = D(10)
# d.add_layer()
# d.add_layer()

# d.alpha_forward(torch.rand(1,20))
# d(torch.rand(1,40))
