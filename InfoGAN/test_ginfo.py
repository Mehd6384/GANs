import torch 
import torch.nn as nn 
import torch.nn.functional as F 

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



g = Generator()
g.load_state_dict(torch.load('info_g'))

f, ax = plt.subplots()
count = 0
while True: 


	data = torch.zeros(32,10)
	data[:,count] = 1 
	count = (count +1)%10

	z = torch.randn(32,50)

	prod = g(z, data)

	grid = make_grid(prod.detach().numpy().reshape(32,1,28,28))

	ax.matshow(grid[:,:,0])
	ax.set_title(str(count))
	plt.pause(0.1)

	input()
	ax.clear()