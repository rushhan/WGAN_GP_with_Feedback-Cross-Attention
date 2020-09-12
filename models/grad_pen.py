import torch
import torch.nn as nn
import torch.nn.parallel
import torch.autograd as autograd
class grad_pen:
	"""docstring for grad_pen"""
	def __init__(self,lambd,batch_size,cuda):
		super(grad_pen, self).__init__()
		self.batch_size = batch_size
		self.cuda = cuda
		self.lambd = lambd
	
	def loss(self,disc,real_data,gen_data):
		alpha = torch.rand(self.batch_size,1)
		alpha = alpha.expand(real_data.size())
		if self.cuda:
			alpha = alpha.cuda()
		interpolates = alpha*real_data+ (1-alpha)*gen_data
		if self.cuda:
			interpolates = interpolates.cuda()

		interpolates = autograd.Variable(interpolates,requires_grad=True)

		disc_interpolates,_= disc(interpolates)

		outs = torch.ones(disc_interpolates.size())
		if self.cuda:
			outs = outs.cuda()
		gradient = autograd.grad(outputs = disc_interpolates,inputs= interpolates,
								grad_outputs=outs,create_graph = True,
								retain_graph = True, only_inputs= True)[0]

		gradient_penalty = ((gradient.norm(2,dim=1)-1)**2).mean()* self.lambd

		return gradient_penalty