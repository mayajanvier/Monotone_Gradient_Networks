import torch.nn as nn
import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt

class CMGN(nn.Module):
    def __init__(self, input_dim, output_dim, num_layers, hidden_dim):
        super(CMGN, self).__init__()
        self.num_layers = num_layers
        self.sigma_layers = nn.ModuleList([nn.Sigmoid() for _ in range(num_layers)])
        self.biases = nn.ParameterList([nn.Parameter(torch.randn(hidden_dim)) for _ in range(num_layers)])
        self.W = nn.Parameter(torch.randn(input_dim, hidden_dim)) # Share the weight matrix W across all layers
        self.V = nn.Parameter(torch.randn(hidden_dim, input_dim))
        self.bL = nn.Parameter(torch.randn(output_dim))

    def forward(self, x):

        Wx = F.linear(x, self.W)
        z = Wx + self.biases[0] 
        for i in range(1,self.num_layers):
            z = Wx + self.sigma_layers[i-1](z) + self.biases[i]

        VTx = F.linear(x, self.V.t())
        VTVx = F.linear(VTx, self.V)
        Wsigma = F.linear(self.sigma_layers[self.num_layers-1](z), self.W.T)

        output = Wsigma + VTVx + self.bL
        return output
    

class MMGN(nn.Module):
    def __init__(self, input_dim, output_dim, num_modules, hidden_dim):
        super(MMGN, self).__init__()
        self.num_modules = num_modules
        self.W_list = nn.ParameterList([nn.Parameter(torch.randn(input_dim, hidden_dim)) for _ in range(num_modules)])
        self.biases = nn.ParameterList([nn.Parameter(torch.randn(hidden_dim)) for _ in range(num_modules)])
        self.activation = nn.Tanh()

        self.V = nn.Parameter(torch.randn(hidden_dim, input_dim))
        self.a = nn.Parameter(torch.randn(output_dim))

    def forward(self, x):
        def log_cosh(x):
          return torch.sum( torch.log(torch.cosh(x)) )

        sum = 0
        for i in range(self.num_modules):
            Wx = F.linear(x, self.W_list[i].T)
            z = Wx + self.biases[i]
            Wsigma = F.linear(self.activation(z), self.W_list[i])
            sum += log_cosh(z) * Wsigma
        VTx = F.linear(x, self.V)
        VTVx = F.linear(VTx, self.V.t())

        output = self.a + VTVx + sum 
        return output
    
#models the gradient by integrating the jacobian
#justOrtho = nn.Linear
class ICGN(nn.Module):
    def __init__(self,in_dim,hidden,layers=1,ortho=1,**kwargs,):

        super(ICGN,self).__init__()

        self.hidden = hidden
        self.in_dim = in_dim
        self.out_dim = in_dim


        self.lin = nn.ModuleList([
                nn.Linear(in_dim,hidden,bias=True),
                *[nn.Linear(hidden,hidden) for _ in range(layers)],
        ])

        self.act = nn.Sigmoid()#lambda x: nn.Softplus()(x) - np.log(2)

    def jvp(self, x,v):
        with torch.enable_grad():
            #computes w = V(x)v
            w = torch.autograd.functional.jvp(self.forward_, inputs=x, v=v, create_graph=True)[1]
            #compute w = V.T(x)w
            w = torch.autograd.functional.vjp(self.forward_,inputs=x,v=w, create_graph=True)[1]
        return w


    def forward_(self,x):
        a = self.act
        y = a(self.lin[0](x))

        for lay in self.lin[1:-1]:
            y = a(lay(y))

        return self.lin[-1](y) if len(self.lin) > 1 else y

        # for lay in self.lin[1:][::-1]:
        #     y = a(F.linear(y,lay.weight.data.T))


        # return F.linear(y,self.lin[0].weight.data.T)

    def checkjac(self,simp=False,N=100):
        x = torch.rand(1,self.in_dim)
        print("computing jacobian using autograd on forward")
        if self.in_dim < 10:
            if simp:
                M = torch.autograd.functional.jacobian(self.forward_simp,inputs=x,vectorize=True).squeeze()
            else:
                M = torch.autograd.functional.jacobian(lambda x: self.forward(x,N=N),inputs=x,vectorize=True).squeeze()
            print(M)

        print("computing jacobian using autograd on forward_")
        if self.in_dim < 10:
            V = torch.autograd.functional.jacobian(self.forward_,inputs=x,vectorize=True).squeeze()
            V = V.T@V
            print(V)

        print("checking PSD")
        M.detach()
        with torch.no_grad():
            for _ in range(100):
                v = torch.rand(self.in_dim)
                if (v*M@v).sum() < -1e-2:
                    print("Failed PSD test", (v*M@v).sum(), "vector", v, "at point", x)
                    return
        print("passed psd test")
        return

    #this evaluates the PDE given in the paper, averaging over N randomly chosen points
    # the expected result (if the hidden network satisfies the PDE) is 0
    def checkPDE(self):
        pass


    #numerically computes the line integral
    def forward(self,x,N=100,**kwargs):
        pts = torch.rand(size=(N,)).reshape(1,-1,1).to(x.device)

        in_size = x.shape[0]
        in_dim = (self.in_dim,)
        out_dim = (self.out_dim,)

        z = x.unsqueeze(1) * pts
        v = x.unsqueeze(1) * torch.ones_like(pts)

        z = z.reshape(-1,*in_dim)
        v = v.reshape(-1,*in_dim)


        #this computes the integral
        y = self.jvp(z,v)
        y = y.reshape(in_size,-1,*out_dim).sum(dim=1) / N

        return y

    #uses simpsons rule for computing integral, more accurate but problematic because eval points are fixed
    #can't be used for learning because the model learns to avoid those points and do weird things
    #however, is very accurate at inference time
    def forward_simp(self,x,w=None,**kwargs):
        pts = torch.tensor([0,1/3,2/3,1]).reshape(1,-1,1).to(x.device)


        scale = torch.tensor([1,3,3,1]).reshape(1,-1,1).to(x.device)

        in_size = x.shape[0]
        in_dim = (self.in_dim,)
        out_dim = (self.out_dim,)

        z = x.unsqueeze(1) * pts
        if w is None: v = x.unsqueeze(1) * torch.ones_like(pts) #probably a better way to do this
        else: v = w.unsqueeze(1) * torch.ones_like(pts)
        #print(z.shape,v.shape)
        #not necessary for linear layers, but will be for convolutions so good practice
        z = z.reshape(-1,*in_dim)
        v = v.reshape(-1,*in_dim)

        #this computes the integral
        y = self.jvp(z,v)
        #print("after jvp", y.reshape(in_size,-1,*out_dim).shape)
        y = 1/8 * (y.reshape(in_size,-1,*out_dim)*scale).sum(dim=1)

        #print("output", y.shape)
        return y