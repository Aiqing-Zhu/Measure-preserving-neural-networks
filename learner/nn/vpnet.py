import torch
import torch.nn as nn

from .module import Module, StructureNN
class GradientModule(Module):
    '''Gradient volume preserving module.
    '''
    def __init__(self, dim, h=0.01, mode=0, s=1, width=10, activation='sigmoid'):
        super(GradientModule, self).__init__()
        self.dim = dim
        self.h=h
        self.width = width
        self.activation = activation
        self.mode = mode
        self.s = s
        
        self.params = self.__init_params()
        
    def forward(self, x):
        i = int(self.mode)
        p = x[...,i:i+self.s] 
        q = torch.cat([x[...,:i], x[...,i+self.s:]],-1)
        subVP = self.act(q @ self.params['K1'] + self.params['b']) @ self.params['K2']
        return torch.cat([x[...,:i], p + subVP, x[...,i+self.s:]], -1)
    
    def __init_params(self):
        d = int(self.dim)
        params = nn.ParameterDict()
        params['K1'] = nn.Parameter((torch.randn([d-self.s, self.width]) ).requires_grad_(True))
        params['b'] = nn.Parameter((torch.randn([self.width]) ).requires_grad_(True))
        params['K2'] = nn.Parameter((torch.randn([self.width, self.s])* self.h).requires_grad_(True))
        return params

class VPNet(StructureNN):
    '''Volume preserving network composed by GradientModule.
    '''
    def __init__(self, dim, h=0.1, s=2, layers=5, sublayers=3, width=30, activation='sigmoid'):
        super(VPNet, self).__init__()
        self.dim = dim
        self.h = h
        self.s = s
        self.layers = layers
        self.sublayers = sublayers
        self.width = width
        self.activation = activation
        
        self.modus = self.__init_modules()
        
    def forward(self, x):
        for i in range(self.layers):
            for j in range(2):
                subVP = self.modus['sub{}{}'.format(i+1,j+1)]
                x = subVP(x)
        return x
    
    def __init_modules(self):
        modules = nn.ModuleDict()
        for i in range(self.layers):
                modules['sub{}{}'.format(i+1,1)] = GradientModule(
                    dim=self.dim, h=self.h, mode=0, s =self.s, width=self.width, activation=self.activation)
                modules['sub{}{}'.format(i+1,2)] = GradientModule(
                    dim=self.dim, h=self.h, mode=self.s, s =self.dim-self.s, width=self.width, activation=self.activation)
        return modules
  
    def predict(self, xh, steps=1, keepinitx=False, returnnp=False):
        dim = xh.size(-1)
        size = len(xh.size())
        if dim == self.dim:
            pred = [xh]
            for _ in range(steps):
                pred.append(self(pred[-1]))
        else:
            x0, h = xh[..., :-1], xh[..., -1:] 
            pred = [x0]
            for _ in range(steps):
                pred.append(self(torch.cat([pred[-1], h], dim=-1)))
        if keepinitx:
            steps = steps + 1
        else:
            pred = pred[1:]
        res = torch.cat(pred, dim=-1).view([-1, steps, self.dim][2 - size:])
        return res.cpu().detach().numpy() if returnnp else res
