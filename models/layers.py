import torch
import torch.nn as nn
import torch.nn.functional as F

class TensorNormalization(nn.Module):
    def __init__(self,mean, std):
        super(TensorNormalization, self).__init__()
        if not isinstance(mean, torch.Tensor):
            mean = torch.tensor(mean)
        if not isinstance(std, torch.Tensor):
            std = torch.tensor(std)
        self.mean = mean
        self.std = std
    def forward(self,X):
        return normalizex(X,self.mean,self.std)
    def extra_repr(self) -> str:
        return 'mean=%s, std=%s'%(self.mean, self.std)

def normalizex(tensor, mean, std):
    mean = mean[None, :, None, None]
    std = std[None, :, None, None]
    if mean.device != tensor.device:
        mean = mean.to(tensor.device)
        std = std.to(tensor.device)
    return tensor.sub(mean).div(std)

class MergeTemporalDim(nn.Module):
    def __init__(self, T):
        super().__init__()
        self.T = T

    def forward(self, x_seq: torch.Tensor):
        return x_seq.flatten(0, 1).contiguous()

class ExpandTemporalDim(nn.Module):
    def __init__(self, T):
        super().__init__()
        self.T = T

    def forward(self, x_seq: torch.Tensor):
        y_shape = [self.T, int(x_seq.shape[0]/self.T)]
        y_shape.extend(x_seq.shape[1:])
        return x_seq.view(y_shape)

class ZIF(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, gama):
        out = (input >= 0).float()
        L = torch.tensor([gama])
        ctx.save_for_backward(input, out, L)
        return out

    @staticmethod
    def backward(ctx, grad_output):
        (input, out, others) = ctx.saved_tensors
        gama = others[0].item()
        grad_input = grad_output
        tmp = (1 / gama) * (1 / gama) * ((gama - input.abs()).clamp(min=0))
        grad_input = grad_input * tmp
        return grad_input, None

class RateBp(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):
        mem = 0.
        spike_pot = []
        T = x.shape[0]
        for t in range(T):
            mem = mem + x[t, ...]
            spike = ((mem - 1.) > 0).float()
            mem = (1 - spike) * mem
            spike_pot.append(spike)
        out = torch.stack(spike_pot, dim=0)
        ctx.save_for_backward(out)
        return out
    
    @staticmethod
    def backward(ctx, grad_output):
        out, = ctx.saved_tensors
        T = out.shape[0]
        out = out.mean(0).unsqueeze(0)
        grad_input = grad_output * (out > 0).float()
        return grad_input
class SgNormal(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input_, gama):
        L = torch.tensor([gama])
        ctx.save_for_backward(input_,L)
        out = (input_ > 0).float()
        return out


    @staticmethod
    def backward(ctx, grad_output):
        (input_,others) = ctx.saved_tensors
        gama = others[0].item()
        grad_input = grad_output.clone()
        grad = grad_input*torch.exp(-(input_**2)/(2*(gama**2)))/(gama*torch.sqrt(2*torch.tensor(torch.pi)))
        return grad, None
class LIFSpike(nn.Module):
    def __init__(self, T, thresh=1.0, tau=1., gama=1.0):
        super(LIFSpike, self).__init__()
        self.act = ZIF.apply
        self.thresh = thresh
        self.tau = tau
        self.gama = gama
        self.expand = ExpandTemporalDim(T)
        self.merge = MergeTemporalDim(T)
        self.relu = nn.ReLU(inplace=True)
        self.ratebp = RateBp.apply
        self.mode = 'bptt'
        self.T = T

    def forward(self, x):
        if self.mode == 'bptr' and self.T > 0:
            x = self.expand(x)
            x = self.ratebp(x)
            x = self.merge(x)
        elif self.T > 0:
            x = self.expand(x)
            mem = 0
            spike_pot = []
            for t in range(self.T):
                mem = mem * self.tau + x[t, ...]
                spike = self.act(mem - self.thresh, self.gama)
                mem = (1 - spike) * mem
                spike_pot.append(spike)
            x = torch.stack(spike_pot, dim=0)
            x = self.merge(x)
        else:
            x = self.relu(x)
        return x
class LIFSpikeIN(nn.Module):
    def __init__(self, T=1, thresh=1.0, tau=1., gama=0.5):
        super(LIFSpikeIN, self).__init__()
        self.act = SgNormal.apply
        self.thresh = thresh
        self.tau = tau
        self.gama = gama
        self.expand = ExpandTemporalDim(T)
        self.merge = MergeTemporalDim(T)
        self.relu = nn.ReLU(inplace=True)
        self.ratebp = RateBp.apply
        self.mode = 'bptt'
        self.T = T

    def forward(self, x):
        # print(self.T)
        if self.mode == 'bptr' and self.T > 0:
            x = self.expand(x)
            x = self.ratebp(x)
            x = self.merge(x)
        elif self.T > 0:
            # x = self.expand(x)
            mem = 0
            spike_pot = []
            for t in range(self.T):
                mem = mem * self.tau + x[t, ...]
                spike = self.act(mem - self.thresh, self.gama)
                mem = (1 - spike) * mem
                spike_pot.append(spike)
            x = torch.stack(spike_pot, dim=0)
            # x = self.merge(x)
        else:
            x = self.relu(x)
        return x
# class LIFSpike(nn.Module):
#     def __init__(self, T=8,thresh=1.0, tau=1., gama=0.5):
#     # def __init__(self, thresh=1.0, tau=0.5, gama=0.5):
#         super(LIFSpike, self).__init__()
#         self.act = ZIF.apply
#         # self.act = SgNormal.apply
#         # self.act = ZO.apply
#         # self.k = 10
#         # self.act = F.sigmoid
#         self.thresh = thresh
#         self.tau = tau
#         self.gama = gama
#         self.mode = 'bptt'
#         self.T = T

#     def forward(self, x):
#         mem = 0
#         spike_pot = []
#         # T = x.shape[0]
#         for t in range(self.T):
#             mem = mem * self.tau + x[t, ...]
#             spike = self.act(mem - self.thresh, self.gama)
#             # spike = self.act((mem - self.thresh)*self.k)
#             mem = (1 - spike) * mem
#             spike_pot.append(spike)
#         return torch.stack(spike_pot, dim=0)
# def add_dimention(x, T):
#     x.unsqueeze_(1)
#     x = x.repeat(T, 1, 1, 1, 1)
#     return x

def const_encode(x, T):
    #print(x.size())
    x = x.repeat(T, 1, 1, 1)
    #print(x.size())
    return x

## using No trick
# def rate_encode(x, T):
#     #print(torch.min(x), torch.max(x))
#     x = torch.bernoulli(x.repeat(T, 1, 1, 1))
#     #print(x.size())
#     return x

## using Reparameterization trick
# def rate_encode(x, T):
#     #print(torch.min(x), torch.max(x))
#     x = x.repeat(T, 1, 1, 1)
#     epsilon = torch.rand_like(x) # uniform variable with same shape as x :TxB,c,w,h
#     # x = torch.sigmoid((x - epsilon) * 1e2).float()    #print(x.size())
#     x = (x >= epsilon).float()
#     return x

## using Bhaskar Trick
def rate_encode(x, T):
    #print(torch.min(x), torch.max(x))
    mapping_class = rateEncode_SR.apply
    x = mapping_class(x,T)
    return x
def fgsm_attack(model,images,labels,lossFun,eps):
        images = images.clone().detach().to(images.device)
        labels = labels.clone().detach().to(images.device)
        images.requires_grad = True
        outputs = model(images)    
        cost = lossFun(outputs, labels)
        # Update adversarial images
        grad = torch.autograd.grad(cost, images, retain_graph=False, create_graph=False)[0]
        grad_abs = torch.abs(grad)
        _, indices = torch.topk(grad_abs.view(-1), eps, largest=True)
        adv_images = images.clone().detach()
        adv_images.view(-1)[indices] = images.view(-1)[indices] + torch.sign(grad.view(-1)[indices])
        adv_images = torch.clamp(adv_images, min=0, max=1).detach()
        return adv_images

class rateEncode_SR(torch.autograd.Function):
    @staticmethod
    def forward(ctx,inputs, T):
        out = torch.bernoulli(inputs.repeat(T, 1, 1, 1))
        ctx.save_for_backward(inputs,out,torch.tensor(T).type(torch.FloatTensor))
        return out
    @staticmethod
    def backward(ctx, grad_output):
        (inputs, outs,T) = ctx.saved_tensors
        T = T.clone().type(torch.LongTensor)
        BS = (grad_output.shape[0] / T).type(torch.LongTensor)
        # grad_input = torch.zeros_like(inputs).to(inputs.device)
        # for i in range(0,T):
        #     grad_input += 1/T * grad_output[i*BS:(i+1)*BS,:,:,:]
        # print('here')
        grad_input =  torch.mean(grad_output.view(T, BS, inputs.size(1),inputs.size(2),inputs.size(3) ), dim = 0)
        if torch.linalg.norm(grad_input) == 0 : print('zeroGrad found in rate func')
        #     raise Exception("Sorry, gradient with respect to Bern is zero") 
        # print(grad_input.shape)
        return grad_input.to(inputs.device),None



class ConvexCombination(nn.Module):
    def __init__(self, n):
        super().__init__()
        self.n = n
        self.comb = nn.Parameter(torch.ones(n) / n)

    def forward(self, *args):
        assert(len(args) == self.n)
        out = 0.
        for i in range(self.n):
            out += args[i] * self.comb[i]
        return out
