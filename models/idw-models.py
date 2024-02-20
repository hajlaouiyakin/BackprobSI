from __future__ import annotations
import torch

import abc
import tqdm

import gpytorch

from statistics import mean

from models.kernels import InverseDistanceWithParam
# basic model without parameters
class WeightedInverseDistance(torch.nn.Module):  #  IDW-C (LHOCV ) model
    def __init__(self,x, y, p=None,lenghthscale = True ):
        super(WeightedInverseDistance, self).__init__()
        self.x = x
        self.y = y
        # Add constraints if required
        #constraints = gpytorch.constraints.Interval(torch.tensor([15., 15., 5.]), torch.tensor([30., 30., 15.]))
        constraints = gpytorch.constraints.Interval(torch.tensor([24.6, 24.6, 5.]), torch.tensor([40.13, 40.13, 13]))

        self.kernel = InverseDistanceWithParam(ard_num_dims = self.x.shape[1], lengthscale_constraint=constraints)

    def forward(self, x, y ):


        covar = self.kernel(x)


        #print(self.kernel.a)
        y_int = torch.matmul(covar.evaluate(), y)/torch.matmul(covar.evaluate(), torch.ones_like(y))
        return y_int, covar
    def predict(self, x_test):

        k1 = self.kernel(x_test, self.x,mode = 'test')
        y_pred = torch.matmul(k1.evaluate(), self.y)/torch.matmul(k1.evaluate(), torch.ones_like(self.y))
        return y_pred
    def train(self, x, y, n_epochs=100,lr = 0.01,  verbose = True ):

        optimizer = torch.optim.Adamax(self.parameters(), lr=lr)
        criterion = torch.nn.MSELoss()
        with tqdm.trange(n_epochs, disable=not verbose) as bar:
            for _ in bar:
                optimizer.zero_grad()
                out, covar = self(x,y)


                loss = criterion(out, y)

                loss.backward()

                optimizer.step()

                postfix = dict(Loss=f"{loss.item():.3f}",
                               power=f"{self.kernel.power_param.item():.3f}")

                if (hasattr(self.kernel, 'base_kernel') and hasattr(self.kernel.base_kernel, 'lengthscale')):
                    lengthscale = self.kernel.base_kernel.lengthscale
                    if lengthscale is not None:
                        lengthscale = lengthscale.squeeze(0).detach().cpu().numpy()
                else:
                    lengthscale = self.kernel.lengthscale

                if lengthscale is not None:
                    if len(lengthscale.squeeze(0)) > 1:
                        lengthscale_repr = [f"{l:.3f}" for l in lengthscale.squeeze(0)]
                        postfix['lengthscale'] = f"{lengthscale_repr}"
                    else:
                        print(len(lengthscale))

                        print(lengthscale)
                        postfix['lengthscale'] = f"{lengthscale[0].item():.3f}"

                bar.set_postfix(postfix)
            return (self, loss.item())

# Modified weighted inverse distance

class AdjWeightedInverseDistance(torch.nn.Module): # M-IDW-LHOCV
    def __init__(self,x, y, p=None,lenghthscale = True ):
        super(AdjWeightedInverseDistance, self).__init__()
        self.x = x
        self.y = y
        #constraints = gpytorch.constraints.Interval(torch.tensor([15., 15., 5.]), torch.tensor([30., 30., 15.]))
        constraints = gpytorch.constraints.Interval(torch.tensor([24.6, 24.6, 5.]), torch.tensor([40.13, 40.13, 13]))
        self.kernel = InverseDistanceWithParam(ard_num_dims = self.x.shape[1], lengthscale_constraint=constraints)


    def forward(self, x, c,y ):

        covar = self.kernel(x)

        y_int = torch.matmul(covar.evaluate(), y*c)/torch.matmul(covar.evaluate(), torch.ones_like(y)*c)
        return y_int, covar
    def predict(self, x_test,c):
        k1 = self.kernel(x_test, self.x,mode = 'test')
        y_pred = torch.matmul(k1.evaluate(), self.y*c)/torch.matmul(k1.evaluate(), torch.ones_like(self.y)*c)
        return y_pred
    def train(self, x, y, n_epochs=100,lr = 0.01,  verbose = True ):
        #print(self.para)
        # initialize the coefficients
        c= torch.ones_like(y)
        optimizer = torch.optim.Adamax(self.parameters(), lr=lr)
        criterion = torch.nn.MSELoss()
        with tqdm.trange(n_epochs, disable=not verbose) as bar:
            for _ in bar:
                optimizer.zero_grad()
                out, covar = self(x,c,y)
                e = torch.tensor((out - y)**2, requires_grad=False)
                c= 1+ e/torch.max(e)  # Coeff of the modified approach
                out, covar = self(x, c, y)
                loss = criterion(out, y)
                loss.backward()

                optimizer.step()
                postfix = dict(Loss=f"{loss.item():.3f}",
                               power=f"{self.kernel.power_param.item():.3f}")

                if (hasattr(self.kernel, 'base_kernel') and hasattr(self.kernel.base_kernel, 'lengthscale')):
                    lengthscale = self.kernel.base_kernel.lengthscale
                    if lengthscale is not None:
                        lengthscale = lengthscale.squeeze(0).detach().cpu().numpy()
                else:
                    lengthscale = self.kernel.lengthscale

                if lengthscale is not None:
                    if len(lengthscale.squeeze(0)) > 1:
                        lengthscale_repr = [f"{l:.3f}" for l in lengthscale.squeeze(0)]
                        postfix['lengthscale'] = f"{lengthscale_repr}"
                    else:
                        print(len(lengthscale))

                        print(lengthscale)
                        postfix['lengthscale'] = f"{lengthscale[0].item():.3f}"

                bar.set_postfix(postfix)
            return (self, c,e,loss.item())

# Modified IDW with learnable parameters
class GradAdjWeightedInverseDistance(torch.nn.Module): # GBM-IDW-LHOCV
    def __init__(self, x, y, p=None, lenghthscale=True):
        super(GradAdjWeightedInverseDistance, self).__init__()
        self.x = x
        self.y = y
        #constraints = gpytorch.constraints.Interval(torch.tensor([15., 15., 5.]), torch.tensor([30., 30., 15.]))
        constraints = gpytorch.constraints.Interval(torch.tensor([24.6, 24.6, 5.]), torch.tensor([40.13, 40.13, 13]))
        self.kernel = InverseDistanceWithParam(ard_num_dims=self.x.shape[1], lengthscale_constraint=constraints)
        #self.nn = FeatureExtractor(x.shape[-1], 10, x.shape[-1])
        self.raw_c = torch.nn.Parameter(torch.Tensor(self.y.shape[0], 1))#, requires_grad=True)
        self.reset_parameters()
    @property
    def coefficient(self):
        m = torch.nn.Sigmoid()
        c = m(self.raw_c) + 1.

        return c

    def reset_parameters(self):
        torch.nn.init.xavier_uniform_(self.raw_c)
    def forward(self, x, y):

        c = self.coefficient
        covar = self.kernel(x)
        y_int = torch.matmul(covar.evaluate(), y*c ) / torch.matmul(covar.evaluate(), torch.ones_like(y)*c )
        return y_int, covar

    def predict(self, x_test):
        k1 = self.kernel(x_test, self.x, mode='test')
        c= self.coefficient
        y_pred = torch.matmul(k1.evaluate(), self.y * c) / torch.matmul(k1.evaluate(), torch.ones_like(self.y) * c)
        return y_pred

    def train(self, x, y, n_epochs=100, lr=0.01, verbose=True):

        optimizer = torch.optim.Adamax(self.parameters(), lr=lr)
        criterion = torch.nn.MSELoss()
        with tqdm.trange(n_epochs, disable=not verbose) as bar:
            for _ in bar:
                optimizer.zero_grad()
                out, covar = self(x, y)
                e = torch.tensor((out - y) ** 2, requires_grad=False)
                c= self.coefficient   # IDW coeff
                loss = criterion(out, y)
                loss.backward()
                optimizer.step()

                postfix = dict(Loss=f"{loss.item():.3f}",
                               power=f"{self.kernel.power_param.item():.3f}")

                if (hasattr(self.kernel, 'base_kernel') and hasattr(self.kernel.base_kernel, 'lengthscale')):
                    lengthscale = self.kernel.base_kernel.lengthscale
                    if lengthscale is not None:
                        lengthscale = lengthscale.squeeze(0).detach().cpu().numpy()
                else:
                    lengthscale = self.kernel.lengthscale

                if lengthscale is not None:
                    if len(lengthscale.squeeze(0)) > 1:
                        lengthscale_repr = [f"{l:.3f}" for l in lengthscale.squeeze(0)]
                        postfix['lengthscale'] = f"{lengthscale_repr}"
                    else:
                        print(len(lengthscale))

                        print(lengthscale)
                        postfix['lengthscale'] = f"{lengthscale[0].item():.3f}"

                bar.set_postfix(postfix)
            return (self,c, e, loss.item())




