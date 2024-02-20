from __future__ import annotations
import torch
from torch.nn import Linear
import abc
import tqdm

from gpytorch.means import ConstantMean, LinearMean, ZeroMean
from gpytorch.distributions import MultivariateNormal
import gpytorch

from models.kernels import ExponentialKernel, SimpleSincKernel,  InverseDistance
from gpytorch.kernels import rbf_kernel
from gpytorch import kernels

# exact GP with constraints
class ExactGP(gpytorch.models.ExactGP):
    def __init__(self, x, y, kernel=None, mean_mode='Constant', likelihood=None, **kwargs):
        if likelihood is None:
            likelihood = gpytorch.likelihoods.GaussianLikelihood()
        super().__init__(x, y.flatten(), likelihood)
        if mean_mode == 'Zero':
            self.mean_module = ZeroMean()
        if mean_mode == 'Constant':
            self.mean_module = ConstantMean()
        if mean_mode == 'Linear':
            self.mean_module = LinearMean(x.shape[-1])
        if kernel is None:
            #kernel = gpytorch.kernels.ScaleKernel(gpytorch.kernels.RBFKernel(ard_num_dims=3))
            #kernel = gpytorch.kernels.ScaleKernel(gpytorch.kernels.PeriodicKernel( active_dims=(0,1,2) ))
            #constraints= gpytorch.constraints.Interval(torch.tensor([20.,20., 8]), torch.tensor([30., 30., 12]))
            constraints = gpytorch.constraints.Interval(torch.tensor([24., 24., 5]), torch.tensor([40., 40., 12]))
            #outputscale_constraint = gpytorch.constraints.Interval(torch.tensor(30.), torch.tensor(45.))
            outputscale_constraint = gpytorch.constraints.Interval(torch.tensor(634.), torch.tensor(813.))
            kernel = gpytorch.kernels.ScaleKernel(ExponentialKernel(ard_num_dims = 3, lengthscale_constraint=constraints)
                                                         , outputscale_constraint=outputscale_constraint)
            #kernel = gpytorch.kernels.ScaleKernel(gpytorch.kernels.PeriodicKernel(ard_num_dims=1)
             #                          , outputscale_constraint=outputscale_constraint)
            #kernel.register_constraint('outputscale', outputscale_constraint)
            #kernel = gpytorch.kernels.ScaleKernel(SimpleSincKernel(ard_num_dims=3))

            #kernel = gpytorch.kernels.ScaleKernel(gpytorch.kernels.MaternKernel(ard_num_dims=3))
            #kernel = gpytorch.kernels.ScaleKernel(gpytorch.kernels.SpectralDeltaKernel(num_dims=3,ard_num_dims=3))
        # print(self.prediction_strategy)

        self.kernel = kernel
        self.likelihood = likelihood

    @abc.abstractmethod
    def add_nn_layers(self):
        ...

    pass

    def forward(self, x):
        # if self.train():
        # x = torch.randn_like(x)+ x
        mean_x = self.mean_module(x)
        covar_x = self.kernel(x)
        #print(covar_x)
        return MultivariateNormal(mean_x, covar_x)

    def predict(self, x):
        self.eval()
        self.likelihood.eval()
        with torch.no_grad():
            prediction = self(x)
        self.train()
        return self.likelihood(prediction)

    def train_model(self, x, y, n_epochs=100, lr=1e-1, fix_noise_variance=None, verbose=True):

        # def train_model(self, x, y, **kwargs):
        # self.train()
        if fix_noise_variance is not None:
            self.likelihood.noise = fix_noise_variance
            training_parameters = [p for name, p in self.named_parameters()
                                   if not name.startswith('likelihood')]  # excluding noise from parameters to optimize

        else:
            training_parameters = self.parameters()
        optimizer = torch.optim.Adamax(training_parameters, lr=lr)
        mll = gpytorch.mlls.ExactMarginalLogLikelihood(self.likelihood, self)
        likelihood = gpytorch.likelihoods.GaussianLikelihood()
        with tqdm.trange(n_epochs, disable=not verbose) as bar:
            for _ in bar:
                optimizer.zero_grad()
                out = self(x)

                #print(out.loc)
                loss = -mll(out, y.flatten())
                #print(loss)
                loss.backward()
                #print("loss")
                optimizer.step()
                postfix = dict(Loss=f"{loss.item():.3f}",
                               noise=f"{self.likelihood.noise.item():.3}")

                if (hasattr(self.kernel, 'base_kernel') and hasattr(self.kernel.base_kernel, 'lengthscale')):
                    lengthscale = self.kernel.base_kernel.lengthscale
                    if lengthscale is not None:
                        lengthscale = lengthscale.squeeze(0).detach().cpu().numpy()
                else:
                    lengthscale = self.kernel.lengthscale

                if lengthscale is not None:
                    if len(lengthscale) > 1:
                        lengthscale_repr = [f"{l:.3f}" for l in lengthscale]
                        postfix['lengthscale'] = f"{lengthscale_repr}"
                    else:
                        postfix['lengthscale'] = f"{lengthscale[0]:.3f}"

                bar.set_postfix(postfix)
            return (self)


### Buid a Nested GP model
class NestedExpGP(gpytorch.models.ExactGP):
    def __init__(self, x, y, kernel=None, mean_mode='Constant', likelihood=None, **kwargs):
        if likelihood is None:
            likelihood = gpytorch.likelihoods.GaussianLikelihood()
        super().__init__(x, y.flatten(), likelihood)
        if mean_mode == 'Zero':
            self.mean_module = ZeroMean()
        if mean_mode == 'Constant':
            self.mean_module = ConstantMean()
        if mean_mode == 'Linear':
            self.mean_module = LinearMean(x.shape[-1])
        if kernel is None:

            #constraints= gpytorch.constraints.Interval(torch.tensor([20.,20., 8.]), torch.tensor([30., 30., 12]))
            constraints = gpytorch.constraints.Interval(torch.tensor([24.6, 20.6, 5.]), torch.tensor([40.13, 40.13, 13]))
            #constraints2 = gpytorch.constraints.Interval(torch.tensor([ 8.]), torch.tensor([ 12.]))
            constraints2 = gpytorch.constraints.Interval(torch.tensor([5.]), torch.tensor([13.]))

            #outputscale_constraint = gpytorch.constraints.Interval(torch.tensor(30.), torch.tensor(40.))
            outputscale_constraint = gpytorch.constraints.Interval(torch.tensor(770.), torch.tensor(780.))
            #outputscale_constraint2 = gpytorch.constraints.Interval(torch.tensor(10.), torch.tensor(13.))
            outputscale_constraint2 = gpytorch.constraints.Interval(torch.tensor(40.), torch.tensor(50.))
            kernel1 = gpytorch.kernels.ScaleKernel(ExponentialKernel(ard_num_dims = 3, lengthscale_constraint=constraints)
                                                  , outputscale_constraint=outputscale_constraint)
            kernel2 = gpytorch.kernels.ScaleKernel(ExponentialKernel(active_dims=torch.tensor([2]), lengthscale_constraint=constraints2)
                                                  , outputscale_constraint=outputscale_constraint2)

        self.kernel1 = kernel1
        self.kernel2 = kernel2
        self.likelihood = likelihood
    def forward(self, x):
        # if self.train():
        # x = torch.randn_like(x)+ x
        mean_x = self.mean_module(x)
        covar_x = self.kernel1(x) + self.kernel2(x)
        return MultivariateNormal(mean_x, covar_x)

    def predict(self, x):
        self.eval()
        self.likelihood.eval()
        with torch.no_grad():
            prediction = self(x)
        self.train()
        return self.likelihood(prediction)

    def train_model(self, x, y, n_epochs=100, lr=1e-1, fix_noise_variance=None, verbose=True):

        # def train_model(self, x, y, **kwargs):
        # self.train()
        if fix_noise_variance is not None:
            self.likelihood.noise = fix_noise_variance
            training_parameters = [p for name, p in self.named_parameters()
                                   if not name.startswith('likelihood')]  # excluding noise from parameters to optimize

        else:
            training_parameters = self.parameters()
        optimizer = torch.optim.Adamax(training_parameters, lr=lr)
        mll = gpytorch.mlls.ExactMarginalLogLikelihood(self.likelihood, self)
        likelihood = gpytorch.likelihoods.GaussianLikelihood()
        criterion = torch.nn.MSELoss()
        with tqdm.trange(n_epochs, disable=not verbose) as bar:
            for _ in bar:
                optimizer.zero_grad()
                out = self(x)

                loss = -mll(out, y.flatten())
                loss.backward()
                optimizer.step()
                postfix = dict(Loss=f"{loss.item():.3f}",
                               noise=f"{self.likelihood.noise.item():.3}")

                if (hasattr(self.kernel1, 'base_kernel') and hasattr(self.kernel1.base_kernel, 'lengthscale')):
                    lengthscale1 = self.kernel1.base_kernel.lengthscale
                    if lengthscale1 is not None:
                        lengthscale1 = lengthscale1.squeeze(0).detach().cpu().numpy()
                else:
                    lengthscale1 = self.kernel1.lengthscale

                if lengthscale1 is not None:
                    if len(lengthscale1) > 1:
                        lengthscale_repr1 = [f"{l:.3f}" for l in lengthscale1]
                        postfix['lengthscale1'] = f"{lengthscale_repr1}"
                    else:
                        postfix['lengthscale1'] = f"{lengthscale1[0]:.3f}"
                if (hasattr(self.kernel2, 'base_kernel') and hasattr(self.kernel2.base_kernel, 'lengthscale')):
                    lengthscale2 = self.kernel2.base_kernel.lengthscale
                    if lengthscale2 is not None:
                        lengthscale2 = lengthscale2.squeeze(0).detach().cpu().numpy()
                else:
                    lengthscale2 = self.kernel2.lengthscale

                if lengthscale2 is not None:
                    if len(lengthscale2) > 1:
                        lengthscale_repr2 = [f"{l:.3f}" for l in lengthscale2]
                        postfix['lengthscale2'] = f"{lengthscale_repr2}"
                    else:
                        postfix['lengthscale2'] = f"{lengthscale2[0]:.3f}"

                bar.set_postfix(postfix)
            return (self)











