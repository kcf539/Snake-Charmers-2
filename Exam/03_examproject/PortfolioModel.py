""" a portfolio with a risky and a safe asset (Problem 3)

Starting point for the exam. The methods raising NotImplementedError are the ones
you should write yourself.

"""

from types import SimpleNamespace
import numpy as np

class PortfolioModelClass:
    """ a portfolio of a risky and a safe asset with a rebalancing rule """

    def __init__(self,**kwargs):
        """ set the default parameters, then overwrite with any keyword arguments """

        par = self.par = SimpleNamespace()

        # a. returns
        par.mu = 0.05 # mean log return on the risky asset
        par.sigma = 0.20 # standard deviation of the log return on the risky asset
        par.r = 0.01 # log return on the safe asset

        # b. the rebalancing rule
        par.theta_star = 0.50 # target share of wealth in the risky asset
        par.Delta = 0.10 # width of the no-trade band
        par.tau = 0.01 # proportional transaction cost

        # c. preferences
        par.gamma = 3.0 # relative risk aversion

        # d. simulation settings
        par.W0 = 1.0 # initial wealth
        par.T = 40 # number of periods
        par.N = 50_000 # number of simulated portfolios
        par.seed = 2026 # seed for the random number generator

        # e. overwrite with keyword arguments, e.g. PortfolioModelClass(Delta=0.0)
        for key,value in kwargs.items(): setattr(par,key,value)

        # f. empty container for simulation results
        self.sim = SimpleNamespace()

    def __str__(self):
        """ called when using print """

        par = self.par

        text = 'Portfolio model with:\n'
        text += f'  mu    = {par.mu:.4f}, sigma = {par.sigma:.4f}, r = {par.r:.4f}\n'
        text += f'  theta_star = {par.theta_star:.4f}, Delta = {par.Delta:.4f}, tau = {par.tau:.4f}\n'
        text += f'  gamma = {par.gamma:.4f} (relative risk aversion)\n'
        text += f'  W0 = {par.W0:.2f}, T = {par.T}, N = {par.N:,}, seed = {par.seed}'

        return text

    def draw_returns(self):
        """ draw the gross return on the risky asset in all periods and all portfolios

        Returns:

            (ndarray): gross returns with shape (N,T)

        """

        par = self.par

        rng = np.random.default_rng(par.seed)
        eps = rng.normal(size=(par.N,par.T))

        return np.exp(par.mu + par.sigma*eps)

    def u(self,W):
        """ CRRA utility of wealth """

        par = self.par

        return W**(1-par.gamma)/(1-par.gamma)

    def trade(self,theta):
        """ the share of wealth in the risky asset after trading, and the amount traded

        Args:

            theta (ndarray): share of wealth in the risky asset, before trading

        Returns:

            (tuple): (theta_post, amount traded, as a share of wealth)

        """

        par = self.par

        # a. trade back to the target if outside the no-trade band
        outside_band = np.abs(theta-par.theta_star) > par.Delta
        theta_post = np.where(outside_band,par.theta_star,theta)

        # b. the amount traded, as a share of wealth
        amount = np.abs(theta_post-theta)

        return theta_post,amount

    def simulate(self,R=None):
        """ simulate all N portfolios forward T periods

        Args:

            R (ndarray or None): gross risky returns, shape (N,T). Drawn here if not given.

        Returns:

            (SimpleNamespace): self.sim, holding W, theta and the number of trades

        """

        par = self.par
        sim = self.sim

        # a. the returns to use, and the (constant) gross return on the safe asset
        if R is None: R = self.draw_returns()
        Rf = np.exp(par.r)

        # b. containers for the full path, starting at the target with wealth W0
        sim.W = np.empty((par.N,par.T+1))
        sim.theta = np.empty((par.N,par.T+1))
        sim.n_trades = np.zeros(par.N)

        sim.W[:,0] = par.W0
        sim.theta[:,0] = par.theta_star

        # c. loop over time, vectorized over all N portfolios
        for t in range(par.T):

            theta_post,amount = self.trade(sim.theta[:,t])
            sim.n_trades += amount > 0

            W_post = sim.W[:,t]*(1-par.tau*amount)

            sim.W[:,t+1] = theta_post*W_post*R[:,t] + (1-theta_post)*W_post*Rf
            sim.theta[:,t+1] = theta_post*W_post*R[:,t]/sim.W[:,t+1]

        sim.R = R

        return sim

    def summary(self):
        """ the numbers to report for a rule, including expected utility

        Returns:

            (dict): the six numbers asked for in the exam text

        """

        par = self.par
        sim = self.sim

        WT = sim.W[:,-1]

        return {
            'trades': sim.n_trades.mean(),
            'distance': np.abs(sim.theta[:,:par.T]-par.theta_star).mean(),
            'mean WT': WT.mean(),
            'median WT': np.median(WT),
            '10th pct WT': np.percentile(WT,10),
            'E[u(WT)]': self.u(WT).mean(),
        }
