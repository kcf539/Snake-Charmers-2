from types import SimpleNamespace

import numpy as np
from scipy import optimize


class ConsumerClass:
    """ a consumer who buys food, bus trips and train trips

    The three goods are combined into utility in two steps ("nests"):

        1) bus and train are combined into one travel composite
        2) food and the travel composite are combined into utility

    Bus and train sit together in a nest because they are similar goods (two
    ways of travelling), while food is a completely different kind of good.

    Instead of picking quantities directly, everything below is written in
    terms of *budget shares*: s1 is the share of income spent on food, and w
    is the share of what is left (the travel budget) spent on the bus. Every
    (s1, w) in the unit square [0,1] x [0,1] is then automatically an
    affordable choice, so there is no budget constraint left to check by hand.

    """

    def __init__(self, par=None):
        """ create a consumer, optionally changing some parameters

        Args:

            par (dict or None): parameters to change from their baseline
                value, e.g. par={'sigma_B':3.0} for the "substitutes"
                calibration. Any parameter not mentioned keeps the value
                set in .setup().

        """

        # a. start from the baseline ("complements") parameters
        self.setup()

        # b. overwrite only the parameters we were asked to change
        if par is not None:
            for key, value in par.items():
                self.par.__dict__[key] = value

    def setup(self):
        """ set the baseline parameters -- the "complements" calibration """
        
        par = self.par = SimpleNamespace()

        # a. preference weights (both between 0 and 1)
        par.alpha = 0.60  # weight on food, vs. the travel composite
        par.beta = 0.50  # weight on the bus, vs. the train

        # b. elasticities of substitution
        par.sigma_A = 0.80  # food vs. travel
        par.sigma_B = 0.40  # bus vs. train (0.40 here = complements, 3.00 = substitutes)

        # c. prices and income
        par.p1 = 1.0  # price of food
        par.p2 = 1.0  # price of a bus trip
        par.p3 = 1.5  # price of a train trip
        par.I = 10.0  # income

        # d. a tiny number used in .ces(), see the note there
        par.s_min = 1e-12

    def __str__(self):
        """ a short printout of the parameters, to check a calibration at a glance """

        par = self.par

        lines = ['ConsumerClass']
        lines.append(f'  alpha = {par.alpha:.2f}, beta = {par.beta:.2f}')
        lines.append(f'  sigma_A = {par.sigma_A:.2f}, sigma_B = {par.sigma_B:.2f}')
        lines.append(f'  p1 = {par.p1:.2f}, p2 = {par.p2:.2f}, p3 = {par.p3:.2f}, I = {par.I:.2f}')

        return '\n'.join(lines)

    ###################
    # 1. the CES nest #
    ###################

    def ces(self, z1, z2, weight, sigma):
        """ combine two quantities into one CES ("constant elasticity of
        substitution") aggregate

        This one formula is used twice: once for bus and train, and once for
        food and the travel composite it produces. That is why it is its own
        method instead of being written out twice inside .utility().

        Args:

            z1 (float or ndarray): first quantity
            z2 (float or ndarray): second quantity
            weight (float): weight on z1, between 0 and 1
            sigma (float): elasticity of substitution, must not be 1

        Returns:

            (float or ndarray): the CES aggregate

        """

        par = self.par

        assert not np.isclose(sigma, 1.0), 'sigma = 1 is not allowed (rho would be 0, dividing by it)'

        # a. never raise a zero quantity to a negative power. rho is negative
        #    whenever sigma < 1, and a corner of the budget-share square (e.g.
        #    w=0) does give a quantity of exactly zero, so this really happens.
        #    s_min has to be tiny: L-BFGS-B estimates gradients by nudging the
        #    input by about 1e-8, and if s_min were that large too, two nudged
        #    points could floor to the same number and look like zero slope.
        z1 = np.maximum(z1, par.s_min)
        z2 = np.maximum(z2, par.s_min)

        # b. the CES formula itself
        rho = 1 - 1/sigma
        return (weight*z1**rho + (1-weight)*z2**rho)**(1/rho)

    def utility(self, x1, x2, x3):
        """ nested utility of a bundle of quantities (eq. 1 and 2 in the assignment)

        Args:

            x1 (float or ndarray): quantity of food
            x2 (float or ndarray): quantity of bus trips
            x3 (float or ndarray): quantity of train trips

        Returns:

            (float or ndarray): utility

        """

        par = self.par

        # a. inner nest: combine bus and train into one travel composite
        x_travel = self.ces(x2, x3, par.beta, par.sigma_B)

        # b. outer nest: combine food and the travel composite into utility
        u = self.ces(x1, x_travel, par.alpha, par.sigma_A)

        return u

    ###############################
    # 2. the nested budget shares #
    ###############################

    def shares(self, s1, w):
        """ the three budget shares implied by the nested shares (eq. 3)

        Args:

            s1 (float or ndarray): share of income spent on food
            w (float or ndarray): share of the travel budget spent on the bus

        Returns:

            (tuple): the three budget shares s1, s2, s3 (always summing to one)

        """

        return s1, (1-s1)*w, (1-s1)*(1-w)

    def quantities(self, s1, w):
        """ the three quantities implied by the nested shares

        Args:

            s1 (float or ndarray): share of income spent on food
            w (float or ndarray): share of the travel budget spent on the bus

        Returns:

            (tuple): the three quantities x1, x2, x3

        """

        par = self.par

        s1_share, s2, s3 = self.shares(s1, w)

        return s1_share*par.I/par.p1, s2*par.I/par.p2, s3*par.I/par.p3

    def value_of_choice(self, s1, w):
        """ utility of the bundle implied by the nested shares (eq. 4, without the max)

        This chains .quantities() and .utility() together, so a choice of
        (s1, w) turns directly into a utility number. In the next assignment
        this is exactly the function an optimizer searches over.

        Args:

            s1 (float or ndarray): share of income spent on food
            w (float or ndarray): share of the travel budget spent on the bus

        Returns:

            (float or ndarray): utility

        """

        x1, x2, x3 = self.quantities(s1, w)

        return self.utility(x1, x2, x3)

    def objective(self, s):
        """ minus utility, written for a minimizer (which only knows how to minimize)

        Args:

            s (ndarray): array of length 2, holding (s1,w)

        Returns:

            (float): minus utility

        """

        return -self.value_of_choice(s[0], s[1])

    #################
    # 3. solving it #
    #################

    def solve_grid(self, N=200, do_print=True):
        """ solve by a 2-dimensional grid search over the nested shares

        Every point in the unit square is a possible choice, so there is
        nothing to mask out here: we just check all N*N of them and keep
        the best.

        Args:

            N (int): number of grid points for each variable
            do_print (bool): print the solution

        Returns:

            (SimpleNamespace): the grids, the utility values and the best point

        """

        opt = SimpleNamespace()

        # a. the two grids, and every combination of them
        s1_vec = np.linspace(0, 1, N)
        w_vec = np.linspace(0, 1, N)
        s1_grid, w_grid = np.meshgrid(s1_vec, w_vec, indexing='ij')

        # b. utility in every grid point at once (no loop needed)
        u_grid = self.value_of_choice(s1_grid, w_grid)

        # c. the best point
        i_max = np.unravel_index(np.argmax(u_grid), u_grid.shape)
        opt.s1 = s1_grid[i_max]
        opt.w = w_grid[i_max]
        opt.u = u_grid[i_max]

        # d. the budget shares this implies
        _, opt.s2, opt.s3 = self.shares(opt.s1, opt.w)

        # e. keep the grids too, so we can plot them afterwards
        opt.s1_grid = s1_grid
        opt.w_grid = w_grid
        opt.u_grid = u_grid

        if do_print:
            print(f'grid search, N = {N}: (s1,w) = ({opt.s1:.4f},{opt.w:.4f}), '
                  f'u = {opt.u:.4f}, evaluations = {N*N}')

        return opt

    def solve(self, s0=None, do_print=True, **kwargs):
        """ solve with L-BFGS-B

        The bounds are ((0,1),(0,1)) -- the whole constraint set -- so
        nothing else needs to be arranged.

        Args:

            s0 (ndarray): starting guess for (s1,w), defaults to (0.5,0.5)
            do_print (bool): print the solution
            kwargs: passed on to optimize.minimize, e.g. options={'ftol':1e-10}

        Returns:

            (SimpleNamespace): the solution and the convergence path

        """

        opt = SimpleNamespace()

        # a. starting guess
        if s0 is None: s0 = np.array([0.5, 0.5])
        s0 = np.asarray(s0, dtype=float)

        # b. record the path with a callback. optimize.minimize calls this
        #    once per iteration with the current point, but never for the
        #    starting point itself, so we add s0 to the list ourselves.
        path = [s0.copy()]

        def callback(sk):
            path.append(sk.copy())

        # c. minimize minus utility -- this is the whole solver
        res = optimize.minimize(self.objective, s0, method='L-BFGS-B',
            bounds=((0, 1), (0, 1)), callback=callback, **kwargs)

        # d. results
        opt.s1, opt.w = res.x
        _, opt.s2, opt.s3 = self.shares(opt.s1, opt.w)
        opt.u = -res.fun
        opt.path = np.array(path)
        opt.res = res

        if do_print:
            print(f'L-BFGS-B: (s1,w) = ({opt.s1:.4f},{opt.w:.4f}), u = {opt.u:.4f}, '
                  f'evaluations = {res.nfev}, success = {res.success}')

        return opt


class GovernmentClass(ConsumerClass):
    """ a government raising revenue from the consumer in ConsumerClass

    Two kinds of instrument:

        1) a lump-sum tax T, which reduces income
        2) product taxes tau1, tau2, tau3, which raise the three prices

    GovernmentClass inherits from ConsumerClass, so every method from
    sections 1-3 -- .utility(), .solve(), .solve_grid(), .value_of_choice(),
    ... -- keeps working exactly as before. .set_taxes() only changes what
    par.p1, par.p2, par.p3 and par.I *are*; it never touches how they get
    used.

    """

    def __init__(self, par=None):
        """ create a government, optionally changing some parameters

        Args:

            par (dict or None): parameters to change from their baseline
                value, exactly as in ConsumerClass.

        """

        # a. the consumer's own baseline parameters
        self.setup()

        # b. the tax instruments, all starting at zero
        self.setup_government()

        # c. overwrite only the parameters we were asked to change
        if par is not None:
            for key, value in par.items():
                self.par.__dict__[key] = value

        # d. remember the prices and income *before* any tax -- must happen
        #    after c., or a changed baseline price would not be picked up
        self.sync_pre_tax()

    def setup_government(self):
        """ add the tax instruments to the parameters, all starting at zero """

        par = self.par

        par.T = 0.0  # lump-sum tax (a transfer if negative)

        par.tau1 = 0.0  # tax rate on food
        par.tau2 = 0.0  # tax rate on bus trips
        par.tau3 = 0.0  # tax rate on train trips

    def sync_pre_tax(self):
        """ store the current prices and income as the pre-tax situation

        Revenue is collected at these prices (eq. 5 uses p_j, not
        (1+tau_j)*p_j), so they have to be the ones from *before* any tax.

        """

        par = self.par

        par.p1_pre = par.p1
        par.p2_pre = par.p2
        par.p3_pre = par.p3
        par.I_pre = par.I

    ##############################
    # 1. what the consumer faces #
    ##############################

    def set_taxes(self, T=0.0, tau1=0.0, tau2=0.0, tau3=0.0):
        """ set the taxes, and update the prices and income the consumer faces

        The price the consumer pays for good j is (1+tau_j) times the price
        the seller receives, and income is reduced krone-for-krone by the
        lump-sum tax. Everything is computed from the *pre-tax* values, so
        calling this again with different rates never compounds: the result
        only depends on the rates passed in this call, never on earlier ones.

        Args:

            T (float): lump-sum tax
            tau1 (float): tax rate on food
            tau2 (float): tax rate on bus trips
            tau3 (float): tax rate on train trips

        """

        par = self.par

        # a. remember the taxes themselves, for .tax_revenue()
        par.T = T
        par.tau1 = tau1
        par.tau2 = tau2
        par.tau3 = tau3

        # b. the prices the consumer faces
        par.p1 = (1+tau1)*par.p1_pre
        par.p2 = (1+tau2)*par.p2_pre
        par.p3 = (1+tau3)*par.p3_pre

        # c. income after the lump-sum tax
        par.I = par.I_pre - T

    ##########################################
    # 2. revenue, and what the consumer gets #
    ##########################################

    def tax_revenue(self, opt=None):
        """ total tax revenue given the taxes currently set (eq. 5)

        Revenue is collected at the price the *seller* receives, so the tax
        paid on good j is tau_j*p_j_pre*x_j, not tau_j*p_j*x_j.

        Args:

            opt (SimpleNamespace): a solution from .solve(). Solved for here
                if not given -- pass it in when you already have it, to
                avoid solving the same problem twice.

        Returns:

            (float): tax revenue

        """

        par = self.par

        # a. what the consumer buys, given the taxes currently set
        if opt is None: opt = self.solve(do_print=False)
        x1, x2, x3 = self.quantities(opt.s1, opt.w)

        # b. the lump-sum tax, plus the product tax on each good
        R = par.T + par.tau1*par.p1_pre*x1 + par.tau2*par.p2_pre*x2 + par.tau3*par.p3_pre*x3

        return R

    def revenue_and_utility(self, tau, goods=(2,)):
        """ revenue and utility when the same tax rate is put on each good in goods

        Args:

            tau (float): the common tax rate
            goods (tuple): which goods to tax, e.g. (2,) or (2,3) or (1,2,3)

        Returns:

            (tuple): (revenue, utility)

        """

        # a. tau on the goods in goods, zero on the others
        taus = {1: 0.0, 2: 0.0, 3: 0.0}
        for good in goods:
            taus[good] = tau
        self.set_taxes(T=0.0, tau1=taus[1], tau2=taus[2], tau3=taus[3])

        # b. solve the consumer's problem with these taxes, and its revenue
        opt = self.solve(do_print=False)
        R = self.tax_revenue(opt)

        return R, opt.u

    def revenue_and_utility_lump_sum(self, T):
        """ the same, for a lump-sum tax of T

        Args:

            T (float): the lump-sum tax

        Returns:

            (tuple): (revenue, utility)

        """

        self.set_taxes(T=T)
        opt = self.solve(do_print=False)
        R = self.tax_revenue(opt)

        return R, opt.u

    ##########################################
    # 3. hitting a given revenue requirement #
    ##########################################

    def max_revenue(self, goods=(2,), tau_max=10.0, N=1001):
        """ the largest revenue this instrument can ever raise

        A grid over the tax rate is enough, exactly as in section 2.1:
        compute the revenue at every grid point and keep the best one.

        If the answer comes back at tau_max, the curve was still rising when
        the grid ran out -- there is no top in the range searched.

        Args:

            goods (tuple): which goods to tax
            tau_max (float): largest tax rate to consider
            N (int): number of grid points

        Returns:

            (tuple): (the revenue-maximizing rate, the largest revenue)

        """

        # a. revenue at every point of the grid
        tau_vec = np.linspace(0, tau_max, N)
        R_vec = np.array([self.revenue_and_utility(tau, goods=goods)[0] for tau in tau_vec])

        # b. the best point
        i_max = np.argmax(R_vec)

        return tau_vec[i_max], R_vec[i_max]

    def find_tax_rate(self, R_target, goods=(2,), bracket=(1e-10, 1.0)):
        """ the tax rate on goods that raises exactly R_target

        Careful: revenue is not always increasing in the tax rate (section
        4.3). There can be two rates that raise the same revenue, and a
        revenue target above the largest possible revenue cannot be reached
        at all. In that case there is no sign change in the bracket, and the
        root-finder raises a ValueError -- which is the correct answer, not
        a bug. Catch it and return np.nan.

        Args:

            R_target (float): the revenue requirement
            goods (tuple): which goods to tax
            bracket (tuple): interval of tax rates to search in

        Returns:

            (float): the tax rate, or np.nan if the target cannot be reached

        """

        def f(tau):
            R, u = self.revenue_and_utility(tau, goods=goods)
            return R - R_target

        try:
            res = optimize.root_scalar(f, bracket=bracket, method='brentq')
            return res.root
        except ValueError:
            return np.nan


class ExternalityConsumerClass(ConsumerClass):
    """ a consumer who also cares about the CO2 pollution from travelling

    Section 5's extension. Every bus trip and every train trip emits some
    CO2, and the consumer's utility falls in the total emitted -- e.g.
    because they are environmentally conscious, or simply dislike knowing
    that their travel contributes to climate change. Buses run on diesel and
    (Danish) trains run mostly on electricity, so a bus trip is given a
    higher emission rate than a train trip.

    Concretely, emissions are linear in the two travel quantities,

        E(x2,x3) = c2*x2 + c3*x3,      c2 > c3 > 0,

    and enter utility on top of the ordinary CES utility from section 1,

        U(x1,x2,x3) = u(x1,x2,x3) - gamma*E(x2,x3),      gamma > 0.

    u(x1,x2,x3) itself -- .utility() -- is left completely alone, so it still
    means exactly what it meant in section 1: the private enjoyment of the
    bundle, ignoring pollution. Only .value_of_choice() is changed, to the
    *net* of that enjoyment and the emissions it causes. Since .solve() and
    .solve_grid() are both written in terms of .value_of_choice() (via
    .objective()), inheriting them unchanged is enough to make them solve
    the *right* problem -- nothing about optimization has to be touched.

    """

    def setup(self):
        """ the baseline parameters, plus the two emission rates and the
        marginal disutility of CO2 """

        # a. everything from ConsumerClass
        super().setup()
        par = self.par

        # b. kg of CO2 per trip -- a bus trip pollutes three times as much
        #    as a train trip
        par.c2 = 0.90  # kg CO2 per bus trip
        par.c3 = 0.30  # kg CO2 per train trip

        # c. marginal disutility of one kg of CO2, in utility units
        par.gamma = 0.15

    def emissions(self, x2, x3):
        """ total CO2 emitted by x2 bus trips and x3 train trips

        Args:

            x2 (float or ndarray): quantity of bus trips
            x3 (float or ndarray): quantity of train trips

        Returns:

            (float or ndarray): kg of CO2 emitted

        """

        par = self.par

        return par.c2*x2 + par.c3*x3

    def utility_with_externality(self, x1, x2, x3):
        """ utility net of the disutility from CO2 (U above)

        Args:

            x1 (float or ndarray): quantity of food
            x2 (float or ndarray): quantity of bus trips
            x3 (float or ndarray): quantity of train trips

        Returns:

            (float or ndarray): utility, net of the externality

        """

        par = self.par

        u = self.utility(x1, x2, x3)
        E = self.emissions(x2, x3)

        return u - par.gamma*E

    def value_of_choice(self, s1, w):
        """ utility net of the externality, of the bundle implied by the
        nested shares -- overrides ConsumerClass so the optimizer targets
        the *net* of enjoyment and pollution, not enjoyment alone

        Args:

            s1 (float or ndarray): share of income spent on food
            w (float or ndarray): share of the travel budget spent on the bus

        Returns:

            (float or ndarray): utility, net of the externality

        """

        x1, x2, x3 = self.quantities(s1, w)

        return self.utility_with_externality(x1, x2, x3)
