
Motivation and Theory
=====================

Waiting Times and Memory
^^^^^^^^^^^^^^^^^^^^^^^^

give some motivation?

this time maybe start more direct with the reaction types

maybe start with single cell view and waiting time description... more intuitive
deliver the generator description later on and maybe only then write down
phase-type formulas

maybe focus here in this first section on the "single cell view"; say in next
section that this is can applied automatically for many cells running through the
transition graphs and networks

rates we introduce are the rates of single cells passing through such schemes

generator/ME and waiting time description?

transient and absorbing state?

maybe accompany images below with image for hidden layer structures

mention available reaction types somewhere?

mention phase-type dense and Erlang smallest CV for given transient states

don't forget to mention that n=1 is exponential/Markov case

general master equation for Markov jump processes (continuous time, discrete
state space); SEE NEW notes on Goodnotes: maybe remove this initial condition thing
and directly write the general ME, starting from initial distribution p(w0, t0);
mention that all are 1xn row vectors

.. math::
    \partial_t \pmb{p}(w, t | w_0) = \pmb{p}(w, t | w_0) \, Q

where :math:`Q` is the (possibly infinite sized) transition rate (or generator)
matrix. describe :math:`Q`; also note that ME is sometimes written in transposed
form. with fundamental solution

.. math::
    \pmb{p}(w, t | w_0) = \pmb{p}(w, t | w_0)

however, often hard/impossible to solve.



reaction types; maybe change notation as in methods with :math:`W^{(i)}` and
:math:`W^{(k)}` variables.

- :math:`S \rightarrow E` (cell differentiation),

- :math:`S \rightarrow S + S` (symmetric self-renewing division),

- :math:`S \rightarrow S + E` (asymmetric division),

- :math:`S \rightarrow E + E` (symmetric differentiating division),

- :math:`S \rightarrow` (efflux or cell death) and

- :math:`\rightarrow E` (influx or birth),


.. image:: ../images/wtd_erl_many.png
    :align: center
    :scale: 16 %

hi hi

.. image:: ../images/wtd_phase_type_2.png
    :align: center
    :scale: 15 %


Stochastic Processes
^^^^^^^^^^^^^^^^^^^^

maybe introduce waiting times first (as a single-reaction module, also
mention Markov processes there) and connect different modules here now for
multi-reaction pathways; and then main: how to efficiently characterise them
-> moment simulations (next to stochastic simulations)

maybe say how the competition is implemented (for more than one module starting
at a main variable, we split at its centric hidden variable).

hidden and main variables?

summation formula? for variables and for mean, variance, covariance?

approach based on G to get ODE for the moments? maybe mention at least which
moments are solved

maybe mention summation formulas, but skip the part how to solve them via
PDE (in formulas) at least; just they we get ODE system for these hidden
layer moments

write that MemoCell can produce moment AND stochastic simulations for the
class of cell pathway processes (/general waiting time
stochastic models with the above reaction types)

what about initial conditions? ref to API?

.. math::
    W^{(i)}_t = \sum\nolimits_{j\in\{1,...,u_i\}} W^{(i,j)}_t

for any fixed :math:`i \in \{1,...,v\}`.

mean:

.. math::
    \mathrm{E}\big(W^{(i)}_t\big) = \sum\nolimits_{j\in\{1,...,u_i\}}
    \mathrm{E}\big(W^{(i,j)}_t\big)

covariance and variance:

.. math::
    \mathrm{Var}\big(W^{(i)}_t\big) = \sum\nolimits_{j} \mathrm{Var}\big(W^{(i,j)}_t\big)
    + 2 \sum\nolimits_{j,l | j<l} \mathrm{Cov}\big(W^{(i,j)}_t, W^{(i,l)}_t\big)

where :math:`j,l \in\{1,...,u_i\}`.

.. math::
    \mathrm{Cov}\big(W^{(i)}_t, W^{(k)}_t\big) =
    \sum\nolimits_{j}\sum\nolimits_{l} \mathrm{Cov}\big(W^{(i,j)}_t, W^{(k,l)}_t\big)

where :math:`j \in\{1,...,u_i\}` and :math:`l \in\{1,...,u_k\}`.

the variance and covariances on the hidden layer can be decomposed into
second factorial and mixed moments:
:math:`\mathrm{Var}(X)=\mathrm{E}(X(X-1))+\mathrm{E}(X)-\mathrm{E}(X)^2`
and :math:`\mathrm{Cov}(X, Y)=\mathrm{E}(X Y)-\mathrm{E}(X) \mathrm{E}(Y)`

MemoCell solves :math:`\mathrm{E}\big(W^{(i,j)}_t\big)`,
:math:`\mathrm{E}\big(W^{(i,j)}_t \, (W^{(i,j)}_t-1)\big)` and
:math:`\mathrm{E}\big(W^{(i,j)}_t \, W^{(k,l)}_t\big)` for all hidden variables
:math:`i,k \in \{1,...,v\}`, :math:`i \ne k`, :math:`j \in \{1,...,u_i\}`,
:math:`l \in \{1,...,u_k\}`.


Bayesian Inference
^^^^^^^^^^^^^^^^^^

state main Bayes theorems for model selection and parameter estimation

mention likelihood function? (maybe reference to API here, as log likelihood)

mention nested sampling

allows Bayesian-averaged inference over the complete model space, introduce
formula and sampling procedure (maybe link to API)

.. math::
    p(\pmb{\theta}_k | D, M_k) = \frac{p(D | \pmb{\theta}_k, M_k) \, p(\pmb{\theta}_k| M_k)}{p(D | M_k)}
    = \frac{\mathcal{L}(\pmb{\theta}_k) \, \pi(\pmb{\theta}_k)}{Z_k}

.. math::
    p(M_k | D) = \frac{p(D | M_k) \, p(M_k)}{p(D)}
    = \frac{Z_k \, p(M_k)}{p(D)}

it is sufficient to know model evidence and model prior to know the model
posterior distribution, as :math:`p(D)` can be calculated as
probability-normalizing factor.

parameter prior, for each parameter :math:`\theta` in the vector :math:`\pmb{\theta}`
one has to specify

.. math::
    \pi(\theta) = \left. \begin{cases} 1 / (b_u - b_l) & \text{if } \theta \in [b_l, b_u] \\
    0 & \text{else} \end{cases} \right\}

evidence integral via nested sampling...

.. math::
    Z_k = \int\nolimits_{\Theta_k} \mathcal{L}(\pmb{\theta}_k) \, \pi(\pmb{\theta}_k) \, \mathrm{d}\pmb{\theta}_k
    = \int\nolimits_{0}^{1} \mathcal{L}(X) \, \mathrm{d}X

where :math:`\Theta_k` denotes the entire parameter domain. and the second integral
is the one solved in nested sampling, introducing a prior mass :math:`X` sorted
by the likelihood (ref to dynesty, or methods in release paper). second integral
is reparametrised.

nested sampling also yields bona fide posterior parameter samples, when they
are weighted as :math:`p(\pmb{\theta}_i) = \mathcal{L}_i \, \Delta X_i / Z`, where :math:`i` indicates the
samples of the :math:`i`-th iteration of a nested sampling run. So use
`est.bay_est_samples_weighted` of an estimation instance `est` in MemoCell.

Bayesian-averaged output over entire model space

.. math::
    p(X|D) = \sum\nolimits_{k=1}^{m} \int\nolimits_{\Theta_k} \,
    p(X|\pmb{\theta}_k, M_k, D) \, p(\pmb{\theta}_k | M_k, D) \,
    p(M_k | D) \, \mathrm{d}\pmb{\theta}_k

where typically :math:`p(X|\pmb{\theta}_k, M_k, D)=p(X|\pmb{\theta}_k, M_k)`
(posterior model contains all info to compute :math:`X`). describe sampling
procedure, read eq. from right to left.

Subsampling from Compartments
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

In some experimental settings you may not observe the cell counts of the
biological process directly, but only a subsampled fraction of them. MemoCell
can still be applied in these settings; however the approximate percentage
of subsampling has to be known and one should apply a correction for the
subsampling (e.g., as below).

Let :math:`N` be the random cell numbers of the compartment (which we want to
know for MemoCell) and :math:`X` be the subsampled cell numbers (which we actually
have observed). For :math:`N` much larger than :math:`X`, the binomial distribution
can be used to model the sampling process (otherwise, the hypergeometric
distribution should be used); we have

.. math::
    X | N \sim \mathrm{Bin}(N, \alpha)

where :math:`\alpha \in (0, 1]` is the subsampling fraction. Then, the main idea is
to rescale the observed counts :math:`X` with the subsampling fraction :math:`\alpha`
and introduce

.. math::
    S = \frac{X}{\alpha}

as an estimate for :math:`N` for each cell type / variable of interest.

Based on the law of total expectation (and variance/covariance),
one can directly show relations for the mean

.. math::
    \mathrm{E}(N) = \mathrm{E}(S),

the variance

.. math::
    \mathrm{Var}(N) = \mathrm{Var}(S) - \frac{\alpha (1-\alpha)}{\alpha^2} \mathrm{E}(S)

and the covariance (between two different variables, each subsampled with
:math:`\alpha_1` and :math:`\alpha_2`, respectively)

.. math::
    \mathrm{Cov}(N_1, N_2) = \mathrm{Cov}(S_1, S_2).

These relations mean that the rescaled data correctly reflect the means and
covariances of the original cell counts, whereas the variance needs to be
corrected as above (to remove the additional noise caused by the subsampling,
right term on the rhs, from the biological variability, left term on the rhs).

`Example`: We measure samples of :math:`X` as :math:`x \in \{7, 11, 4\}`
with a subsampling fraction of 20 %, i.e. :math:`\alpha=0.2`. Then, realisations
of :math:`S` are :math:`s \in \{35, 55, 20\}` and estimates for mean and variance
of the rescaled data are :math:`\mathrm{E}(S)\approx 36.7`
and :math:`\mathrm{Var}(S) \approx 308.3` (`ddof=1`). Hence, the subsampling
corrected mean and variance estimates that we load to MemoCell are
:math:`\mathrm{E}(N) = \mathrm{E}(S) \approx 36.7` and
:math:`\mathrm{Var}(N) = \mathrm{Var}(S) -  \frac{\alpha (1-\alpha)}{\alpha^2} \mathrm{E}(S) \approx 161.7`.
