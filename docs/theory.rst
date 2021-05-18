
Theory
======

Waiting Times and Memory
^^^^^^^^^^^^^^^^^^^^^^^^
A single cell spends a stochastic amount of time to execute a division, death
or differentiation event -- the so-called waiting time :math:`\tau`. The
distribution of waiting times in cell populations, as an ensemble of many
single-cell fates, hence quantifies the probabilistic dynamics of basic
cellular decisions. Subsequently, we will outline the theoretical concepts
employed in MemoCell to describe and infer stochastic processes of multiple
possibly interlinked waiting time distributions.

We introduce waiting time distributions from the "view" of a single cell
executing a single reaction (e.g., a differentiation event). The next section
then generalises the concepts for an ensemble of cells, also being subject
to cellular pathways of multiple reactions.

The most basic waiting time distribution is the exponential distribution
(see Figure below); :math:`\tau \sim \mathrm{Exp}(\theta)` where
:math:`\theta` is the rate parameter. The mean waiting time is the inverse of
the rate, :math:`\mathrm{E}(\tau)=1/\theta`, and the variance is
:math:`\mathrm{Var}(\tau)=1/\theta^2`. This also implies that the variability of
exponential waiting times is always fixed, as measured by the coefficient of
variation;
:math:`\mathrm{CV}(\tau)=\frac{\sqrt{\mathrm{Var}(\tau)}}{\mathrm{E}(\tau)} = 1`.

.. image:: ../images/wtd_exp.png
    :align: center
    :scale: 13 %

The exponential distribution is the only continuous distribution that fulfils
the property of memorylessness, meaning
:math:`p(\tau > t + s | \tau > s) = p(\tau > t)`. This is also why stochastic
processes that fulfil the Markov property of memorylessness are characterised
by exponential waiting times for all their transition events. Markov jump
processes with exponential waiting times allow powerful analytical access (will
be used and shown). However, the exponential waiting time distribution in
itself is often not a good assumption for biological transitions. Applied to
cell division, one would assume that the next division event is most likely to
occur immediately after the previous division (the mode is at 0); contradicting
the fact that cells need a minimal time span to replicate the DNA etc.

For this reason MemoCell allows to describe and infer more general,
non-exponential waiting time distributions -- specifically the Erlang
and phase-type distributions. Indeed, phase-type distributions are "dense"
(in the mathematical sense) in the field of all positive-valued distributions;
as such, they can `approximate any` waiting time distribution `arbitrarily closely`
(:math:`\color{red}{\text{CITE}}`; Bladt and Nielsen, 2017; Schassberger, 1973).
Importantly, these distributions are constructed by convolutions or mixtures
of exponential distributions and thus we retain the analytical tractability.

continue with Erlang...

for PH2 maybe mention that this is now a distribution with CV <1 and >1
is possible (maybe check that from the jupyter notebook where I did it)

give some motivation?

this time maybe start more direct with the reaction types

for 3-step Erlang maybe write down states explicitly: (1,0,0)->(0,1,0)->(0,0,1)
-> absorbing state

maybe start with single cell view and waiting time description... more intuitive
deliver the generator description later on and maybe only then write down
phase-type formulas

maybe focus here in this first section on the "single cell view"; say in next
section that this is can applied automatically for many cells running through the
transition graphs and networks

rates we introduce are the rates of single cells passing through such schemes

maybe start with exponential and some words on memory;
mention :math:`\mathrm{CV}(\tau)=\frac{\sqrt{\mathrm{Var}(\tau)}}{\mathrm{E}(\tau)} = 1`
for exponentially distributed :math:`\tau \sim \mathrm{Exp}(\theta)`
(rate notation, or call this q?)

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
    \partial_t \,\pmb{p}(w, t) = \pmb{p}(w, t) \, Q

where :math:`Q` is the (possibly infinite sized) transition rate (or generator)
matrix. describe :math:`Q`; also note that ME is sometimes written in transposed
form. with fundamental solution

.. math::
    \pmb{p}(w, t) = \pmb{p}(w_0, t_0) \, \mathrm{exp}\big(Q\,(t-t_0)\big)

however, often hard/impossible to solve.



reaction types; maybe change notation as in methods with :math:`W^{(i)}` and
:math:`W^{(k)}` variables.

- :math:`S \rightarrow E` (cell differentiation),

- :math:`S \rightarrow S + S` (symmetric self-renewing division),

- :math:`S \rightarrow S + E` (asymmetric division),

- :math:`S \rightarrow E + E` (symmetric differentiating division),

- :math:`S \rightarrow` (efflux or cell death) and

- :math:`\rightarrow E` (influx or birth),

some ph formulas; maybe at least mention that closed-form formulas exist
for mean and variance; also mention ph generally not unique, i.e. multiple
representation can yield the same distribution/pdf

if :math:`\pmb{p}(w, t) = \big(p(w_1, t), ..., p(w_m, t) \big)` of the :math:`m`
transient states then the absorbing probability is; :math:`\tau` is waiting time
to reach absorbing state :math:`w_\infty`)

.. math::
    p(\tau > t) = 1 - p(w_\infty, t) = \pmb{p}(w, t) \, \pmb{1}

where :math:`\pmb{p}(w, t) \, \pmb{1}` is the probability of being
in any of the transient states; 1 column vector of ones.

We can write the generator matrix :math:`Q` (size :math:`(m+1)\times (m+1)`) as

.. math::
    Q =
    \begin{pmatrix}
    0 & \pmb{0} \\
    \pmb{s} & S
    \end{pmatrix}

where :math:`S` is the generator matrix (size :math:`m\times m`) of the
transient states only and :math:`\pmb{s} = - S \, \pmb{1}`, :math:`\pmb{0}`
row vector of zeros.

Then by matrix exponential (can be solved for these finite systems)

.. math::
    F_{\mathrm{PH}}(t) = 1 - p(\tau > t) = 1 - \pmb{p}(w, t) \, \pmb{1}
    = 1 - \pmb{\alpha} \, \mathrm{exp}\big(S\,t\big) \, \pmb{1}

where :math:`\pmb{\alpha} = \pmb{p}(w_0, 0)` initial probs at :math:`t_0=0`. This
also directly implies (pdf)

.. math::
    f_{\mathrm{PH}}(t) = \pmb{\alpha} \, \mathrm{exp}\big(S\,t\big) \, \pmb{s}
    \qquad \Leftrightarrow \qquad \tau \sim \mathrm{PH}(\pmb{\alpha}, S)


make S and alpha generator example for 3-step Erlang, to introduce
notation a bit better

.. math::
    S =
    \begin{pmatrix}
    -3\theta & 3\theta  & 0 \\
    0 & -3\theta & 3\theta \\
    0 & 0 & -3\theta
    \end{pmatrix}

with :math:`\pmb{\alpha}=(1,0,0)` for an :math:`(\theta, n)`-Erlang
channel, i.e. :math:`\tau \sim \mathrm{Erl}(n ,n \theta)`
(shape and rate notation), with :math:`n=3`.
implying :math:`\mathrm{E}(\tau)=1/\theta` and
:math:`\mathrm{Var}(\tau)=1/(n \theta^2)`,
also meaning :math:`\mathrm{CV}(\tau)=1/\sqrt{n} \approx 0.58`.

maybe add most important references here too (or at least books
for further reading?)...
Bladt, Erlang (?), McBay for Bayes, unique ph, erlang cv, ph dense,
maybe this yates (?) ref where they describe many ph schemes

parameters for this:
n_list = [1, 2, 3, 5, 8, 13, 21, 34]
theta_all = 0.1
times = np.linspace(0.0, 20.0, num=200)

.. image:: ../images/net_scheme_erl3.png
    :align: center
    :scale: 14 %

.. image:: ../images/wtd_erl_many.png
    :align: center
    :scale: 16 %

hi hi

parameters for this:
n_d4 = 4
n_d2 = 2
theta_d4 = 0.03
theta_d2 = 0.04
act_times = np.linspace(0.0, 90.0)

.. image:: ../images/net_scheme_ph2_2_4.png
    :align: center
    :scale: 16 %

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
at a main variable, we split at its centric hidden variable), or even show a
scheme (similar to schemes above for Erlang and PH2 channels); maybe also say
that other thing than this competition can be implemented via simulation variables
for example minimum or maximum between two channels (ph closed under order
statistics).

hidden and main variables?

.. image:: ../images/net_scheme_multi.png
    :align: center
    :scale: 22 %

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
procedure, read eq. from right to left; maybe also add topology
inference of an application of this

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
