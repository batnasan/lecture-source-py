.. _aiyagari:

.. include:: /_static/includes/header.raw

.. highlight:: python3

*****************************
A von Neumann growth model
*****************************

and a generalization
---------------------

This notebook uses the class ``neumann`` to calculate key objects of a
linear growth model of John von Neumann (1937) that was generalized by
Kemeny, Moregenstern and Thompson (1956)

Objects of interest are the maximal expansion rate (``α``), the interest factor (``β``), and
the optimal intensities (``x``) and prices (``p``)

In addition to watching how the towering mind of John von Neumann
formulated an equilibrium model of price and quantity vectors in
balanced growth, this notebook shows how fruitfully to employ the
following important tools:

-  a zero-sum two player game

-  linear programming

-  the Perron-Frobenius theorem


.. code-block:: python3

    import numpy as np
    import matplotlib.pyplot as plt
    %matplotlib inline
    
    np.set_printoptions(precision=2)

The next few lines of code fetch the file ``neumann.py`` that does most
of the calculations

.. code-block:: python3

    from quantecon import fetch_nb_dependencies
    fetch_nb_dependencies(["neumann.py"])
    from neumann import neumann

**Notation:** We use the following notation. :math:`\mathbf{0}` denotes
a vector of zeros. We call an :math:`n`-vector - positive or
:math:`x\gg \mathbf{0}` if :math:`x_i>0` for all :math:`i=1,2,\dots,n`
- nonnegative or :math:`x\geq \mathbf{0}` if :math:`x_i\geq 0` for
all :math:`i=1,2,\dots,n` - semipositive or :math:`x > \mathbf{0}` if
:math:`x\geq \mathbf{0}` and :math:`x\neq \mathbf{0}`

For two comformable vectors :math:`x` and :math:`y`, :math:`x\gg y`,
:math:`x\geq y` and :math:`x> y` mean :math:`x-y\gg \mathbf{0}`,
:math:`x-y \geq \mathbf{0}`, and :math:`x-y > \mathbf{0}`

By default, all vectors are column vectors, :math:`x^{T}` denotes the
transpose of :math:`x` (i.e. a row vector). Let :math:`\iota_n` denote a
column vector composed of :math:`n` ones, i.e.
:math:`\iota_n = (1,1,\dots,1)^T`. Let :math:`e^i` denote the vector (of
arbitrary size) containing zeros except for the :math:`i` th position
where it is one.

We denote matrices by capital letters. For an arbitrary matrix
:math:`A`, :math:`a_{i,j}` represents the entry in its :math:`i` th
row and :math:`j` th column. :math:`a_{\cdot j}` and :math:`a_{i\cdot}`
denote the :math:`j` th column and :math:`i` th row of :math:`A`,
respectively.

Model ingredients and assumptions:
----------------------------------

A pair :math:`(A,B)` of :math:`m\times n` nonnegative matrices defines
an economy.

-  :math:`m` is the number of *activities* (or sectors)

-  :math:`n` is the number of *goods* (produced and/or used in the
   economy)

-  :math:`A` is called the *input matrix*; :math:`a_{i,j}` denotes the
   amount of good :math:`j` consumed by activity :math:`i`.

-  :math:`B` is called the *output matrix*; :math:`b_{i,j}` represents
   the amount of good :math:`j` produced by activity :math:`i`.

Two key assumptions restrict economy :math:`(A,B)`: - **Assumption I:**
(every good which is consumed is also produced)

.. math:: b_{.,j} > \mathbf{0}\hspace{5mm}\forall j=1,2,\dots,n

 - **Assumption II:** (no free lunch)

.. math:: a_{i,.} > \mathbf{0}\hspace{5mm}\forall i=1,2,\dots,m

A semipositive :math:`m`-vector:math:`x` denotes the levels at which
activities are operated (*intensity vector*).

Therefore,

-  vector :math:`x^TA` gives the total amount of *goods used in
   production*

-  vector :math:`x^TB` gives *total outputs*

An economy :math:`(A,B)` is said to be *productive*, if there exists a
nonnegative intensity vector :math:`x \geq 0` such
that :math:`x^T B > x^TA`.

The semipositive :math:`n`-vector :math:`p` contains prices assigned to
the :math:`n` goods.

The :math:`p` vector implies *cost* and *revenue* vectors

-  the vector :math:`Ap` tells *costs* of the vector of activities

-  .. rubric:: the vector :math:`Bp` tells *revenues* from the vector of
      activities
      :name: the-vector-bp-tells-revenues-from-the-vector-of-activities

A property of an input-output pair :math:`(A,B)` called *irreducibility*
(or indecomposability) determines whether an economy can be decomposed
into multiple ‘’sub-economies’’:

**Definition:** Given an economy :math:`(A,B)`, the set of goods
:math:`S\subset \{1,2,\dots,n\}` is called an *independent subset* if
it is possible to produce every good in :math:`S` without consuming any
good outside :math:`S`. Formally, the set :math:`S` is independent if
:math:`\exists T\subset \{1,2,\dots,m\}` (subset of activities) such
that :math:`a_{i,j}=0`, :math:`\forall i\in T` and :math:`j\in S^c` and
for all :math:`j\in S`, :math:`\exists i\in T`, s.t. :math:`b_{i,j}>0`.
The economy is **irreducible** if there are no proper independent
subsets.

We study two examples, both coming from Chapter 9.6 of Gale (1960).

.. code-block:: python3

    # (1) Irreducible (A, B) example: alpha_0 = beta_0
    A1 = np.array([[0, 1, 0, 0], 
                   [1, 0, 0, 1], 
                   [0, 0, 1, 0]])
    
    B1 = np.array([[1, 0, 0, 0], 
                   [0, 0, 2, 0], 
                   [0, 1, 0, 1]])
    
    # (2) Reducible (A, B) example: beta_0 < alpha_0 
    A2 = np.array([[0, 1, 0, 0, 0, 0], 
                   [1, 0, 1, 0, 0, 0], 
                   [0, 0, 0, 1, 0, 0], 
                   [0, 0, 1, 0, 0, 1], 
                   [0, 0, 0, 0, 1, 0]])
    
    B2 = np.array([[1, 0, 0, 1, 0, 0], 
                   [0, 1, 0, 0, 0, 0], 
                   [0, 0, 1, 0, 0, 0], 
                   [0, 0, 0, 0, 2, 0], 
                   [0, 0, 0, 1, 0, 1]])

The following code sets up our first Neumann economy or ``neumann``
instance

.. code-block:: python3

    N1 = neumann(A1, B1)
    N1




.. parsed-literal::

    
    Generalized von Neumann expanding model:
      - number of goods          : 4
      - number of activities     : 3
    
    Assumptions:
      - AI:  every column of B has a positive entry    : True
      - AII: every row of A has a positive entry       : True




.. code-block:: python3

    N2 = neumann(A2, B2)
    N2




.. parsed-literal::

    
    Generalized von Neumann expanding model:
      - number of goods          : 6
      - number of activities     : 5
    
    Assumptions:
      - AI:  every column of B has a positive entry    : True
      - AII: every row of A has a positive entry       : True




Dynamic interpretation
----------------------

Attach a time index :math:`t` to the preceding objects, regard aneconomy
as a dynamic system, and study sequences

.. math:: \{(A_t,B_t)\}_{t\geq 0}, \hspace{1cm}\{x_t\}_{t\geq 0},\hspace{1cm} \{p_t\}_{t\geq 0}

An interesting special case holds the technology process constant and
investigates the dynamics of quantities and prices only.

Accordingly, in the rest of this notebook we assume that
:math:`(A_t,B_t)=(A,B)` for all :math:`t\geq 0`.

--------------

A crucial element of the dynamic iterpretation involves the timing of
production.

We assume that production (consumption of inputs) takes place in period
:math:`t`, while the associated output materializes in period
:math:`t+1`, i.e. consumption of :math:`x_{t}^TA` in period :math:`t`
results in :math:`x^T_{t}B` amounts of output in period :math:`t+1`.

These timing conventions imply the following feasibiliy condition:

:raw-latex:`\begin{align}
x^T_{t}B \geq x^T_{t+1} A \hspace{1cm}\forall t\geq 1,
\end{align}`

which asserts that no more goods can be used today than were produced
yesterday.

Accordingly, :math:`Ap_t` tells the costs of production in period
:math:`t` and :math:`Bp_t` tells revenues in period :math:`t+1`.

Balanced growth
---------------

We follow John von Neumann in studying “balanced growth”

Let ``./`` denote elementwise division of one vector by another and let
:math:`\alpha >0` be a scalar.

Then *balanced growth* is a situation in which

.. math:: x_{t+1}./x_t = \alpha , \quad \forall t \geq 0

With balanced growth, the law of motion of :math:`x` is evidently $
x_{t+1}=:raw-latex:`\alpha `x_t$ and so we can rewrite the feasibility
constraint as

.. math:: x^T_{t}B \geq \alpha x^T_t A \hspace{1cm}\forall t

In the same spirit, define :math:`\beta\in\mathbb{R}` as the **interest
factor** per unit of time.

We assume that it is always possible to earn a gross return equal to the
constant interest factor :math:`\beta` by investing “outside the model”.

Under this assumption about outside investment opportunities, a
no-arbitrage condition gives rise to the following (no profit)
restriction on the price sequence:

.. math:: \beta Ap_{t} \geq B p_{t} \hspace{1cm}\forall t

This says that production cannot yield a return greater than that
offered by the investment opportunity (note that we compare values in
period :math:`t+1`).

The balanced growth assumption allows us to drop time subscripts and
conduct an analysis purely in terms of a time-invariant growth rate
:math:`\alpha` and interest factor :math:`\beta`

Duality
-------

The following two problems are connected by a remarkable dual
relationship between the technological and valuation characteristics of
the economy:

**Definition:** The *technological expansion problem* (TEP) for economy
:math:`(A,B)` is to find a semipositive :math:`m`-vector :math:`x>0`
and a number :math:`\alpha\in\mathbb{R}`, s.t.

:raw-latex:`\begin{align*}
    &\max_{\alpha} \hspace{2mm} \alpha\\
    &\text{s.t. }\hspace{2mm}x^T B \geq \alpha x^T A
\end{align*}`

Theorem 9.3 of David Gale’s book assets that if Assumptions I and II are
both satisfied, then a maximum value of :math:`\alpha` exists and it is
positive. It is called the *technological expansion rate* and is denoted
by :math:`\alpha_0`. The associated intensity vector :math:`x_0` is the
*optimal intensity vector*.

--------------

**Definition:** The *economical expansion problem* (EEP) for
:math:`(A,B)` is to find a semipositive :math:`n`-vector :math:`p>0`
and a number :math:`\beta\in\mathbb{R}`, such that

:raw-latex:`\begin{align*}
    &\min_{\beta} \hspace{2mm} \beta\\
    &\text{s.t. }\hspace{2mm}Bp \leq \beta Ap
\end{align*}`

Assumptions I and II imply existence of a minimum value
:math:`\beta_0>0` called the *economic expansion rate*. The
corresponding price vector :math:`p_0` is the *optimal price vector*.

--------------

Evidently, the criterion functions in *technological expansion* problem
and the *economical expansion problem* are both linearly homogeneous, so
the optimality of :math:`x_0` and :math:`p_0` are defined only up to a
positive scale factor.

For simplicity (and to emphasize a close connection to zero-sum games),
in the following, we normalize both vectors
:math:`x_0` and :math:`p_0` to have unit length.

A standard duality argument (see Lemma 9.4. in (Gale, 1960)) implies
that under Assumptions I and II, :math:`\beta_0\leq \alpha_0`

But in the other direction, that is :math:`\beta_0\geq \alpha_0`,
Assumptions I and II are not sufficient.

Nevertheless, von Neumann (1937) proved the following remarkable
“duality-type” result connecting TEP and EEP.

**Theorem 1 (von Neumann):** If the economy :math:`(A,B)` satisfies
Assumptions I and II, then there exists a set
:math:`\left(\gamma^{*}, x_0, p_0\right)`, where
:math:`\gamma^{*}\in[\beta_0, \alpha_0]\subset\mathbb{R}`, :math:`x_0>0`
is an :math:`m`-vector, :math:`p_0>0` is an :math:`n`-vector and the
following holds true

:raw-latex:`\begin{align*}
x_0^T B &\geq \gamma^{*} x_0^T A \\
Bp_0 &\leq \gamma^{*} Ap_0 \\
x_0^T\left(B-\gamma^{*} A\right)p_0 &= 0
\end{align*}`

   *Proof (Sketch):* Assumption I and II imply that there exist
   :math:`(\alpha_0, x_0)` and :math:`(\beta_0, p_0)` solving the TEP
   and EEP, repspectively. If :math:`\gamma^*>\alpha_0`, then by
   defintion of :math:`\alpha_0`, there cannot exist a semipositive
   :math:`x` that satisfies :math:`x^T B \geq \gamma^{*} x^T A`.
   Similarly, if :math:`\gamma^*<\beta_0`, there is no semipositive
   :math:`p` so that :math:`Bp \leq \gamma^{*} Ap`. Let
   :math:`\gamma^{*}\in[\beta_0, \alpha_0]`, then
   :math:`x_0^T B \geq \alpha_0 x_0^T A \geq \gamma^{*} x_0^T A`.
   Moreover, :math:`Bp_0\leq \beta_0 A p_0\leq \gamma^* A p_0`. This two
   inequalities imply :math:`x_0\left(B - \gamma^{*} A\right)p_0 = 0`.

Here the constant :math:`\gamma^{*}` is both expansion and interest
factor (not neccessarily optimal). We have already encountered and
discussed the first two inequalities that represent feasibility and
no-profit conditions. Moreover, the equality compactly captures the
requirements that if any good grows at a rate larger than
:math:`\gamma^{*}` (i.e., if it is ‘’oversupplied’’), then its price
must be zero; and that if any activity provides negative profit, it must
be unused. Therefore, these expressions encode all equilbrium conditions
and Theorem I essentially states that under Assumptions I and II there
always exists an equilibrium :math:`\left(\gamma^{*}, x_0, p_0\right)`
with balanced growth.

Note that Theorem I is silent about uniqueness of the equilibrium. In
fact, it does not rule out (trivial) cases with :math:`x_0^TBp_0 = 0` so
that nothing of value is produced. To exclude such uninteresting cases,
Kemeny, Morgenstern and Thomspson (1956) add an extra requirement

.. math:: x^T_0 B p_0 > 0

and call the resulting equilibria *economic solutions*. They show that
this extra condition does not affect the existence result, while it
significantly reduces the number of (relevant) solutions.

--------------

Interpretation as a game theoretic problem (two-player zero-sum game)
---------------------------------------------------------------------

To compute the equilibrium :math:`(\gamma^{*}, x_0, p_0)`, we follow the
algorithm proposed by Hamburger, Thompson and Weil (1967), building on
the key insight that the equilibrium (with balanced growth) can be
considered as a solution of a particular two-player zero-sum game.
First, we introduce some notations.

Consider the :math:`m\times n` matrix :math:`C` as a payoff matrix,
with the entries representing payoffs from the **minimizing** column
player to the **maximizing** row player and assume that the players can
use mixed strategies: - row player chooses the :math:`m`-vector
:math:`x > \mathbf{0}`, s.t. :math:`\iota_m^T x = 1` - column player
chooses the :math:`n`-vector :math:`p > \mathbf{0}`,
s.t. :math:`\iota_n^T p = 1`

**Definition:** The :math:`m\times n` matrix game :math:`C` has the
*solution* :math:`(x^*, p^*, V(C))` in mixed strategies, if

:raw-latex:`\begin{align}
(x^*)^T C e^j \geq V(C)\quad \forall j\in\{1, \dots, n\}\quad \quad \text{and}\quad\quad (e^i)^T C p^* \leq V(C)\quad \forall i\in\{1, \dots, m\}
\end{align}`

The number :math:`V(C)` is called the *value* of the game.

From the above definition, it is clear that the value :math:`V(C)` has
two alternative interpretations: \* by playing the appropriate mixed
stategy, the maximizing player can assure himself at least :math:`V(C)`
(no matter what the column player chooses) \* by playing the appropriate
mixed stategy, the minimizing player can make sure that the maximizing
player will not get more than :math:`V(C)` (irrespective of what is the
maximizing player’s choice)

From the famous theorem of Nash (1951), it follows that there always
exists a mixed strategy Nash equilibrium for any *finite* two-player
zero-sum game. Moreover, von Neumann’s Minmax Theorem (1928) implies
that

.. math:: V(C) = \max_x \min_p \hspace{2mm} x^T C p = \min_p \max_x \hspace{2mm} x^T C p = (x^*)^T C p^*

Connection with Linear Programming (LP)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Finding Nash equilibria of a finite two-player zero-sum game can be
formulated as a linear programming problem. To see this, we introduce
the following notation - For a fixed :math:`x`, let :math:`v` be the
value of the minimization problem:
:math:`v \equiv \min_p x^T C p = \min_j x^T C e^j` - For a fixed
:math:`p`, let :math:`u` be the value of the maximization problem:
:math:`u \equiv \max_x x^T C p = \max_i (e^i)^T C p`

Then the *max-min problem* (the game from the maximizing player’s point
of view) can be written as the *primal* LP

:raw-latex:`\begin{align}
V(C) = & \max \hspace{2mm} v \\
\text{s.t. } \hspace{2mm} v \iota_n^T &\leq x^T C  \\
x &\geq \mathbf{0} \\
\iota_n^T x & = 1
\end{align}`

while the *min-max problem* (the game from the minimizing player’s point
of view) is the *dual* LP

:raw-latex:`\begin{align}
V(C) = &\min \hspace{2mm} u \\
\text{s.t. } \hspace{2mm}u \iota_m &\geq Cp  \\
p &\geq \mathbf{0} \\
\iota_m^T p & = 1
\end{align}`

--------------

Hamburger, Thompson and Weil (1967) view the input-output pair of the
economy as payoff matrices of two-player zero-sum games. Using this
interpretation, they restate Assumption I and II as follows

.. math:: V(-A) < 0\quad\quad \text{and}\quad\quad V(B)>0

   *Proof (Sketch)*: \* :math:`\Rightarrow` :math:`V(B)>0` implies
   :math:`x_0^T B \gg \mathbf{0}`, where :math:`x_0` is a maximizing
   vector. Since :math:`B` is non-negative, this requires that each
   column of :math:`B` has at least one positive entry, which is
   Assumption I. \* :math:`\Leftarrow` From Assumption I and the fact
   that :math:`p>\mathbf{0}`, it follows that :math:`Bp > \mathbf{0}`.
   This implies that the maximizing player can always choose :math:`x`
   so that :math:`x^TBp>0`, that is it must be the case
   that :math:`V(B)>0`.

In order to (re)state Theorem I in terms of a particular two-player
zero-sum game, we define the matrix for :math:`\gamma\in\mathbb{R}`

.. math:: M(\gamma) \equiv B - \gamma A

For fixed :math:`\gamma`, treating :math:`M(\gamma)` as a matrix game,
we can calculate the solution of the game.

-  If :math:`\gamma > \alpha_0`, then for all :math:`x>0`, there
   :math:`\exists j\in\{1, \dots, n\}`, s.t.
   :math:`[x^T M(\gamma)]_j < 0` implying
   that :math:`V(M(\gamma)) < 0`
-  If :math:`\gamma < \beta_0`, then for all :math:`p>0`, there
   :math:`\exists i\in\{1, \dots, m\}`, s.t.
   :math:`[M(\gamma)p]_i > 0` implying that :math:`V(M(\gamma)) > 0`
-  If :math:`\gamma \in \{\beta_0, \alpha_0\}`, then (by Theorem I) the
   optimal intensity and price vectors :math:`x_0` and :math:`p_0`
   satisfy

:raw-latex:`\begin{align}
x_0^T M(\gamma) \geq \mathbf{0}^T \quad \quad \text{and}\quad\quad M(\gamma) p_0 \leq \mathbf{0} 
\end{align}`

That is, :math:`(x_0, p_0, 0)` is a solution of the game
:math:`M(\gamma)` so
that :math:`V\left(M(\beta_0)\right) = V\left(M(\alpha_0)\right) = 0`.
\* If :math:`\beta_0 < \alpha_0` and
:math:`\gamma \in (\beta_0, \alpha_0)`, then :math:`V(M(\gamma)) = 0`.
Moreover, if :math:`x'` is optimal for the maximizing player in
:math:`M(\gamma')` for :math:`\gamma'\in(\beta_0, \alpha_0)` and
:math:`p''` is optimal for the minimizing player in :math:`M(\gamma'')`
where :math:`\gamma''\in(\beta_0, \gamma')`, then :math:`(x', p'', 0)`
is a solution for
:math:`M(\gamma)`, :math:`\forall \gamma\in (\gamma'', \gamma')`. >
*Proof (Sketch):* If :math:`x'` is optimal for a maximizing player in
game :math:`M(\gamma')`, then :math:`(x')^T M(\gamma')\geq \mathbf{0}^T`
and so for all :math:`\gamma<\gamma'`

.. math:: (x')^T M(\gamma) = (x')^T M(\gamma') + (x')^T(\gamma' - \gamma)A \geq \mathbf{0}^T

 hence :math:`V(M(\gamma))\geq 0`. If :math:`p''` is optimal for a
minimizing player in game :math:`M(\gamma'')`, then $
M(:raw-latex:`\gamma`’‘)p’’:raw-latex:`\leq `:raw-latex:`\mathbf{0}`$
and so for all :math:`\gamma''<\gamma`

.. math:: M(\gamma)p'' = M(\gamma'') + (\gamma'' - \gamma)Ap'' \leq \mathbf{0}

 hence :math:`V(M(\gamma))\leq 0`.

It is clear from the above argument that :math:`\beta_0`,
:math:`\alpha_0` are the minimal and maximal :math:`\gamma` for which
:math:`V(M(\gamma))=0`. Moreover, Hamburger et al. (1967) show that the
function :math:`\gamma \mapsto V(M(\gamma))` is continuous and
nonincreasing in :math:`\gamma`. This suggests an algorithm to compute
:math:`(\alpha_0, x_0)` and :math:`(\beta_0, p_0)` for a given
input-output pair :math:`(A, B)`.

Algorithm
---------

Hamburger, Thompson and Weil (1967) propose a simple bisection algorithm
to find the minimal and maximal roots (i.e. :math:`\beta_0` and
:math:`\alpha_0`) of the function :math:`\gamma \mapsto V(M(\gamma))`.

Step 1
~~~~~~

First, notice that we can easily find trivial upper and lower bounds for
:math:`\alpha_0` and :math:`\beta_0`. \* TEP requires that
:math:`x^T(B-\alpha A)\geq \mathbf{0}^T` and :math:`x > \mathbf{0}`, so
if :math:`\alpha` is so large that
:math:`\max_i\{[(B-\alpha A)\iota_n]_i\} < 0`, then TEP ceases to have a
solution. Accordingly, let **``UB``** be the :math:`\alpha^{*}` that
solves :math:`\max_i\{[(B-\alpha^{*} A)\iota_n]_i\} = 0`. \* Similar to
the upper bound, if :math:`\beta` is so low that
:math:`\min_j\{[\iota^T_m(B-\beta A)]_j\}>0`, then the EEP has no
solution and so we can define **``LB``** as the :math:`\beta^{*}` that
solves :math:`\min_j\{[\iota^T_m(B-\beta^{*} A)]_j\}=0`.

The ``bounds`` method calculates these trivial bounds for us

.. code-block:: python3

    N1.bounds()




.. parsed-literal::

    (1.0, 2.0)



Step 2
~~~~~~

Compute :math:`\alpha_0` and :math:`\beta_0`

-  Finding :math:`\alpha_0`

   1. Fix :math:`\gamma = \frac{UB + LB}{2}` and compute the solution
      of the two-player zero-sum game associated
      with :math:`M(\gamma)`. We can use either the primal or the dual
      LP problem.
   2. If :math:`V(M(\gamma)) \geq 0`, then set :math:`LB = \gamma`,
      otherwise let :math:`UB = \gamma`
   3. Iterate on 1. and 2. until :math:`|UB - LB| < \epsilon`

-  Finding :math:`\beta_0`

   1. Fix :math:`\gamma = \frac{UB + LB}{2}` and compute the solution
      of the two-player zero-sum game associated
      with :math:`M(\gamma)`. We can use either the primal or the dual
      LP problem.
   2. If :math:`V(M(\gamma)) > 0`, then set :math:`LB = \gamma`,
      otherwise let :math:`UB = \gamma`
   3. Iterate on 1. and 2. until :math:`|UB - LB| < \epsilon`

..

   *Existence*: Since :math:`V(M(LB))>0` and :math:`V(M(UB))<0` and
   :math:`V(M(\cdot))` is a continuous, nonincreasing function, there is
   at least one :math:`\gamma\in[LB, UB]`, s.t. :math:`V(M(\gamma))=0`.

The ``zerosum`` method calculates the value and optimal strategies
assocaited with a given :math:`\gamma`

.. code-block:: python3

    gamma = 2
    
    print('Value of the game with gamma = {}'.format(gamma))
    print(N1.zerosum(gamma = gamma)[0])               
    print('Intensity vector (from the primal)')
    print(N1.zerosum(gamma = gamma)[1])               
    print('Price vector (from the dual)')
    print(N1.zerosum(gamma = gamma, dual = True)[1])


.. parsed-literal::

    Value of the game with gamma = 2
    -0.24
    Intensity vector (from the primal)
    [ 0.32  0.28  0.4 ]
    Price vector (from the dual)
    [ 0.4   0.32  0.28  0.  ]


.. code-block:: python3

    numb_grid = 100
    gamma_grid = np.linspace(0.4, 2.1, numb_grid)
    
    value_ex1_grid = np.asarray([N1.zerosum(gamma = gamma_grid[i])[0] for i in range(numb_grid)])
    value_ex2_grid = np.asarray([N2.zerosum(gamma = gamma_grid[i])[0] for i in range(numb_grid)])
    
    fig, ax = plt.subplots(1, 2, figsize = (14, 5), sharey = True)
    fig.suptitle(r'The function $V(M(\gamma))$', fontsize = 16)
    
    ax[0].plot(gamma_grid, value_ex1_grid, lw = 2)
    ax[0].set_title(r'Example 1', fontsize = 15)
    ax[0].axhline(0, color = 'k', lw =1)
    ax[0].set_xlabel(r'$\gamma$', fontsize = 14)
    ax[0].axvline(N1.bounds()[0], color = 'r', lw = 1, linestyle = '--', label = 'lower bound')
    ax[0].axvline(N1.bounds()[1], color = 'g', lw = 1, linestyle = '--', label = 'upper bound')
    ax[0].legend(loc = 'best')
    
    ax[1].plot(gamma_grid, value_ex2_grid, lw = 2)
    ax[1].set_title(r'Example 2', fontsize = 15)
    ax[1].axhline(0, color = 'k', lw =1)
    ax[1].set_xlabel(r'$\gamma$', fontsize = 14)
    ax[1].axvline(N2.bounds()[0], color = 'r', lw = 1, linestyle = '--', label = 'lower bound')
    ax[1].axvline(N2.bounds()[1], color = 'g', lw = 1, linestyle = '--', label = 'upper bound')
    ax[1].legend(loc = 'best')




.. parsed-literal::

    <matplotlib.legend.Legend at 0x7f94bb78a438>




.. image:: output_22_1.png


The ``expansion`` method implements the bisection algorithm for
:math:`\alpha_0` (and uses the primal LP problem for :math:`x_0`)

.. code-block:: python3

    alpha0, x, p = N1.expansion()
    print('alpha_0 = {}'.format(alpha0)) 
    print('x_0 = {}'.format(x))
    print('The corresponding p from the dual = {}'.format(p))


.. parsed-literal::

    alpha_0 = 1.2599210478365421
    x_0 = [ 0.33  0.26  0.41]
    The corresponding p from the dual = [ 0.41  0.33  0.26  0.  ]


The ``interest`` method implements the bisection algorithm for
:math:`\beta_0` (and uses the dual LP problem for :math:`p_0`)

.. code-block:: python3

    beta0, x, p = N1.interest()
    print('beta_0 = {}'.format(beta0)) 
    print('p_0 = {}'.format(p))
    print('The corresponding x from the primal = {}'.format(x))


.. parsed-literal::

    beta_0 = 1.2599210478365421
    p_0 = [ 0.41  0.33  0.26  0.  ]
    The corresponding x from the primal = [ 0.33  0.26  0.41]


Of course, when :math:`\gamma^*` is unique, it is irrelevant which one
of the two methods we use. In particular, as will be shown below, in
case of an irreducible :math:`(A,B)` (like in Example 1), the maximal
and minimal roots of :math:`V(M(\gamma))` necessarily coincide implying
a ‘’full duality’’ result, i.e. :math:`\alpha_0 = \beta_0 = \gamma^*`,
and that the expansion (and interest) rate :math:`\gamma^*` is unique.

Uniqueness and irreducibility
-----------------------------

As an illustration, compute first the maximal and minimal roots of
:math:`V(M(\cdot))` for Example 2, which displays a reducible
input-output pair :math:`(A, B)`.

.. code-block:: python3

    alpha0, x, p = N2.expansion()
    print('alpha_0 = {}'.format(alpha0)) 
    print('x_0 = {}'.format(x))
    print('The corresponding p from the dual = {}'.format(p))


.. parsed-literal::

    alpha_0 = 1.2528658034279943
    x_0 = [ 0.    0.    0.33  0.26  0.41]
    The corresponding p from the dual = [ 0.56  0.44  0.    0.    0.    0.  ]


.. code-block:: python3

    beta0, x, p = N2.interest()
    print('beta_0 = {}'.format(beta0)) 
    print('p_0 = {}'.format(p))
    print('The corresponding x from the primal = {}'.format(x))


.. parsed-literal::

    beta_0 = 1.0000000009313226
    p_0 = [ 0.5  0.5  0.   0.   0.   0. ]
    The corresponding x from the primal = [ 0.33  0.33  0.33  0.    0.  ]


As we can see, with a reducible :math:`(A,B)`, the roots found by the
bisection alhorithms might differ, so there might be multiple
:math:`\gamma^*` that make the value of the game
with :math:`M(\gamma^*)` zero. (see the figure above)

Indeed, although the von Neumann theorem assures existence of the
equilibrium, Assumptions I and II are not sufficient for uniqueness.
Nonetheless, Kemeny et al. (1967) show that there are at most finitely
many economic solutions, meaning that there are only finitely many
:math:`\gamma^*` that satisfy :math:`V(M(\gamma^*)) = 0` and
:math:`x_0^TBp_0 > 0` and that for each such :math:`\gamma^*_i`, there
is a self-sufficient part of the economy (a sub-economy) that in
equilibrium can expand independently with the expansion
coefficient :math:`\gamma^*_i`.

The following theorem (see Theorem 9.10. in Gale, 1960) asserts that
imposing irreducibility is sufficient for uniqueness of
:math:`(\gamma^*, x_0, p_0)`.

**Theorem II:** Consider the conditions of Theorem 1. If the economy
:math:`(A,B)` is irreducible, then :math:`\gamma^*=\alpha_0=\beta_0`.

A special case
--------------

There is a special :math:`(A,B)` that allows us to simplify the solution
method significantly by invoking the powerful Perron-Frobenius theorem
for nonnegative matrices.

**Definition:** We call an economy *simple* if it satisfies 1.
:math:`n=m` 2. Each activity produces exactly one good 3. Each good is
produced by one and only one activity

These assumptions imply that :math:`B=I_n`, i.e., that :math:`B` can be
written as an identity matrix (possibly after reshuffling its rows and
columns).

The simple model has the following special property (Theorem 9.11. in
Gale): if :math:`x_0` and :math:`\alpha_0>0` solve the TEP
with :math:`(A,I_n)`, then

.. math:: x_0^T = \alpha_0 x_0^T A\hspace{1cm}\Leftrightarrow\hspace{1cm}x_0^T A=\left(\frac{1}{\alpha_0}\right)x_0^T

The latter shows that :math:`1/\alpha_0` is a positive eigenvalue of
:math:`A` and :math:`x_0` is the correponding nonnegative left
eigenvector. The classical result of **Perron and Frobenius** implies
that a nonnegative matrix always has a nonnegative
eigenvalue-eigenvector pair. Moreover, if :math:`A` is irreducible, then
the optimal intensity vector :math:`x_0` is positive and *unique* up to
multiplication by a positive scalar.

Suppose that :math:`A` is reducible with :math:`k` irreducible subsets
:math:`S_1,\dots,S_k`. Let :math:`A_i` be the submatrix corresponding to
:math:`S_i` and let :math:`\alpha_i` and :math:`\beta_i` be the
associated expansion and interest factors, respectively. Then we have

.. math:: \alpha_0 = \max_i \{\alpha_i\}\hspace{1cm}\text{and}\hspace{1cm}\beta_0 = \min_i \{\beta_i\}

References
^^^^^^^^^^

Gale, David, **The Theory of Linear Economic Models**, New York:
McGraw-Hill Book Company, 1960.

Hamburger, Michael J., Gerald L. Thompson and Roman L. Weil, Jr., 1967,
“Computation of Expansion Rates for the Generalized von Neumann Model of
an Expanding Economy”, Econometrica Vol. 35, No. 3/4, pp. 542-547

Kemeny, John G.; Morgenstern, Oskar; Thompson, Gerald L., 1956, “A
generalization of the von Neumann model of an expanding economy”.
Econometrica. 24. pp. 115–135.

Nash, John. 1951. “Non-Cooperative Games.” Annals of Mathematics, Second
Series, 54, no. 2, pp 286-95.

von Neumann, John, 1928, “Zur theories der gesellschaftsspiele”.
Mathematische Annalen 100, pp 295–320. English translation by S.
Bergmann in Contributions to the Theory of Games IV, ed. R.D. Luce and
A.W. Tucker. Princeton: Princeton University Press, 1959

von Neumann, John, 1937, “Über ein ökonomisches Gleichuns-system und
eine Verallgeneinerung des Brouwerschen Fixpunksatzes,” Ergebnisse eines
mathematischen Kolloquium, 8, , pp. 73-83, translated as “A Model of
General Economic Equilibrium,” Review of Economic Studies, 13. 1945,
pp. 1-9.

