## Optimization Techniques

### Weighted Least-Squares Fitting Problem

To determine the optimal expansion coefficients $c_{x(c),ij}$ in our functional form, we formulate the parameter optimization as a linear least-squares fitting problem. This approach allows us to fit the exchange and correlation energies calculated from our functional to a set of reference data, minimizing the overall deviation.

For a given set of training data comprising $K$ systems or properties (e.g., total energies, reaction energies, dipole moments, etc.), we aim to minimize the loss function $L$, defined as the sum of weighted squared deviations between the calculated and reference values:

$$
L = \sum_{k=1}^{K} \dfrac{1}{2} w_k \left[ E_k^{\text{calc}} - E_k^{\text{ref}} \right]^2,
$$

where $E_k^{\text{calc}}$ is the reaction energy calculated using COACH for data point $k$, $E_k^{\text{ref}}$ is the corresponding reference energy from high-level calculations or experimental data, and $w_k$ is a weight assigned to data point $k$, reflecting its importance in the fitting procedure.

To simplify the optimization into a linear least-squares problem, we construct the COACH energy based on the energy ($E^{\text{base}}$) and density matrix of a base functional:

$$
E^{\text{calc}} = E^{\text{base}} + \Delta E_{xc}.
$$

For example, we use $\omega$B97X-V as the base functional in this work and retain the same VV10 energy as the dispersion correction:

$$
\begin{aligned}
\Delta E_{xc} &= E_{xc}^{\text{calc}} - E_{xc}^{\omega\text{B97X-V}} \\
&= E_{x,sr}^{\text{SL,calc}} + c_x^{\text{HF,calc}} E_{x,\text{sr}}^{\text{HF}} + E_{c,ss}^{\text{SL,calc}} + E_{c,os}^{\text{SL,calc}} \\
&\quad - \left(E_{x,sr}^{\text{SL,}\omega\text{B97X-V}} + c_x^{\text{HF,}\omega\text{B97X-V}} E_{x,\text{sr}}^{\text{HF}} + E_{c,ss}^{\text{SL,}\omega\text{B97X-V}} + E_{c,os}^{\text{SL,}\omega\text{B97X-V}} \right).
\end{aligned}
$$

Here, the SL terms can be expressed generally as:

$$
E^{\text{SL,calc}} = \sum_{i=0}^{M} \sum_{j=0}^{N} c_{x(c),ij}  F_{x(c),ij}
$$

$$
F_{x(c),ij} = \int e_{x(c)}^{\text{base}} F_{\text{corr}} f_i(2\beta_{\sigma}-1) f_j(u_{x(c),\sigma}) \, d\mathbf{r}.
$$

This allows us to rewrite the loss function as:

$$
L = \sum_{k=1}^{K} \dfrac{1}{2} w_k \left[ \sum_{l=1}^{L} c_l F_{k,l} - E_k^{\text{To-fit}} \right]^2,
$$

where:

- $c_l$ is the $l$-th linear coefficient, obtained by flattening the two-dimensional indices $c_{x(c),ij}$ of the three semi-local terms ($E_{x,sr}^{\text{SL}}$, $E_{c,ss}^{\text{SL}}$, $E_{c,os}^{\text{SL}}$) and $c_{x,\text{sr}}^{\text{HF}}$ into a single vector. The vector length is $3(M+1)(N+1)+1$.
- $F_{k,l}$ is the contribution of the $l$-th basis function to the energy for data point $k$, constructed from the expression for $F_{x(c),ij}$ above for system $k$.
- $E_k^{\text{To-fit}}$ is the energy term to fit. When using $\omega$B97X-V as the base functional, it is defined as:

$$
\begin{aligned}
E_k^{\text{To-fit}} = E_k^{\text{ref}} - E_k^{\omega\text{B97X-V}} + \left( E_{x,sr}^{\text{SL,}\omega\text{B97X-V}} + c_x^{\text{HF,}\omega\text{B97X-V}} E_{x,\text{sr}}^{\text{HF}} + E_{c,ss}^{\text{SL,}\omega\text{B97X-V}} + E_{c,os}^{\text{SL,}\omega\text{B97X-V}} \right),
\end{aligned}
$$

for system $k$.

After applying the weight, we reduce the optimization problem to a linear least-squares form:

$$
L = \dfrac{1}{2} \left\| \mathbf{A} \mathbf{c} - \mathbf{b} \right\|^2,
$$

where:

- $\mathbf{A}$ is a $K \times L$ matrix with elements $A_{k,l} = \sqrt{w_k} F_{k,l}$.
- $\mathbf{c}$ is a vector of the unknown coefficients $c_l$ to be determined.
- $\mathbf{b}$ is a vector with elements $b_k = \sqrt{w_k} E_k^{\text{To-fit}}$.

This setup allows efficient determination of the coefficients $c_l$ through standard linear least-squares techniques. However, the introduction of a base functional while necessary to create this linear least-square problem, makes our fitting problem not self-consistent and the fitting results may be sensitive to the choice of base functional. To address this issue, we decided to use an existing functional with good accuracy and transferability, specifically $\omega$B97X-V, as the base functional.

### Best Subset Selection

In our approach, we set $M = 11$ and $N = 7$ in the SL expansion above, resulting in $3(M+1)(N+1) + 1 = 289$ linear parameters in total. Including such a high number of parameters risks overfitting, reducing the functional's ability to generalize to systems outside the training set. Best subset selection (BSS) is a statistical method used to identify the most significant subset of variables (in this case, expansion coefficients or basis functions) that contribute meaningfully to the model's performance, which keeps the model as simple as possible. Given a subset size $s$, the BSS optimization problem is formulated as:

$$
\min_{\mathbf{c}} \left\| \mathbf{A} \mathbf{c} - \mathbf{b} \right\|^2 \quad \text{subject to} \quad \left\| \mathbf{c} \right\|_0 \leq s,
$$

where the $\ell_0$ (pseudo)norm of a vector $\mathbf{c}$ counts the number of non-zero entries in $\mathbf{c}$, given by $\| \mathbf{c} \|_0 = \sum_{l=1}^L 1(c_l \neq 0)$, with $1(\cdot)$ denoting the indicator function. Due to the cardinality constraint, this problem is NP-hard. A brute-force search through all possible subsets would require $2^{288} \approx 10^{86}$ linear fittings, an infeasible computational task.

To address this challenge, $\omega$B97M-V employed a forward stepwise selection strategy, a greedy algorithm that incrementally builds the model by adding variables one at a time according to predefined criteria. To improve robustness, over $10^{10}$ candidate subsets were explored. However, this strategy remains inefficient for our purposes. In particular, $\omega$B97M-V performed only a single BSS optimization overall, whereas in the present work we aim to perform one BSS optimization for each combination of choices of $F_{\mathrm{corr}}$ and expansion functions in order to systematically assess their effects. Therefore, we adopt the mixed-integer optimization (MIO) algorithm, a state-of-the-art approach introduced in 2016.

Mixed-integer optimization reformulates the subset selection problem as an optimization with both continuous and integer variables. The optimization problem can be expressed as:

$$
\begin{aligned}
Z_1 = \min_{\mathbf{c}, \mathbf{z}} &\quad \dfrac{1}{2} \left\| \mathbf{A} \mathbf{c} - \mathbf{b} \right\|^2 \\
\text{s.t.} &\quad \mathbf{Q} \mathbf{c} \leq \mathbf{t}, \\
&\quad -\mathcal{M}_U z_l \leq c_l \leq \mathcal{M}_U z_l, \quad l = 1, \dots, L, \\
&\quad z_l \in \{0, 1\}, \quad l = 1, \dots, L, \\
&\quad \sum_{l=1}^{L} z_l \leq s,
\end{aligned}
$$

where:

- $z_l \in \{0,1\}$ is a binary variable indicating whether coefficient $c_l$ is included ($z_l = 1$) or excluded ($z_l = 0$) from the model.
- $\sum_{l=1}^{L} z_l \leq s$ imposes a sparsity constraint, limiting the maximum number of non-zero coefficients.
- $\mathcal{M}_U$ is the upper bound for the coefficients (set to 25, as in $\omega$B97M-V).
- $\mathbf{Q} \mathbf{c} \leq \mathbf{t}$ represents numerical physical constraints.

We implement the MIO formulation using the Gurobi solver, with a wall-time limit of 1-2 hours per value of $s$, depending on the subset size, and one restart. All calculations are performed using 16 CPU cores. We scan subset sizes from $s = 24$ to 80, resulting in a total computational cost of approximately 3000 CPU hours for a single combination of $F_{\mathrm{corr}}$ and expansion-function choices.

By employing MIO, we aim to achieve a more efficient and closer-to-global solution of the subset selection problem, thereby improving both the performance and the generalizability of the resulting functional.


Beyond physical constraints, we also introduce grid-sensitivity constraints to improve numerical stability across different integral quadrature resolutions. The procedure is:

1. Construct the matrix $\mathbf{A}$ using the high-resolution grid `(250,974)/SG-1`.
2. Construct $\mathbf{A'}$ in parallel using the lower-resolution grid `(99,590)/SG-1`.
3. Optimize the functional first without grid-sensitivity constraints to identify promising candidates.
4. For each candidate, collect:
   - the 100 data points with the largest deviations $|(\mathbf{A}_{k,:}-\mathbf{A'}_{k,:})\mathbf{c}|$, and
   - an additional 200 data points with the largest $\ell_1$-norms of $\mathbf{A}_{k,:}-\mathbf{A'}_{k,:}$.
5. Take the union of these selected data points and enforce the grid-sensitivity constraints only on that subset.
6. Require the energy differences between the two grids to remain below a predefined threshold, set to `0.015 kcal/mol` in most cases, except in the exploration of Section `subsec:numerical_constraints_effect`.

This selective procedure keeps the functional stable across grid resolutions while avoiding the cost of enforcing the inequality constraints on every data point.
