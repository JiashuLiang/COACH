# COACH Final Functional Form and Parameters

This document summarizes the final COACH functional form and all linear and nonlinear parameters used in the final model. There are 73 independent linear parameters in total, along with 3 reoptimized nonlinear parameters and an optimized 3-body D4 correction. A total of 11 exact conditions are fully satisfied, and 2 more are partially satisfied. Adequate numerical stability was also enforced.

The total exchange-correlation energy is

$$
\begin{aligned}
E_{xc} &= E_x + E_c, \\
E_x &= E_{x,\mathrm{sr}}^{\mathrm{SL}} + c_{x,\mathrm{sr}}^{\mathrm{HF}} E_{x,\mathrm{sr}}^{\mathrm{HF}} + c_{x,\mathrm{lr}}^{\mathrm{HF}} E_{x,\mathrm{lr}}^{\mathrm{HF}}, \\
E_c &= E_{c,\mathrm{ss}}^{\mathrm{SL}} + E_{c,\mathrm{os}}^{\mathrm{SL}} + E_c^{\mathrm{VV10}} + E_c^{\mathrm{D4\text{-}ATM}}.
\end{aligned}
$$

The range-separated Hartree-Fock exchange is split into short-range and long-range contributions. We fix $c_{x,\mathrm{lr}}^{\mathrm{HF}} = 1$, while the trained value of $c_{x,\mathrm{sr}}^{\mathrm{HF}}$ is $0.22878980716640696$. The range-separation parameter is trained to $\omega = 0.27$.

The final dispersion treatment consists of a two-body nonlocal VV10 correlation term and a three-body D4-ATM correction. The VV10 parameters are

$$
b = 5.5, \qquad C = 0.01.
$$

For the D4-ATM three-body contribution, we use

$$
s_6 = 0, \quad s_8 = 0, \quad s_9 = 1.0, \quad a_1 = 0.215, \quad a_2 = 5.8,
$$

which was optimized on the L14 and vL11 benchmark sets without significantly degrading performance on GSCDB137.

The semi-local exchange and correlation components share the general B97-type form

$$
E_{x(c)}^{\mathrm{SL}} = \int e_{x(c)}^{\mathrm{base}}\, F_{\mathrm{corr}}\, g_{x(c)} \, d\mathbf{r}.
$$

Below, we summarize the base energy density $e_{x(c)}^{\mathrm{base}}$, the multiplicative correction factor $F_{\mathrm{corr}}$, and the fitting factor $g_{x(c)}$ for exchange, same-spin correlation, and opposite-spin correlation.

## Semi-local Short-Range Exchange

For exchange, we use the UEG exchange energy density as the base:

$$
e_{x,\sigma}^{\mathrm{UEG}} = -\frac{3}{2} \left( \frac{3}{4\pi} \right)^{1/3} \rho_\sigma^{4/3}.
$$

The short-range correction factor is

$$
F_{x,\sigma}^{\mathrm{sr}} =
1 - \frac{2}{3} a_\sigma \left(
2 \sqrt{\pi} \operatorname{erf} \left( \frac{1}{a_\sigma} \right)
- 3 a_\sigma + a_\sigma^3
+ (2 a_\sigma - a_\sigma^3) \exp \left( - \frac{1}{a_\sigma^2} \right)
\right),
$$

where

$$
a_\sigma = \omega/k_{F_\sigma}, \qquad
k_{F_\sigma} = (6\pi^2 \rho_\sigma)^{1/3}.
$$

We define the dimensionless variables $u_{x,\sigma}$ and $v_\sigma = 2\beta_\sigma - 1$ as

$$
\begin{aligned}
s_\sigma &= \frac{|\nabla \rho_\sigma|}{\rho_\sigma^{4/3}}, \\
u_{x,\sigma} &= \frac{\gamma_x s_\sigma^2}{1 + \gamma_x s_\sigma^2}, \\
\beta_\sigma &= \frac{\tau_\sigma - \tau_\sigma^W}{\tau_\sigma + \tau_\sigma^{\mathrm{UEG}}}, \\
\tau_\sigma^{\mathrm{UEG}} &= \frac{3}{5} \left(6\pi^2\right)^{2/3} \rho_\sigma^{5/3}, \\
\tau_\sigma^W &= \frac{1}{4} \frac{|\nabla \rho_\sigma|^2}{\rho_\sigma},
\end{aligned}
$$

with $\gamma_x = 0.004$.

The exchange fitting factor $g_{x,\sigma}$ is expanded using Legendre polynomials in $v_\sigma$ and monomials in $u_{x,\sigma}$:

$$
g_{x,\sigma} = g_{x,\sigma}(\nabla \rho_\sigma, \tau_\sigma)
= \sum_{i=0}^{12} \sum_{j=0}^{8} c_{x,ij} \, f_i(v_\sigma) \, f_j(u_{x,\sigma}),
$$

where $f_i(v) \equiv P_i(v)$ are the standard Legendre polynomials on $[-1,1]$ with $P_0(v)=1$ and $P_1(v)=v$, and $f_j(u) \equiv u^j$ are monomials on $[0,1]$.

### Nonzero Exchange Expansion Coefficients

| Coefficient | Coefficient |
| --- | --- |
| $c_{0,0} = 0.7080650005052257$ | $c_{5,2} = -3.463397249605392$ |
| $c_{0,2} = 5.973423554868972$ | $c_{5,3} = 3.75514691066743$ |
| $c_{0,3} = -17.33477136180401$ | $c_{6,0} = 0.2402820743898442$ |
| $c_{0,5} = 25.0$ | $c_{6,1} = -0.8660852096125707$ |
| $c_{0,7} = -13.303231846931668$ | $c_{7,0} = 0.10480172805257999$ |
| $c_{1,0} = -0.20577489944575586$ | $c_{7,1} = -0.5657722952687481$ |
| $c_{2,0} = -0.10779574940300828$ | $c_{8,1} = -0.43586821950068677$ |
| $c_{2,2} = 5.491317004106042$ | $c_{8,4} = 0.9891478454228174$ |
| $c_{2,3} = -6.658515516611359$ | $c_{9,0} = -0.09350557083826855$ |
| $c_{3,1} = 1.839797240566924$ | $c_{9,1} = 0.2557276229171211$ |
| $c_{3,2} = -7.509521330876057$ | $c_{9,2} = -0.7171538898115716$ |
| $c_{3,3} = 18.654513589197446$ | $c_{10,0} = -0.034948386270515386$ |
| $c_{3,4} = -12.139029019506687$ | $c_{11,0} = -0.03464312342837082$ |
| $c_{4,0} = 0.20195969717314624$ | $c_{11,1} = 0.20050993281346377$ |
| $c_{5,0} = 0.2600717991008879$ | $c_{11,2} = -0.3773473250434576$ |

## Same-Spin Correlation

For correlation, we adopt SCAN correlation ($\alpha = 1$) as the base functional for both same-spin and opposite-spin components. In the $\alpha = 1$ limit, SCAN correlation reduces to a revised PBE-like form:

$$
\begin{aligned}
\varepsilon_c^{1} &= \varepsilon_c^{\mathrm{LSDA1}} + H_1, \\
H_1 &= \gamma\,\phi^{3}\,\ln\!\left[1+w_1\left(1-g\!\left(A t^2\right)\right)\right], \\
w_1 &= \exp\!\left[-\frac{\varepsilon_c^{\mathrm{LSDA1}}}{\gamma\,\phi^{3}}\right]-1, \\
g\!\left(A t^2\right) &= \frac{1}{\left(1+4 A t^2\right)^{1/4}}, \\
A &= \frac{\beta(r_s)}{\gamma\, w_1}, \\
t &= \left(\frac{3\pi^2}{16}\right)^{1/3}\frac{s}{\phi\, r_s^{1/2}}, \\
s &= \frac{|\nabla\rho|}{2(3\pi^2)^{1/3}\rho^{4/3}}.
\end{aligned}
$$

Here $\varepsilon_c^{\mathrm{LSDA1}}$ denotes the LSDA correlation energy density (PW92), with $\gamma = 0.031091$ and

$$
\beta(r_s)=0.066725\,\frac{1+0.1 r_s}{1+0.1778 r_s}, \qquad
\phi=\frac{(1+\zeta)^{2/3}+(1-\zeta)^{2/3}}{2},
$$

where

$$
r_s=\left(\frac{3}{4\pi\rho}\right)^{1/3}, \qquad
\zeta=(\rho_\alpha-\rho_\beta)/\rho.
$$

The same-spin and opposite-spin base correlation energy densities are

$$
\begin{aligned}
e_{c,\sigma\sigma}^{\mathrm{base}} &= e_c^{\mathrm{base}}(\rho_\sigma,0), \\
e_{c,\mathrm{os}}^{\mathrm{base}} &= e_c^{\mathrm{base}}(\rho_\alpha,\rho_\beta) - e_c^{\mathrm{base}}(\rho_\alpha,0) - e_c^{\mathrm{base}}(0,\rho_\beta).
\end{aligned}
$$

To enforce the self-correlation correction (SCC), we use the multiplicative correction factor

$$
F_{c,\mathrm{ss}} = 2\beta_\sigma.
$$

For same-spin correlation, we use the dimensionless variables $v_\sigma = 2\beta_\sigma - 1$  (the same variable as the exchange)  and $u_{c,\sigma\sigma}$:

$$
u_{c,\sigma\sigma} = \frac{\gamma_{c,\mathrm{ss}} s_{\sigma}^2}{1 + \gamma_{c,\mathrm{ss}} s_{\sigma}^2},
$$

with $\gamma_{c,\mathrm{ss}} = 0.01$.

The fitting factor $g_{c,\sigma\sigma}$ is expanded using Legendre polynomials in $v_\sigma$ and monomials in $u_{c,\sigma\sigma}$.

### Nonzero Same-Spin Expansion Coefficients

| Coefficient | Coefficient |
| --- | --- |
| $c_{0,0} = -0.3039405655250467$ | $c_{3,1} = -2.226676731002602$ |
| $c_{0,1} = -1.2900096158738523$ | $c_{3,2} = -2.7311170048590196$ |
| $c_{0,4} = 17.715459474491766$ | $c_{5,0} = -1.4633617968363573$ |
| $c_{0,7} = -13.929080028872658$ | $c_{7,0} = -0.8348281349925031$ |
| $c_{1,0} = -1.8711635396581212$ | $c_{8,0} = 0.4818154118907339$ |
| $c_{1,1} = -10.122833804966692$ | $c_{8,1} = -3.3444231176840256$ |
| $c_{1,2} = 22.05988705553151$ | $c_{8,2} = 6.986341349369924$ |
| $c_{2,0} = -1.8278556540465263$ | $c_{8,3} = 14.980984577715752$ |
| $c_{2,1} = 6.994998018578696$ | $c_{8,4} = -25.0$ |
| $c_{2,4} = -8.349141007699943$ | $c_{11,3} = 1.5676639733349753$ |

## Opposite-Spin Correlation

For opposite-spin correlation, we use the same base correlation functional defined above. The opposite-spin base contribution is

$$
e_{c,\mathrm{os}}^{\mathrm{base}} = e_c^{\mathrm{base}}(\rho_\alpha,\rho_\beta) - e_c^{\mathrm{base}}(\rho_\alpha,0) - e_c^{\mathrm{base}}(0,\rho_\beta).
$$

There is no multiplicative correction factor for the opposite-spin correlation energy.

We use the dimensionless variables $u_{c,\alpha\beta}$ and $w_{c,\alpha\beta}$ defined by

$$
\begin{aligned}
s_{\alpha\beta}^2 &= \frac{1}{2}(s_\alpha^2 + s_\beta^2), \\
u_{c,\alpha\beta} &= \frac{\gamma_{c,\mathrm{os}} s_{\alpha\beta}^2}{1 + \gamma_{c,\mathrm{os}} s_{\alpha\beta}^2}, \\
w_{c,\alpha\beta} &= \frac{t_{\alpha\beta} - 1}{t_{\alpha\beta} + 1}, \\
t_{\alpha\beta} &= \frac{1}{2}(t_\alpha + t_\beta), \\
t_{\sigma} &= \frac{\tau_\sigma^{\mathrm{UEG}}}{\tau_\sigma}.
\end{aligned}
$$

with $\gamma_{c,\mathrm{os}} = 0.006$, identical to the value used in $\omega$B97M-V.

The opposite-spin fitting factor is expanded using Legendre polynomials in both $w_{c,\alpha\beta}$ and $u_{c,\alpha\beta}$.

### Nonzero Opposite-Spin Expansion Coefficients

| Coefficient | Coefficient |
| --- | --- |
| $c_{0,0} = 2.0330833570991405$ | $c_{5,6} = -22.533887529213906$ |
| $c_{0,5} = -3.071472897617402$ | $c_{5,7} = 9.590467389102345$ |
| $c_{0,7} = -1.9671740896453935$ | $c_{6,4} = 4.4188486804036184$ |
| $c_{1,0} = -3.124170248419258$ | $c_{8,4} = 4.293474773028771$ |
| $c_{1,1} = 7.594716450411337$ | $c_{9,2} = -1.5286479621022213$ |
| $c_{2,4} = 4.618674090063557$ | $c_{9,7} = 2.3555422987546892$ |
| $c_{3,7} = 2.572431599509026$ | $c_{10,0} = 0.7896380110221484$ |
| $c_{4,4} = -1.120055362237976$ | $c_{10,3} = -1.9225815901087469$ |
| $c_{4,7} = -2.5762282779751264$ | $c_{11,1} = 2.229310613916473$ |
| $c_{5,1} = -1.4299548850828665$ | $c_{11,6} = -1.5811003317333285$ |
| $c_{5,4} = -20.76844377770899$ | $c_{11,7} = 1.1456216634190832$ |
| $c_{5,5} = 25.0$ |  |
