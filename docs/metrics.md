# Metric Definitions

This document describes the operational formulas used in code for the MPC
metrics and CI.

## RAM
\[
\mathrm{RAM}
=
\frac{M}{T + \epsilon}\cdot Q,
\qquad
\epsilon = \Delta t_{\mathrm{res}},
\]
\[
Q
=
\left(
G^{w_G}F^{w_F}U^{w_U}
\right)^{\frac{1}{w_G+w_F+w_U}},
\qquad
G,F,U\in[0,1].
\]

- \(M\): response magnitude (GLM stimulus beta magnitude, scaled by default factor 0.5)
- \(T\): response latency (seconds)
- \(\epsilon\): stabilizer tied to temporal resolution (defaults to TR)
- \(G\): goal-alignment quality from event-locked neural objective/response coupling
- \(F\): feedback-integration quality from feedback-locked neural coupling to feedback signal
- \(U\): adaptive-update quality from feedback-driven trial-to-trial policy updates
- \(Q\): weighted geometric quality factor (dimensionless)

Implementation notes:
- `compute_RAM(..., epsilon=None)` sets `epsilon = tr`.
- Latency is constrained to be non-negative.
- `compute_RAM` accepts either a legacy onset list or a structured event bundle:
  `{"onsets","goal_onsets","feedback_onsets","feedback_values"}`.
- RAM is treated as undefined (`NaN`) when required event annotations are
  missing; no synthetic onset at `t=0` is injected.
- By default, RAM requires explicit feedback events/values
  (`require_explicit_feedback=True`), so feedback terms are not inferred from
  placeholder assumptions.
- The speed term \(M/(T+\epsilon)\) remains in \(\mathrm{s}^{-1}\); \(Q\) is unitless,
  so RAM remains in \(\mathrm{s}^{-1}\).

## PDI
\[
\mathrm{PDI}_{\mathrm{raw}}
=
D_{\mathrm{core}}\,
C_{\mathrm{stab}}\,
C_{\mathrm{noise}}
\]
\[
\mathrm{PDI}^{+} = \max\!\left(\mathrm{PDI}_{\mathrm{raw}},0\right).
\]
\[
\mathrm{PDI}^{\mathrm{raw}}_{\mathrm{anchor}}(r)
=
\mathrm{PDI}_{\mathrm{raw}}\!\left(r;\,B=B_{\mathrm{deep\_rest,subject}}\right),
\quad
\mathrm{PDI}^{\mathrm{raw}}_{\mathrm{task}}(r)
=
\mathrm{PDI}_{\mathrm{raw}}\!\left(r;\,B=B_{\mathrm{state\_rest,subject}}\right).
\]

Core components (all measured from node\(\times\)time data):
\[
X_{\Xi}
=
\mathrm{clip}\!\left(
\frac{H_R-h_R}{\log_2 K},0,1
\right),\quad
X_{\Delta}
=
\frac{2}{R(R-1)}\sum_{n<m}\mathrm{JSD}(p_n,p_m),
\]
\[
X_D
=
\frac{d_{\mathrm{eff}}-1}{Q-1},\quad
d_{\mathrm{eff}}=\frac{(\sum_q s_q^2)^2}{\sum_q s_q^4},
\]
\[
X_C
=
\frac{1}{|\mathcal A|R}\sum_{a\in\mathcal A}\sum_{n=1}^{R}
\left(
\frac{H(\pi_{n,a})}{\log_2(m!)}
\cdot
\frac{\mathrm{JSD}(\pi_{n,a},u_m)}{\mathrm{JSD}(\delta_m,u_m)}
\right).
\]

For \(j\in\{\Xi,\Delta,D,C\}\), baseline references are weighted by run size:
\[
\bar X_{j,\mathrm{base}}
=
\sum_{b\in B}w_bX_j^{(b)},
\quad
\sigma_{j,\mathrm{base}}
=
\sqrt{\sum_{b\in B}w_b(X_j^{(b)}-\bar X_{j,\mathrm{base}})^2},
\quad
w_b=\frac{N_b}{\sum_{\ell\in B}N_\ell}.
\]
\[
g_j
=
\mathrm{clip}\!\left(
\frac{X_{j,\mathrm{obs}}-\bar X_{j,\mathrm{base}}}
{1+\sigma_{j,\mathrm{base}}},
0,1
\right),
\quad
D_{\mathrm{core}}=\sum_j\alpha_j g_j,\ \sum_j\alpha_j=1.
\]
\[
C_{\mathrm{stab}}
=
\mathrm{clip}\!\left(
\frac{\Upsilon_{\mathrm{obs}}}{\bar\Upsilon_{\mathrm{base}}+\epsilon_s},0,1
\right),
\quad
C_{\mathrm{noise}}
=
\exp\!\left[-\kappa\max(0,\nu_{\mathrm{obs}}-\bar\nu_{\mathrm{base}})\right].
\]

Implementation notes:
- `compute_PDI` uses four measurable dimensions:
  normalized excess differentiation, spatial repertoire divergence, effective
  dimensionality, and multiscale ordinal complexity.
- Default component weights are
  \((\alpha_{\Xi},\alpha_{\Delta},\alpha_D,\alpha_C)=(0.35,0.25,0.20,0.20)\).
- Strict validation mode uses measured baselines:
  deep-rest anchor and state-matched rest control endpoint.
- Strict validation mode does not use shuffled/surrogate fallback; missing required
  baseline keeps PDI endpoints undefined (`NaN`).
- CI consumes the anchored non-negative component
  \(\max(\mathrm{PDI}^{\mathrm{raw}}_{\mathrm{anchor}},0)\).

## NAS
\[
\mathrm{NAS}_f
=
(\bar L_f)^{\alpha}
(\bar B_f)^{\beta}
(\bar H_f)^{\gamma}
(\bar D_f)^{\delta}
(\bar W_f)^{\eta}
(\bar E_f)^{\zeta}
(\bar R_f)^{\rho},
\qquad
\alpha+\beta+\gamma+\delta+\eta+\zeta+\rho=1,
\]
\[
\mathrm{NAS}
=
\sum_{f\in\mathcal F}\omega_f\,\mathrm{NAS}_f,
\qquad
\sum_f \omega_f=1.
\]
\[
G=\mathrm{TopK}_q(s^0),\quad
s_i^0=\frac{1}{R-1}\sum_{j\neq i}A^0_{ij},\quad
A^0_{ij}=|\mathrm{corr}(x_i,x_j)|.
\]
\[
V_{f,w}=\{i:\ s_i^{f,w}\ge Q_{1-\tau}(s^{f,w})\},\quad
s_i^{f,w}=\frac{1}{R-1}\sum_{j\neq i}A_{ij}^{f,w}.
\]
\[
W_{f,w}=r_{f,w}\,\xi_{f,w},\quad
r_{f,w}=\frac{|V_{f,w}\cap G|}{|G|},\quad
\xi_{f,w}=\mathrm{clip}\!\left(\frac{L_{f,w}-\bar A_{\bar V\bar V}^{f,w}}{1-\bar A_{\bar V\bar V}^{f,w}+\epsilon},0,1\right).
\]
\[
\Gamma_{ij}^{f,w}=\max\!\left(C_{ij}^{f,w}-C_{ji}^{f,w},0\right),\quad
C_{ij}^{f,w}=\frac{1}{T'}\sum_t\tilde x_i^{f,w}(t)\tilde x_j^{f,w}(t+\ell),
\]
\[
E_{f,w}=\mathrm{clip}\!\left(
\frac{\langle\Gamma_{ij}^{f,w}\rangle_{i\in V,j\notin V}
-\langle\Gamma_{ij}^{f,w}\rangle_{i\notin V,j\in V}}
{\langle\Gamma_{ij}^{f,w}\rangle_{i\in V,j\notin V}
+\langle\Gamma_{ij}^{f,w}\rangle_{i\notin V,j\in V}+\epsilon},0,1\right).
\]
\[
R_{f,w}=r_{f,w}\cdot
\frac{1}{|\mathcal L|}\sum_{\ell\in\mathcal L}\max\!\left(\mathrm{corr}(y_{f,w}(t),y_{f,w}(t+\ell)),0\right),
\quad
y_{f,w}(t)=\frac{1}{|G|}\sum_{i\in G}x_i^{f,w}(t).
\]

Implementation notes:
- Code uses the richer synchrony construction (`compute_NAS`) with:
  intra-broadcast synchrony \(L\), broadcast reach \(B\), triadic closure \(H\),
  dynamic stability \(D\), dedicated-workspace recruitment/ignition \(W\),
  directed broadcast efficacy \(E\), and reverberatory persistence \(R\).
- Default NAS exponents are
  \((\alpha,\beta,\gamma,\delta,\eta,\zeta,\rho)=
  (0.20,0.16,0.14,0.12,0.16,0.12,0.10)\) (renormalized internally).
- NAS computation requires explicit measured timing and analysis settings:
  positive `tr`, explicit `tau`, explicit `bands`, explicit `band_weights`,
  explicit `window_len`, and explicit `step_len`.
- Pairwise synchrony is phase/envelope mixed from explicit bandpass-filtered
  data; non-bandpassed correlation fallback is not used.
- Dedicated workspace \(G\) can be provided explicitly (`workspace_nodes`) or
  inferred from global synchrony-centrality quantiles.
- Directed dissemination is measured from lagged asymmetric coupling
  (`directed_lag`) and reverberation from positive lagged autocorrelation of
  workspace activity (`reverberation_lags`).
- In validation mode, the reported value is normalized rich NAS
  (`normalize=True`) without baseline subtraction.

## IIM
Code uses the finite-order IIT-leaning surrogate based on mechanism/purview
irreducibility mass \(\Psi\), not the simple lagged-MI decomposition.

For full TPM mass \(\Psi\) and cut-specific mass \(\Psi^\kappa\):
\[
\mathrm{IIM}_{\mathrm{raw}}
=
\frac{\Psi-\max_{\kappa}\Psi^\kappa}{\Psi+\epsilon},
\qquad
\epsilon>0,
\]
\[
\mathrm{IIM}_{\mathrm{can}}
=
\mathrm{clip}\!\left(\mathrm{IIM}_{\mathrm{raw}},0,1\right)\in[0,1].
\]

Interpretation:
- \(\mathrm{IIM}_{\mathrm{raw}}\): signed contrast relative to the MIP.
- \(\mathrm{IIM}_{\mathrm{can}}\): canonical CI component (\(\mathrm{IIM}_{\mathrm{raw}}=0 \Rightarrow \mathrm{IIM}_{\mathrm{can}}=0\)).

Implementation notes:
- If `n_parts=None`, cuts are exhaustive over unique bipartitions.
  If `n_parts` is set, cut candidates are sampled without replacement.
- Execution is staged:
  1) \(\Psi_{\mathrm{full}}\) for intact TPM,
  2) induced-partition kernel materialization into disk-backed cache,
  3) cut search using lookup-only aggregation (automatic fallback to per-cut recompute on cache-miss).
- Code returns `IIM` (canonical), `IIM_raw`, and `IIM_raw_scaled`
  (`IIM_raw_scaled` is compatibility-preserving; default scale is 1.0).
- Undefined IIM is explicit (`IIM_defined=False`, `IIM_undefined_reason`) and
  must be handled as missingness, not as numeric zero.

## SRPI
For each self/non-self event \(e\), with pre-window \(W_{\mathrm{pre}}\),
response lag \(L\), and response window \(W_{\mathrm{resp}}\):
\[
p_e=\frac{1}{W_{\mathrm{pre}}}\sum_{\tau=e-W_{\mathrm{pre}}}^{e-1}x(\tau),\quad
r_e=\frac{1}{W_{\mathrm{resp}}}\sum_{\tau=e+L}^{e+L+W_{\mathrm{resp}}-1}x(\tau),\quad
d_e=r_e-p_e.
\]
\[
R=\left[\frac{\Gamma_s-\Gamma_n}{\Gamma_s+\Gamma_n+\epsilon}\right]_+,\quad
\Gamma_c=\left\langle\|d_e\|_2\right\rangle_{e\in\mathcal E_c},
\]
\[
S=1-\exp\!\left(-\tfrac{1}{2}\Delta\mu^\top(\Sigma_{\mathrm{pool}}+\lambda I)^{-1}\Delta\mu\right),
\]
\[
K=\left[\frac{\bar\rho_s-\bar\rho_n}{2}\right]_+,\quad
\bar\rho_c=\Big\langle\mathrm{corr}(d_{c,i},d_{c,j})\Big\rangle_{i<j},
\]
\[
I=\left[\left|\mathrm{corr}(u_s,m_s)\right|-\left|\mathrm{corr}(u_n,m_n)\right|\right]_+.
\]
Here \(u_c=P_cv_1\), \(P_c=\{p_e\}_{e\in\mathcal E_c}\), \(v_1\) is the first
right-singular vector of pooled pre-event states, and
\(m_c=(\|d_e\|_2)_{e\in\mathcal E_c}\).

Event-count reliability attenuation:
\[
r_N=1-\exp\!\left(-\frac{N_{\mathrm{eff}}}{\tau_N}\right),\quad
N_{\mathrm{eff}}=\min(|\mathcal E_s|,|\mathcal E_n|),
\]
\[
\widetilde R=r_NR,\ \widetilde S=r_NS,\ \widetilde K=r_NK,\ \widetilde I=r_NI.
\]
\[
\mathrm{SRPI}
=
\left(
\widetilde R^{w_R}
\widetilde S^{w_S}
\widetilde K^{w_K}
\widetilde I^{w_I}
\right)^{\frac{1}{w_R+w_S+w_K+w_I}}
\in[0,1].
\]

Implementation notes:
- Code uses four jointly required components:
  reactivity bias, separability, self-pattern stability, and internal-state
  coupling (`compute_SRPI`).
- SRPI is undefined (`NaN`) when self/non-self events are missing or
  insufficient after event-windowing; no neutral placeholder value is used.
- Pipeline event parsing now carries explicit `self_onsets` and
  `nonself_onsets` from BIDS `events.tsv`.
- SRPI hyperparameters are explicit in pipeline execution
  (`srpi_require_explicit_params=True`).

## CI
\[
\mathrm{CI}
= (\mathrm{RAM}^*)^{\alpha}
  (\mathrm{PDI}^{+*})^{\beta}
  (\widetilde{\mathrm{NAS}}^*)^{\gamma}
  (\mathrm{IIM}_{\mathrm{can}}^{*})^{\delta}
  (\mathrm{SRPI}^*)^{\rho},
\quad
\alpha+\beta+\gamma+\delta+\rho=1.
\]
\[
M^* = \frac{M}{\langle M\rangle_{\mathrm{hum}}}.
\]
\[
\widetilde{\mathrm{NAS}}
=
\mathrm{NAS}\cdot C_{\mathrm{CB}}(\theta),
\qquad
C_{\mathrm{CB}}(\theta)=\max(\text{HypergraphSynergy}(\theta),0).
\]

Implementation notes:
- CI uses a weighted geometric mean (`compute_CI`).
- CI uses hard criteria: if any weighted metric is undefined, CI is set to 0.
- The NAS term used inside CI is internally coherence-modulated
  (\(\widetilde{\mathrm{NAS}}\), as above).
- If no human references are supplied, `compute_synergy_ci` estimates references
  from awake-session cohort means.
- Any zero component with non-zero weight makes CI exactly zero.
