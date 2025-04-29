
```markdown
# Metric Definitions

## RAM
\[
\mathrm{RAM} = \frac{\mathrm{AC}}{\mathrm{RT} + \epsilon},
\]
– AC = Var(regional TS)  
– RT = 1 / mean regional SD  
– ε = 10⁻⁶  
_Casali et al., Sci. Transl. Med. 5(198):198ra105 (2013). doi:10.1126/scitranslmed.3006294_

## PDI
\[
\mathrm{PDI} = \frac{\overline{H_{\mathrm{obs}}} - \overline{H_{\mathrm{base}}}}%
{H_{\mathrm{max}} - \overline{H_{\mathrm{base}}}},
\]
– \(\overline{H_{\mathrm{obs}}} = \frac1R\sum_{r} H(\mathrm{TS}_r)\)  
– \(\overline{H_{\mathrm{base}}} = \frac1R\sum_{r} H(\mathrm{shuffle}(\mathrm{TS}_r))\)  
– \(H_{\max} = \log_2(\#\text{bins})\)  
> **Baseline is per-region shuffle**—see compute_PDI implementation.  
_Mediano et al., Entropy 21(1):17 (2019). doi:10.3390/e21010017_

## NAS
\[
\mathrm{NAS} = \frac{1}{\binom{N}{2}}\sum_{i<j}|\rho_{ij}|.
\]
_Luppi et al., Neuron 102(4):712–727 (2024). doi:10.1016/j.neuron.2024.01.005_

## IIM
\[
\mathrm{IIM} = \min\bigl(\max\{I(X;X') - [I_A + I_B],0\},1\bigr).
\]
_Tononi, BMC Neuroscience 5(1):42 (2004). doi:10.1186/1471-2202-5-42_

## SRPI
\[
\mathrm{SRPI} = \frac{\mathrm{Var}(ts_{\mathrm{self}}) - \mathrm{Var}(ts_{\mathrm{nonself}})}%
{\mathrm{Var}(ts_{\mathrm{self}}) + \mathrm{Var}(ts_{\mathrm{nonself}})}.
\]
_Sui & Humphreys, J. Cogn. Neurosci. 27(7):1230–1241 (2015). doi:10.1162/jocn_a_00825_

## Hypergraph Synergy
\[
S = I \times G \times B,\quad
B = 1 - \frac{|W_{\mathrm{local}} - W_{\mathrm{global}}|}{W_{\mathrm{local}} + W_{\mathrm{global}}}.
\]
_Jang et al., PNAS 121(15):e240123011 (2024). doi:10.1073/pnas.240123011_
