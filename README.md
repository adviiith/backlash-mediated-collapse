# brand-collapse-model

Simulation code for the backlash-mediated brand collapse framework introduced in:

> Sharma, A. (2026). *Brand Fragility and Backlash-Mediated Collapse: A Controlled Nonlinear Framework for Identity Brand Risk.* AetherLabs Working Paper.

Built on the Cultural Debt Score (CDS) developed in:

> Sharma, A. (2026). *The Identity Brand's Trilemma: Scale, Authenticity, Survival (Pick Two).* AetherLabs Working Paper.

---

## What this is

The model tracks how a triggering event converts into meaning collapse (MC) and capital collapse (CC) over time, depending on a brand's pre-crisis structural fragility. Each time step is one week.

Three long-run regimes emerge from the same equation structure depending on parameter configuration:

- **Survival** — backlash absorbs, MC stabilises below the bifurcation point, brand recovers
- **Critical slow decay** — MC plateaus in a prolonged intermediate state, outcome uncertain
- **Collapse** — threshold crossed and held, recovery capacity depleted, MC → 1

The Yeezy (2022) collapse is the motivating empirical case. The default identity brand parameters are calibrated to CDS ≈ 0.685, matching the pre-crisis architecture documented in the paper.

---

## Requirements

```
pip install numpy matplotlib
```

Python 3.8+. No other dependencies.

---

## Usage

Run the full figure suite from the paper:

```bash
python simulation.py
```

Figures are saved to `outputs/`.

To run a custom scenario:

```python
from simulation import run

# Low-CDS brand, moderate event — should survive
model = run(
    brand_type='identity',
    T=200,
    overrides={'CDS': 0.25, 'L0': 0.50, 'D': 0.60}
)

h = model.history_arrays()
print(f"Final MC: {h['MC'][-1]:.3f}")
print(f"Collapsed: {bool(h['Xi'].max())}")
```

To model an institutional brand:

```python
model = run(brand_type='institutional', T=300)
```

To schedule multiple events (e.g. norm drift simulation):

```python
events = {
    10:  {'L0': 0.40, 'D': 0.50},
    120: {'L0': 0.40, 'D': 0.50},
    250: {'L0': 0.55, 'D': 0.65},  # same severity, lower threshold by now
}
model = run('identity', T=300, overrides={'CDS': 0.50}, events=events)
```

To attempt an identity reset after collapse:

```python
success = model.attempt_reset(delta_id=0.70)
print("Reset succeeded:", success)
```

---

## Parameters

Full parameter documentation is in Appendix B of the paper. The two presets — `identity_params()` and `institutional_params()` — are midpoints of the ranges in Table 7.

The key structural inputs are:

| Parameter | What it does |
|-----------|-------------|
| `CDS` | Cultural Debt Score — primary fragility input, normalised to [0, 1] |
| `PD` | Platform Dependence — sub-component of CDS used in economic injection |
| `L0` | Event severity at injection |
| `D` | Diffusion speed of the information environment |

CDS is the only parameter you need to set to model a specific brand. Everything else can stay at preset midpoints for a first-pass analysis.

---

## Repo structure

```
simulation.py     main model, parameter presets, figure functions
outputs/          generated figures (created on first run)
```

---

## Citation

If you use this code, please cite the working paper:

```bibtex
@techreport{sharma2026collapse,
  author      = {Sharma, Advith},
  title       = {Brand Fragility and Backlash-Mediated Collapse:
                 A Controlled Nonlinear Framework for Identity Brand Risk},
  institution = {AetherLabs},
  year        = {2026},
  type        = {Working Paper}
}
```

---

## Contact

Advith Sharma — [Linkedin](https://linkedin.com/in/adviith/)