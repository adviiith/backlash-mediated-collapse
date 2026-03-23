#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Brand Fragility & Backlash-Mediated Collapse
Simulation engine — Version 3

Sharma, A. (2026). Brand Fragility and Backlash-Mediated Collapse:
A Controlled Nonlinear Framework for Identity Brand Risk.
AetherLabs Working Paper.

The model tracks how backlash from a triggering event converts into
meaning collapse (MC) and eventually capital collapse (CC), depending
on a brand's pre-crisis structural fragility (CDS). Each time step
is one week. Full parameter documentation in Appendix B of the paper.

Run this file directly to reproduce all figures from Appendix C.
"""

import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


OUTPUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'outputs')
os.makedirs(OUTPUT_DIR, exist_ok=True)


# ------------------------------------------------------------------
# Core model
# ------------------------------------------------------------------

class BrandCollapseModel:
    """
    Discrete-time simulation of the backlash-mediated collapse system.

    The update order each period matters — E(t) has to be computed
    before MC(t) so that recovery uses the current capacity, not last
    period's. Everything else follows the pseudocode in Appendix A.
    """

    def __init__(self, params=None, brand_type='identity',
                 use_noise=False, noise_sigma=0.02):
        if params is None:
            params = identity_params() if brand_type == 'identity' else institutional_params()
        self.p = dict(params)
        self.use_noise = use_noise
        self.noise_sigma = noise_sigma
        self._check_params()
        self._init_state()

    def _check_params(self):
        # Just making sure nothing got dropped when overriding presets
        required = [
            'B0', 'B_floor', 'beta', 'delta', 'kappa', 'gamma',
            'tau_B', 'tau_MC', 'gamma_c', 'gamma_e',
            'g_c', 'g_e', 'g_o', 'd_c', 'd_e', 'd_o', 'r',
            'phi_c', 'phi_e', 'phi_o', 'MC_gate', 'psi', 'xi',
            'R_total', 'rho', 'eta', 'w_c', 'w_e', 'w_o',
            'a', 'kappa_s', 'lam', 'mu', 'theta', 'omega', 'b',
            'eta_E', 'zeta', 'alpha_coll', 'kappa_b', 'theta_L',
            'c', 'v', 'e', 'k', 'tau_min', 'CDS', 'PD',
            'phi_s', 'Delta_min', 'A_inf', 'A_min', 'alpha_A', 'sigma_A',
            'E_min', 'gamma_r', 'lam_CC', 'phi_R'
        ]
        missing = [k for k in required if k not in self.p]
        if missing:
            raise ValueError(f"Missing parameters: {missing}")

    def _init_state(self):
        p = self.p

        # Backlash components and aggregates
        self.B_core  = 0.0
        self.B_econ  = 0.0
        self.B_outer = 0.0
        self.B       = 0.0

        # Propagation chain: raw -> smoothed -> saturated
        self.B_hat      = 0.0
        self.B_hat_prev = 0.0
        self.B_tilde    = 0.0

        # Collapse variables
        self.MC     = 0.0
        self.MC_hat = 0.0
        self.MC_hat_prev = 0.0
        self.CC     = 0.0

        # Capacity and environment
        self.E       = 1.0          # recovery capacity starts full
        self.L       = 0.0          # legitimacy / event severity
        self.N       = p.get('N0', 0.0)  # norm context

        # Regime tracking
        self.Xi      = 0            # 0 = elastic, 1 = collapsed
        self.counter = 0            # consecutive periods above threshold
        self.MC_lag  = 0.0          # MC from previous period (for feedback)
        self.R_avail = p['R_total']
        self.t_elapsed = 0          # time since collapse, used for A(t)

        # Initial threshold (no backlash yet, so just the CDS-compressed baseline)
        self.B_star = max(
            p['B0'] * np.exp(-p['beta'] * p['CDS']) * (1.0 + p['delta'] * self.N),
            p['B_floor']
        )

        self.t       = 0
        self.events  = {}
        self.history = []

    def reset(self):
        self.events = {}
        self._init_state()

    def schedule_event(self, t_event, L0=None, D=None):
        """Register a crisis event at period t_event with severity L0 and diffusion speed D."""
        self.events[t_event] = {
            'L0': L0 if L0 is not None else self.p.get('L0', 0.85),
            'D':  D  if D  is not None else self.p.get('D',  0.90),
        }

    # ------------------------------------------------------------------
    # Per-period update methods (called in order inside step())
    # ------------------------------------------------------------------

    def _apply_event(self):
        # Inject backlash pulses if an event fires this period.
        # The CDS amplifies how hard each audience segment reacts —
        # that's the whole point of measuring structural fragility upfront.
        if self.t not in self.events:
            return
        p  = self.p
        ev = self.events[self.t]
        L0, D = ev['L0'], ev['D']

        self.B_core  = np.clip(self.B_core  + L0 * D * (1.0 + p['gamma_c'] * p['CDS']), 0, 1)
        self.B_econ  = np.clip(self.B_econ  + L0 * D * (1.0 + p['gamma_e'] * p['CDS']) * p['PD'], 0, 1)
        self.B_outer = np.clip(self.B_outer + L0 * D, 0, 1)
        self.L = L0

    def _apply_response(self):
        # Response is proportional to current backlash, capped by what's
        # left in the budget. Once the budget runs dry the brand is
        # suppressing nothing regardless of intent.
        p = self.p
        R_desired = p['r'] * self.B
        R_t = min(R_desired, self.R_avail)

        # Budget depletes through use and through a passive exhaustion term
        # (fragile brands burn credibility just by being in the news)
        drain = p['rho'] * R_t + p['phi_R'] * self.B
        self.R_avail = np.clip(self.R_avail - drain + p['eta'], 0, p['R_total'])
        return R_t

    def _update_backlash(self, R_t):
        # Each audience segment follows logistic pile-on with decay and
        # response suppression. Once MC crosses MCgate, the cancel-cycle
        # feedback kicks in and starts feeding back into backlash.
        p  = self.p
        fb = max(self.MC_lag - p['MC_gate'], 0.0)

        for attr, g, d, phi in [
            ('B_core',  p['g_c'], p['d_c'], p['phi_c']),
            ('B_econ',  p['g_e'], p['d_e'], p['phi_e']),
            ('B_outer', p['g_o'], p['d_o'], p['phi_o']),
        ]:
            b_i = getattr(self, attr)
            b_i += g * b_i * (1 - b_i) - d * b_i - p['r'] * R_t * b_i + phi * fb
            if self.use_noise:
                b_i += np.random.normal(0, self.noise_sigma)
            setattr(self, attr, np.clip(b_i, 0, 1))

        # Media outrage bleeds into the core tribe and institutional partners
        # even when neither directly engaged — the slow siege mechanism
        self.B_core = np.clip(self.B_core + p['psi'] * self.B_outer, 0, 1)
        self.B_econ = np.clip(self.B_econ + p['xi']  * self.B_outer, 0, 1)

        self.B = np.clip(
            p['w_c'] * self.B_core + p['w_e'] * self.B_econ + p['w_o'] * self.B_outer,
            0, 1
        )

    def _propagate_backlash(self):
        # Audiences don't process backlash the instant it exists.
        # B_hat is the smoothed signal that actually reaches belief systems.
        # B_tilde applies Michaelis-Menten saturation — at high backlash
        # levels each additional unit does less damage to meaning.
        p = self.p
        self.B_hat_prev = self.B_hat
        self.B_hat   = np.clip(self.B_hat + (1 / p['tau_B']) * (self.B - self.B_hat), 0, 1)
        self.B_tilde = self.B_hat / (1 + p['kappa_s'] * self.B_hat)

    def _update_norms_and_threshold(self):
        # Norms erode slowly under sustained backlash. This is what causes
        # the "suddenly unacceptable" pattern — nothing changed except
        # the threshold kept quietly dropping for years.
        p = self.p
        self.N = np.clip(self.N - p['kappa'] * self.B, -1, 1)
        self.B_star = max(
            p['B0'] * np.exp(-p['beta'] * p['CDS']) * (1 + p['delta'] * self.N)
            + p['gamma'] * self.MC_lag,
            p['B_floor']
        )

    def _decay_legitimacy(self):
        # Event severity fades over time unless new revelations arrive
        self.L = np.clip(self.L - self.p['theta_L'] * self.L, 0, 1)

    def _update_regime(self):
        # Count consecutive periods above the collapse threshold.
        # Collapse requires sustained crossing — a single bad week isn't enough.
        p = self.p
        if self.B >= self.B_star:
            self.counter += 1
        else:
            self.counter = 0

        if self.counter >= p['tau_min'] and self.Xi == 0:
            self.Xi = 1
            self.t_elapsed = 0

        if self.Xi == 0:
            return 1.0, p['b']                      # normal regime
        else:
            self.t_elapsed += 1
            return p['alpha_coll'], p['b'] / p['kappa_b']  # collapsed regime

    def _update_recovery_capacity(self):
        # Organisational capacity to recover regenerates when MC is low,
        # depletes quadratically when MC is high. The bifurcation at MC†
        # is where regeneration can no longer outpace depletion.
        p = self.p
        regen_factor = self.R_avail / max(p['R_total'], 1e-9)
        new_E = (self.E
                 + p['eta_E'] * regen_factor * (1 - self.MC) * (1 - self.E)
                 - p['zeta'] * self.MC ** 2)
        self.E = np.clip(new_E, p['E_min'], 1.0)

    def _update_mc(self, alpha_xi, b_xi):
        # Meaning collapse has three inputs:
        #   level damage  — backlash magnitude, saturated and amplified
        #   velocity shock — how fast backlash is accelerating
        #   recovery      — capacity-weighted healing
        # The ceiling factor (1 - MC^1.2) prevents runaway near unity
        # while remaining roughly linear at low MC values.
        p   = self.p
        A_t = 1 + p['lam'] * self.L + p['mu'] * p['CDS']

        level_dmg = alpha_xi * p['a'] * self.B_tilde * A_t * (1 - self.MC ** 1.2)

        dB_hat = max(self.B_hat - self.B_hat_prev, 0.0)
        velocity = min(p['theta'] * dB_hat, p['omega'] * (1 - self.MC))

        recovery   = b_xi * self.E * self.MC
        resistance = p['gamma_r'] * self.MC ** 2   # quadratic floor prevents full absorption

        self.MC = np.clip(self.MC + level_dmg + velocity - recovery - resistance, 0, 1)

    def _update_mc_hat(self):
        # Institutional partners act on processed MC signals, not raw ones.
        # Board decisions, legal review, and partner deliberations introduce
        # a real-world lag between belief erosion and contractual action.
        p = self.p
        self.MC_hat_prev = self.MC_hat
        self.MC_hat = np.clip(
            self.MC_hat + (1 / p['tau_MC']) * (self.MC - self.MC_hat),
            0, 1
        )

    def _update_cc(self):
        # Capital collapse comes through three pathways that arrive in sequence:
        #   1. Economic partners react directly to Becon (no lag)
        #   2. Velocity shock fires as processed MC propagates through networks
        #   3. Chronic meaning damage accumulates while identity stays destroyed
        # This temporal ordering matches what actually happened with Adidas/Yeezy.
        p = self.p
        dMC_hat = max(self.MC_hat - self.MC_hat_prev, 0.0)
        self.CC = np.clip(
            self.CC
            + p['c'] * self.MC_hat          # chronic meaning pathway
            + p['v'] * dMC_hat              # velocity shock
            + p['e'] * self.B_econ          # direct economic pathway
            - p['k'] * self.CC              # stabilisation
            - p['lam_CC'] * self.CC ** 2,   # quadratic damping at high levels
            0, 1
        )

    # ------------------------------------------------------------------
    # Main loop
    # ------------------------------------------------------------------

    def step(self):
        """Advance the simulation by one weekly period."""
        self.t += 1

        self._apply_event()
        R_t = self._apply_response()

        self._update_backlash(R_t)
        self._propagate_backlash()
        self._update_norms_and_threshold()
        self._decay_legitimacy()
        alpha_xi, b_xi = self._update_regime()

        # E(t) must come before MC(t) — recovery uses current capacity
        self._update_recovery_capacity()
        self._update_mc(alpha_xi, b_xi)
        self._update_mc_hat()
        self._update_cc()

        self.MC_lag = self.MC
        self._record()

    def _record(self):
        self.history.append({
            'B_core': self.B_core, 'B_econ': self.B_econ, 'B_outer': self.B_outer,
            'B': self.B, 'B_hat': self.B_hat, 'B_tilde': self.B_tilde,
            'MC': self.MC, 'MC_hat': self.MC_hat, 'CC': self.CC,
            'E': self.E, 'L': self.L, 'N': self.N,
            'Xi': self.Xi, 'R_avail': self.R_avail, 'B_star': self.B_star,
            'counter': self.counter, 't_elapsed': self.t_elapsed,
        })

    def run(self, T=260, check_stability=False):
        """Run for T weekly periods. Schedules a default event at t=5 if none set."""
        if not self.events:
            self.schedule_event(t_event=5)
        if check_stability:
            self._stability_check()
        for _ in range(T):
            self.step()

    def history_arrays(self):
        """Return simulation history as a dict of numpy arrays, one per variable."""
        if not self.history:
            return {}
        return {k: np.array([h[k] for h in self.history]) for k in self.history[0]}

    def attempt_reset(self, delta_id, cds_new=None, b0_new=None):
        """
        Attempt an identity reset (Path 1 recovery, Section 5.2).
        Requires delta_id > Delta_min AND societal acceptance A(t) >= A_min.
        Returns True if the reset succeeded, False otherwise.
        """
        p = self.p
        if self.Xi != 1:
            return False

        # Acceptance grows with elapsed time but is permanently capped by event severity
        A = p['A_inf'] * (1 - np.exp(-p['alpha_A'] * self.t_elapsed)) * (1 - p['sigma_A'] * p.get('L0', 0.85))
        if delta_id <= p['Delta_min'] or A < p['A_min']:
            return False

        # Reset state — brand re-enters elastic regime with stigma discount applied
        self.Xi      = 0
        self.MC      = p['phi_s'] * (1 - delta_id)
        self.MC_hat  = self.MC
        self.B_hat   = 0.0
        self.B_hat_prev = 0.0
        self.B_tilde = 0.0
        self.counter = 0

        cds_n = cds_new if cds_new is not None else p['CDS']
        b0_n  = b0_new  if b0_new  is not None else p['B0']
        self.B_star = max(
            b0_n * np.exp(-p['beta'] * cds_n) * (1 + p['delta'] * self.N) + p['gamma'] * self.MC,
            p['B_floor']
        )
        return True

    def _stability_check(self):
        """
        Pre-run sanity check against the three instability conditions from Section 4.11.
        Prints warnings but doesn't stop the simulation — you might be deliberately
        running a collapse scenario.
        """
        p = self.p
        A_max  = 1 + p['lam'] + p['mu']
        eta_E, zeta = p['eta_E'], p['zeta']
        mc_dag = (-eta_E + np.sqrt(eta_E**2 + 4 * zeta * eta_E)) / (2 * zeta)

        warns = []
        ratio = p['b'] / (p['a'] * A_max)
        if ratio < 1.5:
            warns.append(f"  b/(a*A_max) = {ratio:.2f} < 1.5  →  survival equilibrium unlikely")
        if mc_dag <= 0.50:
            warns.append(f"  MC† = {mc_dag:.2f} <= 0.50  →  healing window very narrow")
        if p['kappa_s'] < 1.50:
            warns.append(f"  kappa_s = {p['kappa_s']:.2f} < 1.50  →  saturation not doing much")

        if warns:
            print("Stability warnings:")
            for w in warns:
                print(w)
            print()


# ------------------------------------------------------------------
# Parameter presets
# ------------------------------------------------------------------

def identity_params():
    """
    Default parameters for an identity brand (Yeezy-type architecture).
    Values are midpoints of the ranges in Table 7 / Appendix B.
    CDS defaults to 0.685 — the Yeezy calibration from the paper.
    Swap CDS before running if you're modelling a different brand.
    """
    return {
        # Threshold and norm dynamics
        'B0':       0.325,   # baseline collapse threshold before CDS compression
        'B_floor':  0.17,    # hard floor — no brand collapses from a single tweet
        'beta':     0.50,    # how aggressively CDS compresses the threshold
        'delta':    0.13,    # sensitivity of threshold to norm context N
        'kappa':    0.03,    # rate at which backlash erodes norms
        'gamma':    0.125,   # threshold resistance from accumulating MC

        # Propagation lags
        'tau_B':    4.5,     # weeks for backlash to propagate to audience belief
        'tau_MC':   3.0,     # weeks for institutional partners to process MC

        # Event injection amplifiers (how much CDS worsens the initial pulse)
        'gamma_c':  1.00,
        'gamma_e':  0.80,

        # Backlash dynamics per audience segment
        'g_c':  0.50,  'd_c':  0.200,  'phi_c':  0.200,   # core tribe
        'g_e':  0.35,  'd_e':  0.275,  'phi_e':  0.100,   # economic partners
        'g_o':  0.40,  'd_o':  0.350,  'phi_o':  0.065,   # outer/media

        'MC_gate': 0.30,    # MC must cross this before cancel-cycle feedback activates
        'psi':     0.175,   # outer -> core bleed-through per period
        'xi':      0.100,   # outer -> econ bleed-through per period

        # Response budget
        'r':       0.45,
        'R_total': 0.80,
        'rho':     0.275,   # response depletes budget at this rate
        'eta':     0.10,    # budget replenishment per period
        'phi_R':   0.04,    # passive exhaustion from sustained media exposure

        # Audience weights (must sum to 1)
        'w_c': 0.575,
        'w_e': 0.250,
        'w_o': 0.175,

        # Meaning collapse
        'a':       0.125,   # base backlash-to-MC conversion
        'kappa_s': 3.25,    # saturation coefficient — higher = more diminishing returns
        'lam':     0.65,    # legitimacy amplifies MC damage
        'mu':      0.375,   # CDS amplifies MC sensitivity
        'theta':   0.115,   # velocity channel weight
        'omega':   0.175,   # velocity cap per period
        'b':       0.34,    # base MC recovery rate
        'gamma_r': 0.10,    # intrinsic quadratic resistance (prevents full absorption)
        'lam_CC':  0.15,    # quadratic damping on CC at high levels
        'alpha_coll': 3.25, # damage multiplier post-collapse
        'kappa_b':    10.0, # recovery suppressor post-collapse

        # Recovery capacity
        'eta_E':  0.25,     # regeneration rate when MC is low
        'zeta':   0.115,    # quadratic depletion rate when MC is high
        'E_min':  0.15,     # floor — organisation never fully loses capacity to act

        # Capital collapse pathways
        'c':  0.35,    # chronic meaning pathway weight
        'v':  0.65,    # velocity shock weight
        'e':  0.225,   # direct economic backlash weight
        'k':  0.175,   # CC stabilisation rate

        # Legitimacy decay
        'theta_L': 0.04,

        # Regime transition
        'tau_min': 4,        # consecutive above-threshold periods needed for collapse

        # Acceptance and recovery (Section 5.3)
        'phi_s':     0.75,
        'Delta_min': 0.60,
        'A_inf':     0.65,
        'A_min':     0.55,
        'alpha_A':   0.035,  # per weekly period; half-ceiling reached ~23 weeks
        'sigma_A':   0.55,

        # Default scenario inputs (override per scenario)
        'CDS': 0.685,
        'PD':  0.85,
        'L0':  0.90,
        'D':   0.90,
        'N0':  0.0,
    }


def institutional_params():
    """
    Default parameters for an institutional brand (slower dynamics, higher threshold,
    CC-first pathway dominates over MC pathway).
    """
    return {
        'B0':       0.65,
        'B_floor':  0.125,
        'beta':     0.225,
        'delta':    0.07,
        'kappa':    0.025,
        'gamma':    0.075,
        'tau_B':    6.5,
        'tau_MC':   4.5,
        'gamma_c':  0.45,
        'gamma_e':  0.60,
        'g_c':  0.225, 'd_c':  0.325, 'phi_c':  0.065,
        'g_e':  0.300, 'd_e':  0.225, 'phi_e':  0.100,
        'g_o':  0.275, 'd_o':  0.400, 'phi_o':  0.050,
        'MC_gate': 0.425,
        'psi':     0.065,
        'xi':      0.140,
        'r':       0.35,
        'R_total': 0.85,
        'rho':     0.20,
        'eta':     0.10,
        'phi_R':   0.03,
        'w_c': 0.325,
        'w_e': 0.425,
        'w_o': 0.250,
        'a':       0.09,
        'kappa_s': 2.25,
        'lam':     0.325,
        'mu':      0.20,
        'theta':   0.085,
        'omega':   0.125,
        'b':       0.23,
        'gamma_r': 0.065,
        'lam_CC':  0.12,
        'alpha_coll': 2.0,
        'kappa_b':    6.5,
        'eta_E':  0.20,
        'zeta':   0.09,
        'E_min':  0.115,
        'c':  0.25,
        'v':  0.30,
        'e':  0.425,
        'k':  0.175,
        'theta_L': 0.04,
        'tau_min': 8,
        'phi_s':     0.425,
        'Delta_min': 0.325,
        'A_inf':     0.80,
        'A_min':     0.40,
        'alpha_A':   0.05,
        'sigma_A':   0.325,
        'CDS': 0.40,
        'PD':  0.70,
        'L0':  0.75,
        'D':   0.80,
        'N0':  0.0,
    }


# ------------------------------------------------------------------
# Scenario runner
# ------------------------------------------------------------------

def run(brand_type='identity', T=260, overrides=None, events=None,
        check_stability=False):
    """
    Convenience wrapper. Pass overrides={} to change specific parameters
    without rebuilding the whole dict. Events is a dict of {period: {L0, D}}.
    """
    params = identity_params() if brand_type == 'identity' else institutional_params()
    if overrides:
        params.update(overrides)

    model = BrandCollapseModel(params=params)

    if events:
        for t_ev, ev in events.items():
            model.schedule_event(t_ev, **ev)
    else:
        model.schedule_event(5)

    model.run(T, check_stability=check_stability)
    return model


# ------------------------------------------------------------------
# Colour palette and plot defaults
# ------------------------------------------------------------------

COLOURS = {
    'B':      '#E63946',
    'B_hat':  '#FF6B6B',
    'B_tilde':'#F4845F',
    'MC':     '#1D3557',
    'MC_hat': '#457B9D',
    'CC':     '#E9A820',
    'E':      '#06D6A0',
    'L':      '#7B2D8E',
    'N':      '#2A9D8F',
    'B_star': '#6C757D',
    'Xi':     '#D62828',
    'R':      '#2D6A4F',
}

plt.rcParams.update({
    'font.family': 'sans-serif',
    'font.size':   11,
    'axes.grid':   True,
    'grid.alpha':  0.3,
    'figure.dpi':  150,
})


# ------------------------------------------------------------------
# Figure functions
# ------------------------------------------------------------------

def fig_time_series(model, save_path=None, title=None):
    """Core variable time series for a single run."""
    C = COLOURS
    h = model.history_arrays()
    T = len(h['B'])
    t = np.arange(1, T + 1)

    fig, axes = plt.subplots(3, 1, figsize=(14, 12), sharex=True)

    ax = axes[0]
    ax.plot(t, h['B'],       color=C['B'],      lw=2.0, label='B(t)')
    ax.plot(t, h['B_hat'],   color=C['B_hat'],  lw=1.8, ls='--', label='B̂(t)')
    ax.plot(t, h['B_tilde'], color=C['B_tilde'],lw=1.5, ls=':',  label='B̃(t)')
    ax.plot(t, h['B_star'],  color=C['B_star'], lw=1.5, ls='-.', label='B*(t)')
    ax.fill_between(t, 0, h['Xi'], alpha=0.08, color='red', label='Ξ=1')
    ax.set_ylabel('Value'); ax.set_ylim(-0.05, 1.05)
    ax.legend(loc='upper right', fontsize=9, ncol=2)
    ax.set_title(title or 'Brand Collapse Model — core dynamics', fontsize=13, fontweight='bold')

    ax = axes[1]
    ax.plot(t, h['MC'],     color=C['MC'],     lw=2.0, label='MC(t)')
    ax.plot(t, h['MC_hat'], color=C['MC_hat'], lw=1.8, ls='--', label='MĈ(t)')
    ax.plot(t, h['CC'],     color=C['CC'],     lw=2.0, label='CC(t)')
    ax.plot(t, h['L'],      color=C['L'],      lw=1.2, ls=':', alpha=0.7, label='L(t)')
    ax.set_ylabel('Value'); ax.set_ylim(-0.05, 1.05)
    ax.legend(loc='upper right', fontsize=9)

    ax = axes[2]
    ax.plot(t, h['E'], color=C['E'], lw=2.5, label='E(t)')
    mc_dag = (-model.p['eta_E'] + np.sqrt(model.p['eta_E']**2 + 4*model.p['zeta']*model.p['eta_E'])) / (2*model.p['zeta'])
    ax.axhline(mc_dag, color='grey', ls=':', lw=1.0, alpha=0.6, label=f'MC† ≈ {mc_dag:.2f}')
    ax.fill_between(t, 0, h['Xi'], alpha=0.06, color='red')
    ax.set_xlabel('Time (weeks)'); ax.set_ylabel('E(t)'); ax.set_ylim(-0.05, 1.05)
    ax.legend(loc='upper right', fontsize=9)

    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, bbox_inches='tight')
    plt.close(fig)


def fig_phase(model, save_path=None):
    """MC vs CC phase portrait — shows the temporal ordering of collapse."""
    h  = model.history_arrays()
    mc, cc = h['MC'], h['CC']
    norm_t = np.linspace(0, 1, len(mc))

    fig, ax = plt.subplots(figsize=(8, 8))
    ax.plot([0, 1], [0, 1], color='lightgrey', lw=1, ls='--', alpha=0.5)
    sc = ax.scatter(mc, cc, c=norm_t, cmap='RdYlBu_r', s=14, alpha=0.7, zorder=3)
    ax.scatter(mc[0],  cc[0],  s=200, c='green', marker='o', zorder=5, edgecolors='black', lw=1.5, label='Start')
    ax.scatter(mc[-1], cc[-1], s=200, c='red',   marker='X', zorder=5, edgecolors='black', lw=1.5, label='End')
    ax.set_xlabel('MC — Meaning Collapse', fontsize=12)
    ax.set_ylabel('CC — Capital Collapse',  fontsize=12)
    ax.set_xlim(-0.05, 1.05); ax.set_ylim(-0.05, 1.05); ax.set_aspect('equal')
    ax.legend(loc='upper left', fontsize=10)
    ax.set_title('MC vs CC Phase Trajectory', fontsize=13, fontweight='bold')
    plt.colorbar(sc, ax=ax, label='Time (normalised)')
    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, bbox_inches='tight')
    plt.close(fig)


def fig_three_regimes(save_path=None):
    """
    The three long-run outcomes — survival, critical slow decay, collapse —
    from the same equation structure, different parameter configurations.
    """
    C = COLOURS
    T = 200
    scenarios = [
        ('Survival (Mode A)',
         {'L0': 0.50, 'D': 0.50, 'CDS': 0.25, 'R_total': 1.0,
          'a': 0.12, 'b': 0.35, 'lam': 0.50, 'mu': 0.40,
          'kappa_s': 3.0, 'eta_E': 0.25, 'zeta': 0.12,
          'gamma': 0.15, 'gamma_r': 0.10, 'E_min': 0.15},
         '#2A9D8F'),
        ('Slow Decay (Critical)',
         {'L0': 0.65, 'D': 0.55, 'CDS': 0.50,
          'kappa_s': 2.0, 'eta_E': 0.25, 'zeta': 0.15,
          'gamma_r': 0.05, 'lam': 0.60, 'mu': 0.40},
         '#E9A820'),
        ('Rapid Collapse (Mode B)',
         {'L0': 0.85, 'D': 0.85, 'CDS': 0.70,
          'a': 0.15, 'b': 0.25, 'lam': 0.65, 'mu': 0.45,
          'kappa_s': 1.5, 'eta_E': 0.15, 'zeta': 0.25,
          'gamma': 0.06, 'gamma_r': 0.02},
         '#E63946'),
    ]

    fig, axes = plt.subplots(1, 3, figsize=(18, 7), sharey=True)
    for ax, (label, ov, col) in zip(axes, scenarios):
        m = run('identity', T=T, overrides=ov, events={5: {'L0': ov['L0'], 'D': ov['D']}})
        h = m.history_arrays()
        t = np.arange(1, T + 1)
        ax.plot(t, h['B'],     color=C['B'],     lw=2.0, label='B(t)')
        ax.plot(t, h['B_hat'], color=C['B_hat'], lw=1.5, ls='--', label='B̂(t)')
        ax.plot(t, h['MC'],    color=C['MC'],    lw=2.0, label='MC(t)')
        ax.plot(t, h['CC'],    color=C['CC'],    lw=2.0, label='CC(t)')
        ax.plot(t, h['E'],     color=C['E'],     lw=2.0, label='E(t)')
        ax.plot(t, h['B_star'],color=C['B_star'],lw=1.2, ls='-.', label='B*(t)')
        ax.fill_between(t, 0, h['Xi'], alpha=0.08, color='red')
        ax.set_title(label, fontsize=12, fontweight='bold', color=col)
        ax.set_xlabel('Time (weeks)'); ax.set_ylim(-0.05, 1.05)
        if ax == axes[0]:
            ax.set_ylabel('Value')
        ax.legend(fontsize=7, loc='right')

    fig.suptitle('Regime Demonstration: Survival / Slow Decay / Rapid Collapse',
                 fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, bbox_inches='tight')
    plt.close(fig)


def fig_cds_comparison(save_path=None):
    """P1: same event, same severity, different CDS — different outcome."""
    C = COLOURS
    T = 200
    base = {'L0': 0.60, 'D': 0.60}
    configs = [
        ('High CDS = 0.70', {**base, 'CDS': 0.70}, '#E63946'),
        ('Low CDS = 0.25',  {**base, 'CDS': 0.25}, '#2A9D8F'),
    ]
    fig, axes = plt.subplots(1, 2, figsize=(16, 7), sharey=True)
    for ax, (label, ov, col) in zip(axes, configs):
        m = run('identity', T=T, overrides=ov, events={5: {'L0': ov['L0'], 'D': ov['D']}})
        h = m.history_arrays()
        t = np.arange(1, T + 1)
        ax.plot(t, h['B'],     color=C['B'],     lw=2.0)
        ax.plot(t, h['B_hat'], color=C['B_hat'], lw=1.5, ls='--')
        ax.plot(t, h['MC'],    color=C['MC'],    lw=2.0, label='MC')
        ax.plot(t, h['CC'],    color=C['CC'],    lw=2.0, label='CC')
        ax.plot(t, h['E'],     color=C['E'],     lw=1.8, label='E')
        ax.plot(t, h['B_star'],color=C['B_star'],lw=1.2, ls='-.')
        ax.fill_between(t, 0, h['Xi'], alpha=0.08, color='red')
        ax.set_title(label, fontsize=12, fontweight='bold', color=col)
        ax.set_ylim(-0.05, 1.05)
        ax.legend(fontsize=8, loc='right')
    fig.suptitle('P1: Threshold Asymmetry — CDS as Structural Predictor',
                 fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, bbox_inches='tight')
    plt.close(fig)


def fig_diffusion(save_path=None):
    """P2 / P6a: fast spike vs slow sustained outrage at matched total backlash."""
    C = COLOURS
    T   = 200
    fast = run('identity', T=T, overrides={'D': 0.95, 'L0': 0.80},
               events={5: {'L0': 0.80, 'D': 0.95}}).history_arrays()
    slow = run('identity', T=T, overrides={'D': 0.40, 'L0': 0.80},
               events={5: {'L0': 0.80, 'D': 0.40}}).history_arrays()
    t = np.arange(1, T + 1)

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes[0,0].plot(t, fast['MC'], color='#E63946', lw=2, label='Fast (D=0.95)')
    axes[0,0].plot(t, slow['MC'], color='#1D3557', lw=2, label='Slow (D=0.40)')
    axes[0,0].set_title('MC(t) — Meaning Collapse'); axes[0,0].legend()

    axes[0,1].plot(t, fast['CC'], color='#E63946', lw=2)
    axes[0,1].plot(t, slow['CC'], color='#1D3557', lw=2)
    axes[0,1].set_title('CC(t) — Capital Collapse')

    axes[1,0].plot(t, fast['B_hat'],   color='#E63946', lw=2)
    axes[1,0].plot(t, slow['B_hat'],   color='#1D3557', lw=2)
    axes[1,0].plot(t, fast['B_tilde'], color='#E63946', lw=1.5, ls=':')
    axes[1,0].plot(t, slow['B_tilde'], color='#1D3557', lw=1.5, ls=':')
    axes[1,0].set_title('B̂(t) vs B̃(t) — Saturation Effect')

    axes[1,1].plot(t, fast['E'], color='#E63946', lw=2)
    axes[1,1].plot(t, slow['E'], color='#1D3557', lw=2)
    axes[1,1].set_title('E(t) — Recovery Capacity')

    fig.suptitle('P2 / P6a: Fast vs Slow Outrage — Velocity & Level Damage',
                 fontsize=14, fontweight='bold')
    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, bbox_inches='tight')
    plt.close(fig)


def fig_response_exhaustion(save_path=None):
    """What happens when the response budget runs dry mid-crisis."""
    T = 200
    high = run('identity', T=T, overrides={'R_total': 1.0, 'L0': 0.70, 'D': 0.70}).history_arrays()
    low  = run('identity', T=T, overrides={'R_total': 0.3, 'L0': 0.70, 'D': 0.70}).history_arrays()
    t = np.arange(1, T + 1)

    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    axes[0].plot(t, high['B'],      color='#2D6A4F', lw=2, label='High budget')
    axes[0].plot(t, low['B'],       color='#E63946', lw=2, label='Low budget')
    axes[0].set_title('B(t) — Backlash'); axes[0].legend()

    axes[1].plot(t, high['R_avail'],color='#2D6A4F', lw=2)
    axes[1].plot(t, low['R_avail'], color='#E63946', lw=2)
    axes[1].set_title('R_avail(t) — Response Budget')

    axes[2].plot(t, high['MC'],     color='#2D6A4F', lw=2)
    axes[2].plot(t, low['MC'],      color='#E63946', lw=2)
    axes[2].set_title('MC(t)')

    fig.suptitle('Response Exhaustion: Containment vs Failure',
                 fontsize=14, fontweight='bold')
    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, bbox_inches='tight')
    plt.close(fig)


def fig_identity_vs_institutional(save_path=None):
    """P4: same event, same severity — meaning pathway vs economic pathway dominance."""
    T  = 200
    ev = {5: {'L0': 0.85, 'D': 0.90}}
    h_id   = run('identity',     T=T, events=ev).history_arrays()
    h_inst = run('institutional', T=T,
                 overrides={'L0': 0.85, 'D': 0.90}, events=ev).history_arrays()
    t = np.arange(1, T + 1)

    fig, axes = plt.subplots(2, 2, figsize=(16, 11))
    axes[0,0].plot(t, h_id['B'],   color='#E63946', lw=2, label='Identity')
    axes[0,0].plot(t, h_inst['B'], color='#1D3557', lw=2, label='Institutional')
    axes[0,0].set_title('B(t), B̂(t), B*(t)'); axes[0,0].legend()

    axes[0,1].plot(t, h_id['MC'],   color='#E63946', lw=2)
    axes[0,1].plot(t, h_inst['MC'], color='#1D3557', lw=2)
    axes[0,1].set_title('MC(t)')

    axes[1,0].plot(t, h_id['CC'],   color='#E63946', lw=2)
    axes[1,0].plot(t, h_inst['CC'], color='#1D3557', lw=2)
    axes[1,0].set_title('CC(t)')

    axes[1,1].plot(t, h_id['Xi'],   color='#E63946', lw=2.5)
    axes[1,1].plot(t, h_inst['Xi'], color='#1D3557', lw=2.5)
    axes[1,1].set_title('Ξ(t) — Regime')

    fig.suptitle('Identity vs Institutional: Same Event, Different Architecture',
                 fontsize=14, fontweight='bold')
    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, bbox_inches='tight')
    plt.close(fig)


def fig_norm_drift(save_path=None):
    """
    500-period simulation showing how repeated sub-threshold events
    gradually compress B*(t) until a moderate event finally triggers collapse.
    The 'suddenly unacceptable' mechanism.
    """
    C = COLOURS
    T = 500
    # Six sub-threshold events spaced roughly every 65 periods
    events = {t_ev: {'L0': 0.50, 'D': 0.55} for t_ev in [5, 65, 130, 210, 300, 400]}
    h = run('identity', T=T, overrides={'CDS': 0.50}, events=events).history_arrays()
    t = np.arange(1, T + 1)

    fig, axes = plt.subplots(2, 1, figsize=(14, 8), sharex=True)
    axes[0].plot(t, h['N'], color=C['N'], lw=2)
    axes[0].set_title('Norm Drift: Cumulative Cultural Pressure')

    axes[1].plot(t, h['B_star'], color=C['B_star'], lw=2, label='B*(t)')
    axes[1].plot(t, h['B'],      color=C['B'],      lw=1.5, alpha=0.7, label='B(t)')
    axes[1].set_title('Threshold Compression + B_floor')
    axes[1].legend()

    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, bbox_inches='tight')
    plt.close(fig)


# ------------------------------------------------------------------
# Main — reproduce all paper figures
# ------------------------------------------------------------------

def main():
    print("Running Brand Collapse Model — Version 3")
    print("Output directory:", OUTPUT_DIR)
    print()

    # Default identity brand run (Yeezy-calibrated parameters)
    print("Running default identity brand scenario...")
    default_model = run('identity', T=260)

    print("Generating figures...")
    fig_time_series(default_model,
                    save_path=os.path.join(OUTPUT_DIR, '01_time_series.png'),
                    title=f"Identity Brand Dynamics (CDS = {default_model.p['CDS']:.3f})")

    fig_phase(default_model,
              save_path=os.path.join(OUTPUT_DIR, '02_phase_portrait.png'))

    fig_three_regimes(save_path=os.path.join(OUTPUT_DIR, '03_three_regimes.png'))

    fig_cds_comparison(save_path=os.path.join(OUTPUT_DIR, '04_cds_comparison.png'))

    fig_diffusion(save_path=os.path.join(OUTPUT_DIR, '05_diffusion.png'))

    fig_response_exhaustion(save_path=os.path.join(OUTPUT_DIR, '06_response_exhaustion.png'))

    fig_identity_vs_institutional(save_path=os.path.join(OUTPUT_DIR, '07_identity_vs_institutional.png'))

    fig_norm_drift(save_path=os.path.join(OUTPUT_DIR, '08_norm_drift.png'))

    print("Done. Figures saved to:", OUTPUT_DIR)
    print()

    # Print key derived quantities for the default parameterisation
    p = default_model.p
    eta_E, zeta = p['eta_E'], p['zeta']
    mc_dag = (-eta_E + np.sqrt(eta_E**2 + 4 * zeta * eta_E)) / (2 * zeta)
    print("Key derived values (default identity params):")
    print(f"  MC†        = {mc_dag:.3f}  (recovery bifurcation point)")
    print(f"  B*(t=0)    = {default_model.p['B0'] * np.exp(-p['beta'] * p['CDS']):.3f}  (initial threshold)")
    print(f"  B_tilde at B_hat=1  = {1/(1+p['kappa_s']):.3f}  (saturation ceiling)")
    print()


if __name__ == '__main__':
    main()