"""
FIGURES - cellulose model 
"""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from cellulose_model import Parameters, Model

T_END_H    = 12.0          
N_POINTS   = 300
THERMAL    = True

COL = dict(cellulose="#4c4c4c", glucose="#1f77b4", cellobiose="#d62728",
           c4="#2ca02c", nolpmo="#1f77b4", lpmo="#d62728", dead="#7f7f7f")


def run(use_lpmo, t_end=T_END_H, **kw):
    m = Model(Parameters(**kw))
    return m.analyse(m.simulate(use_lpmo=use_lpmo, thermal_decay=THERMAL,
                                t_end_h=t_end, n_points=N_POINTS))


def _time_to(d, pct):
    c = d["conversion_pct"]; i = int(np.argmax(c >= pct))
    return d["t"][i] if c[i] >= pct else float("nan")


def figure_conversion(outfile="fig_conversion_lpmo.png"):
    base, lp = run(False), run(True)
    t = base["t"]
    fig, ax = plt.subplots(figsize=(7.2, 5.0))
    ax.plot(t, base["conversion_pct"], color=COL["nolpmo"], lw=2.6,
            label="EG + CBH + BG  (no LPMO)")
    ax.plot(t, lp["conversion_pct"], color=COL["lpmo"], lw=2.6,
            label="EG + CBH + BG + LPMO")
    ax.fill_between(t, base["conversion_pct"], lp["conversion_pct"],
                    where=(lp["conversion_pct"] >= base["conversion_pct"]),
                    color=COL["lpmo"], alpha=0.12, label="LPMO contribution")
    ratio = lp["conversion_pct"][-1]/base["conversion_pct"][-1]
    tb, tl = _time_to(base, 75.0), _time_to(lp, 75.0)
    ax.set_xlim(0, T_END_H); ax.set_ylim(0, 105)
    ax.set_xlabel("Time (h)"); ax.set_ylabel("Cellulose solubilized (%)")
    #ax.set_title("Cellulose solubilization with and without LPMO", fontsize=12)
    ax.legend(fontsize=9, loc="center right", framealpha=0.95)
    ax.grid(alpha=0.25, lw=0.5)

    fig.tight_layout(); fig.savefig(outfile, dpi=170); plt.close(fig)
    print(f"[FIG 1] {outfile}   ratio {ratio:.3f}x  ({T_END_H:.0f} h)")


# ============================================================================
# FIG 2 - species: LEFT axis mM (cellulose, glucose), RIGHT axis uM (cellobiose)
# ============================================================================
def figure_species(outfile="fig_species_lpmo.png"):
  
    base, lp = run(False), run(True)

    mM_max = 1.08*max(base["solid_glucose"].max(), base["glucose"].max(),
                      lp["solid_glucose"].max(),   lp["glucose"].max())
    uM_max = 1.25*1e3*max(base["cellobiose"].max(), lp["cellobiose"].max())

    fig, ax = plt.subplots(1, 2, figsize=(12.6, 5.0))   
    for k, (axis, d, title) in enumerate(
            ((ax[0], base, "A"),
             (ax[1], lp,   "B"))):
        t = d["t"]

        # LEFT axis: mM 
        l1, = axis.plot(t, d["solid_glucose"], color=COL["cellulose"], lw=2.6,
                        label="Initial cellulose")
        l2, = axis.plot(t, d["glucose"], color=COL["glucose"], lw=2.4,
                        label="glucose")
        axis.set_xlim(0, T_END_H); axis.set_ylim(0, mM_max)
        axis.set_xlabel("Time (h)")
        axis.set_ylabel("Cellulose / glucose  (mmol L$^{-1}$)")   # on BOTH panels
        axis.set_title(title, fontsize=11)
        axis.grid(alpha=0.25, lw=0.5)

        axis.legend(loc="upper right")

        # RIGHT axis: uM 
        ax2 = axis.twinx()
        l3, = ax2.plot(t, d["cellobiose"]*1e3, color=COL["cellobiose"], lw=2.4,
                       ls="--", label="cellobiose (right axis)")
        ax2.set_ylim(0, uM_max)
        ax2.set_ylabel("Cellobiose  (µmol L$^{-1}$)",              # on BOTH panels
                       color=COL["cellobiose"])
        ax2.tick_params(axis="y", labelcolor=COL["cellobiose"])

        axis.legend(handles=[l1, l2, l3], fontsize=8.5,
                    loc="center right", framealpha=0.95)

   
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    fig.savefig(outfile, dpi=170); plt.close(fig)

    print(f"[FIG 2] {outfile}   ({T_END_H:.0f} h)")
    for tag, d in (("no LPMO", base), ("+LPMO", lp)):
        oligo = (d["cellotriose"] + d["cellotetraose"]
                 + d["cellopentaose"] + d["cellohexaose"])
        print(f"         {tag:8s} peak glucose {d['glucose'].max():6.2f} mM | "
              f"peak cellobiose {d['cellobiose'].max()*1e3:7.2f} uM")
        print(f"         {'':8s} not plotted - peak C3 {d['cellotriose'].max()*1e3:.4f} uM, "
              f"C4 {d['cellotetraose'].max()*1e3:.4f} uM, "
              f"C5 {d['cellopentaose'].max()*1e3:.4f} uM, "
              f"C6 {d['cellohexaose'].max()*1e3:.4f} uM "
              f"(total C3-C6 peak {oligo.max()*1e3:.4f} uM)")




if __name__ == "__main__":
    figure_conversion()
    figure_species()
   