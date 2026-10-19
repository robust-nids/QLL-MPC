# run with:
#   python visualize.py [model] [--property k]
#   e.g. python visualize.py 1.1
# Recreates the "Neural Network" panel of Fig. 7 in julian2016.pdf (advisories for a head-on encounter)
# for the official network, followed by every finetuned network of the same model, saved in figs/
# With --property k it instead shows the slice of the input space through property k's region: the region
# outlined in red, and in red where the property fails

import argparse
import glob
import os
from collections import namedtuple

import matplotlib.pyplot as plt
import torch
from matplotlib.colors import ListedColormap
from matplotlib.patches import Annulus, Patch, Wedge
from matplotlib.path import Path
from matplotlib.ticker import FuncFormatter
from matplotlib.transforms import Affine2D

from finetune import HERE, IN_MAX, IN_MEAN, IN_MIN, PI, PROPS, REGIONS, load_onnx, normalise, sample_region

# what is plotted: the intruder's position (x, y) in kft over xlim and ylim, every other input fixed,
# and optionally a property whose region is drawn
View = namedtuple("View", "xlim ylim psi v_own v_int ticks prop")

# Fig. 7's encounter: the intruder flies straight at the ownship (psi = pi). The paper does not give the
# speeds; 200 ft/s for both best matches Fig. 7's alerting region with acasXu_1_1
FIG7 = View(xlim=(-6, 36), ylim=(-10, 10), psi=PI, v_own=200.0, v_int=200.0,
            ticks=([0, 10, 20, 30], [-10, -5, 0, 5, 10]), prop=None)

RES = 250  # the resolution of the paper's plots
RED = (0.85, 0, 0)

# advisory colours in the network's output order: COC, weakLeft, weakRight, strongLeft, strongRight
COLOURS = [(1, 1, 1), (0.5, 0.5, 0.5), (0.7, 0.9, 0), (0, 0, 0.5), (0, 0.6, 0)]
# the legend in the paper's order, (output index, label); positive turn rates are left turns
LEGEND = [(0, "COC"), (4, r"$-3.0\,^{\circ}/\mathrm{s}$"), (2, r"$-1.5\,^{\circ}/\mathrm{s}$"),
          (1, r"$1.5\,^{\circ}/\mathrm{s}$"), (3, r"$3.0\,^{\circ}/\mathrm{s}$")]

# the paper's aircraft outline in pt, nose along +x, around the aircraft's position
AIRCRAFT = [(7.01, -0.01), (7.01, -0.05), (7.0, -0.11), (6.88, -0.2), (6.42, -0.43), (5.78, -0.58),
            (4.97, -0.65), (4.27, -0.69), (2.15, -0.69), (-3.41, -6.26), (-3.58, -6.34), (-4.03, -6.31),
            (-4.23, -6.23), (-2.2, -3.0), (-1.4, -0.69), (-4.66, -0.7), (-6.09, -0.53), (-7.81, -2.63),
            (-8.45, -2.63), (-7.94, -0.2), (-8.58, -0.03), (-8.58, 0.03), (-7.94, 0.19), (-8.45, 2.63),
            (-7.81, 2.63), (-6.09, 0.53), (-4.66, 0.69), (-1.4, 0.68), (-2.2, 3.0), (-4.23, 6.23),
            (-4.03, 6.3), (-3.58, 6.33), (-3.41, 6.26), (2.15, 0.68), (4.27, 0.68), (4.97, 0.65),
            (5.78, 0.57), (6.42, 0.42), (6.88, 0.19), (7.0, 0.1), (7.01, 0.04), (7.01, -0.01)]

# the paper's pgfplots style: Times text, Computer Modern numbers, 8 pt, ticks inside on every side
STYLE = {
    "font.family": "serif",
    "font.serif": ["Times New Roman", "Nimbus Roman", "STIXGeneral"],
    "mathtext.fontset": "cm",
    "font.size": 8,
    "axes.linewidth": 0.4,
    "axes.titlepad": 9,
    "xtick.direction": "in",
    "ytick.direction": "in",
    "xtick.top": True,
    "ytick.right": True,
    "xtick.major.size": 4.25,
    "ytick.major.size": 4.25,
    "xtick.major.width": 0.2,
    "ytick.major.width": 0.2,
    "xtick.major.pad": 2.5,
    "ytick.major.pad": 3.5,
}

# panel size and spacing in pt, as in the paper
AX_W, AX_H, AX_GAP = 136.58, 74.23, 28.35

def region_boxes(k):
    """property k's region as boxes [n, 5, 2] in problem units, clipped to the valid inputs"""
    boxes = torch.tensor(REGIONS[k], dtype=torch.float32)
    return torch.stack([torch.maximum(boxes[..., 0], IN_MIN), torch.minimum(boxes[..., 1], IN_MAX)], dim=-1)

@torch.no_grad()
def property_view(k, net):
    """the slice through property k's region where `net` breaks it worst (its centre if it never does),
    zoomed out to the region's (rho, theta) sectors around the ownship"""
    X = sample_region(k, 100000)
    value = PROPS[k](1e6, *net(X).T)  # p = 1e6: practically exact, > 0 where the property fails
    boxes = region_boxes(k)
    centre = (boxes[..., 0].min(0).values + boxes[..., 1].max(0).values) / 2
    worst = X[value.argmax()] * (IN_MAX - IN_MIN) + IN_MEAN  # back to problem units
    psi, v_own, v_int = (worst if value.max() > 0 else centre)[2:].tolist()
    rho = torch.cat([b[0, 0] + torch.rand(10000) * (b[0, 1] - b[0, 0]) for b in boxes]) / 1000
    theta = torch.cat([b[1, 0] + torch.rand(10000) * (b[1, 1] - b[1, 0]) for b in boxes])
    xs = torch.cat([rho * torch.cos(theta), torch.zeros(1)])  # the sectors, and the ownship
    ys = torch.cat([rho * torch.sin(theta), torch.zeros(1)])
    cx, cy = (xs.max() + xs.min()).item() / 2, (ys.max() + ys.min()).item() / 2
    half = 0.55 * max((xs.max() - xs.min()).item(), (ys.max() - ys.min()).item())
    return View(xlim=(cx - half, cx + half), ylim=(cy - half, cy + half), psi=psi, v_own=v_own, v_int=v_int, ticks=None, prop=k)

@torch.no_grad()
def scores(net, view):
    """the network's scores and inputs with the intruder at each point of the plot"""
    x = view.xlim[0] + (torch.arange(RES) + 0.5) * (view.xlim[1] - view.xlim[0]) / RES
    y = view.ylim[0] + (torch.arange(RES) + 0.5) * (view.ylim[1] - view.ylim[0]) / RES
    Y, X = torch.meshgrid(y * 1000, x * 1000, indexing="ij")  # kft -> ft
    rho, theta = torch.hypot(X, Y), torch.atan2(Y, X)
    inputs = torch.stack([rho, theta, torch.full_like(rho, view.psi), torch.full_like(rho, view.v_own),
                          torch.full_like(rho, view.v_int)], dim=-1).reshape(-1, 5)
    return net(normalise(inputs)), inputs

def advisories(net, view=FIG7):
    """the advisory (index of the lowest score) with the intruder at each point of the plot"""
    return scores(net, view)[0].argmin(1).reshape(RES, RES)

def draw_aircraft(ax, x, y, heading):
    path = Affine2D().rotate(heading).transform_path(Path(AIRCRAFT, closed=True))
    size = 2 * abs(path.vertices).max()  # a marker is scaled so its largest coordinate is markersize / 2
    ax.plot(x, y, marker=path, markersize=size, color="black", markeredgewidth=0)

def panel(ax, net, title, view):
    y, inputs = scores(net, view)
    extent = (*view.xlim, *view.ylim)
    ax.imshow(y.argmin(1).reshape(RES, RES), cmap=ListedColormap(COLOURS), vmin=-0.5, vmax=4.5,
              extent=extent, origin="lower", aspect="auto", interpolation="nearest")
    if view.prop is not None:
        boxes = region_boxes(view.prop)
        inside = ((inputs[:, None] >= boxes[..., 0]) & (inputs[:, None] <= boxes[..., 1])).all(-1).any(-1)
        fails = inside & (PROPS[view.prop](1e6, *y.T) > 0)
        ax.imshow(fails.reshape(RES, RES), cmap=ListedColormap([(0, 0, 0, 0), RED]), vmin=0, vmax=1,
                  extent=extent, origin="lower", aspect="auto", interpolation="nearest")
        for b in boxes:  # the region's (rho, theta) sector, a whole ring if theta goes all the way round
            r, width = b[0, 1].item() / 1000, (b[0, 1] - b[0, 0]).item() / 1000
            theta = b[1].rad2deg().tolist()  # degrees, counterclockwise from +x
            sector = Annulus((0, 0), r, width) if theta[1] - theta[0] > 359.9 else Wedge((0, 0), r, *theta, width=width)
            sector.set(fill=False, color=RED, lw=0.6)
            ax.add_patch(sector)
    draw_aircraft(ax, 0, 0, 0)
    # the intruder's heading, where the paper draws it: 88% across and 85% up
    draw_aircraft(ax, view.xlim[0] + 0.881 * (view.xlim[1] - view.xlim[0]), view.ylim[0] + 0.85 * (view.ylim[1] - view.ylim[0]), view.psi)
    ax.set_xlim(view.xlim)
    ax.set_ylim(view.ylim)
    if view.ticks:
        ax.set_xticks(view.ticks[0], [f"${t}$" for t in view.ticks[0]])
        ax.set_yticks(view.ticks[1], [f"${t}$" for t in view.ticks[1]])
    else:
        ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _: f"${v:g}$"))
        ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"${v:g}$"))
    ax.set_xlabel("Downrange (kft)")
    ax.set_title(title)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("model", help="official network, e.g. 1.1")
    parser.add_argument("--property", type=int, choices=range(10), help="show the slice through this property's region")
    args = parser.parse_args()
    model = args.model.replace(".", "_")

    paths = [os.path.join(HERE, "official_nets", f"acasXu_{model}.onnx")]
    paths += sorted(glob.glob(os.path.join(HERE, "finetuned_nets", f"acasXu_{model}_*.onnx")))
    titles = [f"Official Network {model}"]
    for path in paths[1:]:  # acasXu_<model>_<properties>[_<sampling>].onnx
        props, *sampling = os.path.basename(path).removesuffix(".onnx").removeprefix(f"acasXu_{model}_").split("_")
        title = f"Finetuned: {'property' if len(props) == 1 else 'properties'} {', '.join(props)}"
        titles.append(title + "".join(f" ({s})" for s in sampling))

    view = FIG7 if args.property is None else property_view(args.property, load_onnx(paths[0]))
    ax_h = AX_H if view.ticks else AX_W  # property views keep x and y to the same scale

    plt.rcParams.update(STYLE)
    width = len(paths) * AX_W + (len(paths) - 1) * AX_GAP
    fig = plt.figure(figsize=(width / 72, ax_h / 72))
    for i, (path, title) in enumerate(zip(paths, titles)):
        ax = fig.add_axes((i * (AX_W + AX_GAP) / width, 0, AX_W / width, 1))
        panel(ax, load_onnx(path), title, view)
        if i == 0:
            ax.set_ylabel("Crossrange (kft)")

    handles = [Patch(color="none")] + [Patch(facecolor=COLOURS[i], edgecolor="black", linewidth=0.4) for i, _ in LEGEND]
    labels = ["Advisories:"] + [label for _, label in LEGEND]
    if view.prop is not None:
        handles += [Patch(fill=False, edgecolor=RED, linewidth=0.6), Patch(facecolor=RED, edgecolor=RED)]
        labels += [f"property {view.prop}", "fails"]
        fig.text(0.5, -60 / ax_h, rf"$\psi = {view.psi:.2f}$ rad, $v_{{own}} = {view.v_own:.0f}$ ft/s, $v_{{int}} = {view.v_int:.0f}$ ft/s",
                 ha="center")
    fig.legend(handles, labels, loc="center", bbox_to_anchor=(0.5, -39.4 / ax_h), ncol=len(handles), frameon=False,
               handlelength=1.25, handleheight=1.25, handletextpad=0.45, columnspacing=1.25)

    name = "advisories" if view.prop is None else f"property{view.prop}"
    path = os.path.join(HERE, "figs", f"acasXu_{model}_{name}.png")
    os.makedirs(os.path.dirname(path), exist_ok=True)
    fig.savefig(path, dpi=300, bbox_inches="tight")
    print(f"saved {path}")

if __name__ == "__main__":
    main()
