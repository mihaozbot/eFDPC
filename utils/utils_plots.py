import os
import numpy as np
import matplotlib.pyplot as plt
import torch
from matplotlib import cm
from matplotlib.patches import Ellipse
from matplotlib.ticker import ScalarFormatter
from matplotlib.pyplot import get_cmap
from matplotlib.lines import Line2D
from sklearn.decomposition import PCA

    

def plot_first_feature_combinations(
    dataset,
    model,
    N_max,
    num_sigma,
    N_combinations: int = 5,
    colormap: str = "tab10",
    flag_unlabeled: bool = False,
    flag_axis: bool = True,
):
    data, labels = dataset
    data   = data.cpu().detach().numpy()   if torch.is_tensor(data)   else data
    labels = labels.cpu().detach().numpy() if torch.is_tensor(labels) else labels
    feature_dim = int(model.feature_dim.item())

    if data.ndim == 1:
        data = data[None, :]
    if labels.ndim == 0:
        labels = labels[None]

    num_samples = min(10_000, len(data))
    idx         = np.random.choice(len(data), num_samples, replace=False)
    data, labels = data[idx], labels[idx]

    n_i = model.n[: model.c].cpu()
    top_clusters_per_class = {}
    for cls in range(model.num_classes):
        c_idx = torch.where(model.cluster_labels[: model.c, cls])[0]
        top_clusters_per_class[cls] = c_idx

    unique_lbls = int(len(np.unique(labels)) + 1)
    cm_vec = get_cmap(colormap)(np.linspace(0, 0.5, unique_lbls - 1)).tolist()
    cm_vec.insert(0, (1, 0, 0, 1))
    label_color = dict(zip(range(unique_lbls), cm_vec))

    data_colors = [
        label_color[(lbl + 1) if flag_unlabeled else lbl] for lbl in labels
    ]

    pairs = [(0, k) for k in range(1, data.shape[1])]
    pairs = pairs[: min(len(pairs), N_combinations)]

    fig, axes = plt.subplots(
        1, len(pairs), figsize=(4 * len(pairs), 4), constrained_layout=True
    )
    axes = np.atleast_1d(axes)

    for ax, (f1, f2) in zip(axes, pairs):
        ax.scatter(
            data[:, f1],
            data[:, f2],
            c=data_colors,
            alpha=0.7,
            s=max(1, min(10, 1000 // max(1, np.unique(data[:, f1]).size))),
        )

        for cls, top_clusters in top_clusters_per_class.items():
            for c in top_clusters:
                if model.n[c] <= N_max:
                    continue

                mu = model.mu[c].cpu().numpy()
                mu_pair = mu[[f1, f2]]

                S = model.S[c]
                if S.dim() == 1:
                    var = S.cpu().numpy() /model.n[c].item()
                    cov = np.diag(var)
                else:
                    cov = S.cpu().numpy() / model.n[c].item()
                    

                cov_pair = cov[[f1, f2]][:, [f1, f2]]

                vals, vecs = np.linalg.eigh(cov_pair)
                angle = np.degrees(np.arctan2(*vecs[:, 0][::-1]))
                width, height = num_sigma * np.sqrt(vals)

                if model.cluster_labels[c, 0]:
                    ec = label_color[0]
                else:
                    ec = label_color[int(torch.where(model.cluster_labels[c])[0])]

                dark = tuple(0.8 * np.array(ec[:3])) + (1,)
                ell = Ellipse(
                    xy=(mu_pair[0], mu_pair[1]),
                    width=width,
                    height=height,
                    angle=angle,
                    edgecolor=dark,
                    lw=2,
                    facecolor="none",
                )
                ax.add_patch(ell)

                ax.scatter(mu_pair[0], mu_pair[1], color="black", s=50, marker=".")

        ax.xaxis.set_major_formatter(ScalarFormatter(useMathText=True))
        ax.yaxis.set_major_formatter(ScalarFormatter(useMathText=True))
        ax.ticklabel_format(style="sci", axis="both", scilimits=(0, 0))
        ax.set_xlabel(f"$x_{{{f1+1}}}$")
        ax.set_ylabel(f"$x_{{{f2+1}}}$")
        ax.grid(False)

        if not flag_axis:
            ax.set_xticks([])
            ax.set_yticks([])
            ax.set_xlabel("")
            ax.set_ylabel("")

    return fig


def plot_pca_reduction(dataset, model, N_max, num_sigma, N_combinations=5, colormap='tab10', 
                                    flag_unlabeled=False, flag_axis=True):
    data, labels = dataset
    data = data.cpu().detach().numpy() if isinstance(data, torch.Tensor) else data
    labels = labels.cpu().detach().numpy() if isinstance(labels, torch.Tensor) else labels

    if len(data.shape) == 1:
        data = np.expand_dims(data, axis=0)
    if np.isscalar(labels):
        labels = np.array([labels])

    num_samples = min(10000, len(data))
    sampled_indices = np.random.choice(len(data), num_samples, replace=False)

    data = data[sampled_indices]
    labels = labels[sampled_indices]

    pca = PCA(n_components=2)
    data_proj = pca.fit_transform(data)

    unique_labels = len(np.unique(labels)) + 1
    label_colors = plt.get_cmap(colormap)(np.linspace(0, 0.5, unique_labels - 1))
    label_colors = label_colors.tolist()
    unlabelled_color = [1, 0, 0, 1]
    label_colors.insert(0, unlabelled_color)
    label_color_dict = dict(zip(range(unique_labels), label_colors))

    if flag_unlabeled:
        data_colors = [label_color_dict[label.item() + 1] for label in labels]
    else:
        try:
            data_colors = [label_color_dict[label.item()] for label in labels]
        except Exception as e:
            print(f"Error in label color assignment: {e}")

    n_i = model.n[:model.c].clone().cpu().detach()
    top_clusters_per_class = {}
    for class_idx in range(model.num_classes):
        cluster_indices = torch.where(model.cluster_labels[:model.c, class_idx].cpu().detach())[0]
        sorted_indices = torch.argsort(n_i[cluster_indices], descending=True)[:1000]
        top_clusters_per_class[class_idx] = cluster_indices[sorted_indices]

    fig, ax = plt.subplots(figsize=(4, 4), constrained_layout=True)

    ax.scatter(data_proj[:, 0], data_proj[:, 1], c=data_colors, alpha=0.7, s=10)

    for class_idx, top_clusters in top_clusters_per_class.items():
        for cluster_idx in top_clusters:
            mu_val = model.mu[cluster_idx].cpu().detach().numpy()
            mu_proj = pca.transform(mu_val.reshape(1, -1))[0]

            if model.n[cluster_idx] > N_max:
                S = model.S[cluster_idx].cpu().detach().numpy()
                n_cluster = float(model.n[cluster_idx].cpu().detach().numpy())

                cov_diag = S / max(n_cluster, 1.0)

                P = pca.components_

                P_scaled = P * cov_diag
                cov_proj = P_scaled @ P.T

                cov_proj = np.asarray(cov_proj)
                if cov_proj.shape != (2, 2):
                    var = np.mean(cov_diag)
                    cov_proj = var * np.eye(2)

                vals, vecs = np.linalg.eigh(cov_proj)
                angle = np.degrees(np.arctan2(vecs[1, 0], vecs[0, 0]))

                factor = num_sigma
                width, height = factor * np.sqrt(vals)

                if model.cluster_labels[cluster_idx, 0].item() == 1:
                    ellipse_color = label_color_dict[0]
                else:
                    active_indices = torch.where(model.cluster_labels[cluster_idx] == 1)[0]
                    if len(active_indices) > 0:
                        ellipse_color = label_color_dict[active_indices[0].item()]
                    else:
                        ellipse_color = [0, 0, 0, 1]

                ellipse_color_rgba = plt.cm.colors.to_rgba(ellipse_color)
                dark_factor = 0.8
                darker_ellipse_color = (ellipse_color_rgba[0] * dark_factor, 
                                          ellipse_color_rgba[1] * dark_factor, 
                                          ellipse_color_rgba[2] * dark_factor, 1)

                ell = Ellipse(xy=mu_proj, width=width, height=height,
                              angle=angle, edgecolor=darker_ellipse_color, lw=2, facecolor='none')
                ax.add_patch(ell)

            ax.scatter(mu_proj[0], mu_proj[1], color='black', s=50, marker='.')

    ax.xaxis.set_major_formatter(ScalarFormatter(useMathText=True))
    ax.yaxis.set_major_formatter(ScalarFormatter(useMathText=True))
    ax.ticklabel_format(style='sci', axis='x', scilimits=(0, 0))
    ax.ticklabel_format(style='sci', axis='y', scilimits=(0, 0))
    ax.set_xlabel("PCA Component 1")
    ax.set_ylabel("PCA Component 2")
    ax.grid(False)

    if not flag_axis:
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_xlabel('')
        ax.set_ylabel('')

    return fig

    

def plot_noniid_partition(y_true_split, num_clients, num_classes, ax=None,
                          count_max=None, show_legend=False, label_size=12,
                          tick_size=11, legend_size=10, cmap_name='tab10',
                          target_area=200.0):
    cmap = plt.get_cmap(cmap_name)
    counts = np.zeros((num_clients, num_classes))
    for k in range(num_clients):
        yk = y_true_split[k]
        yk = yk.cpu().numpy() if hasattr(yk, 'cpu') else np.asarray(yk)
        counts[k] = np.bincount(yk.astype(int), minlength=num_classes)
    if count_max is None:
        count_max = counts.max()
    scale = target_area / max(float(count_max), 1.0)
    if ax is None:
        _, ax = plt.subplots(figsize=(2.7, 2.8))
    xs, ys, sizes, cols = [], [], [], []
    for k in range(num_clients):
        for c in range(num_classes):
            if counts[k, c] > 0:
                xs.append(k + 1); ys.append(c)
                sizes.append(counts[k, c] * scale); cols.append(cmap(c))
    ax.scatter(xs, ys, s=sizes, color=cols, alpha=0.85, edgecolors='white', linewidths=0.3)
    ax.set_xlabel(r"Client $k$", fontsize=label_size)
    ax.set_ylabel("True cluster", fontsize=label_size)
    ax.set_xticks(range(1, num_clients + 1))
    ax.set_yticks(range(num_classes))
    ax.tick_params(labelsize=tick_size)
    ax.set_xlim(0.3, num_clients + 0.7)
    ax.set_ylim(-0.7, num_classes - 0.3)
    ax.grid(True, alpha=0.25, lw=0.4)
    if show_legend:
        refs = [1000, 3000, 6000]
        handles = [ax.scatter([], [], s=r * scale, color='0.45', alpha=0.85) for r in refs]
        leg = ax.legend(handles, [f"{r // 1000}k samples" for r in refs], fontsize=legend_size,
                        loc='upper right', frameon=True, framealpha=0.9, edgecolor='0.6',
                        labelspacing=0.9, handletextpad=0.3, borderpad=0.5)
        leg.get_frame().set_facecolor('white')
    return ax, count_max


def plot_cluster_density_heatmap(model, grid_size=100):
    n_clusters = int(model.c.item() if hasattr(model.c, 'item') else model.c)
    centers = model.mu[:n_clusters].cpu().numpy()
    covariances = (model.S[:n_clusters] / model.n[:n_clusters, None, None]).cpu().numpy()
    samples = model.n[:n_clusters].cpu().numpy()

    n_max = np.max(samples)

    x_min, y_min = centers.min(axis=0) - 2 * np.sqrt(np.max(covariances))
    x_max, y_max = centers.max(axis=0) + 2 * np.sqrt(np.max(covariances))

    x = np.linspace(x_min, x_max, grid_size)
    y = np.linspace(y_min, y_max, grid_size)
    X, Y = np.meshgrid(x, y)
    pos = np.dstack((X, Y))

    Z = np.zeros((grid_size, grid_size))

    for i in range(n_clusters):
        mean = centers[i][:2]
        cov = covariances[i][:2, :2]
        
        if np.linalg.det(cov) == 0:
            cov += np.eye(cov.shape[0]) * 1e-6

        diff = pos - mean
        inv_cov = np.linalg.inv(cov)
        mahalanobis_dist_sq = np.einsum('...i,ij,...j->...', diff, inv_cov, diff)

        density_factor = samples[i] / n_max
        Z += np.exp(-0.5 * density_factor * mahalanobis_dist_sq)

    Z /= Z.max()

    plt.figure(figsize=(8, 6))
    plt.contourf(X, Y, Z, levels=50, cmap='viridis')
    plt.colorbar(label='Density')
    plt.scatter(centers[:, 0], centers[:, 1], c='red', marker='x', s=100, label='Cluster Centers')
    plt.title('Cluster Density Heatmap with Sample Scaling')
    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    plt.legend()
    plt.show()

    

def _fixed_axes_box(ax, left=0.16, bottom=0.16, right=0.98, top=0.96):
    ax.set_position([left, bottom, right - left, top - bottom])

def _new_fig_ax(figsize=(5, 4), left=0.16, bottom=0.16, right=0.98, top=0.96):
    fig, ax = plt.subplots(figsize=figsize)

    try:
        fig.set_layout_engine(None)
    except Exception:
        pass
    fig.set_constrained_layout(False)

    _fixed_axes_box(ax, left, bottom, right, top)
    return fig, ax

def _save_standard(fig, path_pdf, path_png=None, path_svg=None, dpi_png=300):
    os.makedirs(os.path.dirname(path_pdf), exist_ok=True)

    with plt.rc_context({"savefig.bbox": "standard", "savefig.pad_inches": 0.0}):
        fig.savefig(path_pdf, bbox_inches=None, pad_inches=0.0)
        if path_png is not None:
            fig.savefig(path_png, bbox_inches=None, pad_inches=0.0, dpi=dpi_png)
        if path_svg is not None:
            fig.savefig(path_svg, bbox_inches=None, pad_inches=0.0)

def plot_results_acc(
    X_test,
    y_test,
    centers,
    assignments,
    name,
    execution_time=None,
    silhouette=None,
    ari=None,
    nmi=None,
    acc=None,
    flag_save_img=True,
    out_dir=".Images/related_methods/results_acc",
    kind="acc",
    ext_pdf="pdf",
    save_png=True,
    save_svg=True,
    show=True,
    seed=None,
    max_points=5000,
    show_ticks=True,
    tick_labelsize=10,
    sci_limits=(-2, 2),
    pad_frac=0.10,
    min_span=0.10,
    point_size=5.0,
    center_size=60.0,
    legend_loc="upper left",
):
    X_test = np.asarray(X_test)
    y_test = np.asarray(y_test).reshape(-1)
    assignments = np.asarray(assignments).reshape(-1)
    centers = np.asarray(centers)

    if X_test.ndim != 2 or X_test.shape[1] < 2:
        raise ValueError("X_test must be (N, D) with D>=2.")
    if centers.ndim != 2 or centers.shape[1] < 2:
        raise ValueError("centers must be (K, D) with D>=2.")

    rng = np.random.default_rng(seed)

    N = int(X_test.shape[0])
    perm = rng.permutation(N)
    Xs = X_test[perm]
    ys = y_test[perm]
    ps = assignments[perm]

    finite = np.isfinite(Xs[:, 0]) & np.isfinite(Xs[:, 1])
    Xs, ys, ps = Xs[finite], ys[finite], ps[finite]
    if Xs.shape[0] == 0:
        return None

    if (max_points is not None) and (Xs.shape[0] > int(max_points)):
        idx = rng.choice(Xs.shape[0], size=int(max_points), replace=False)
        Xs, ys, ps = Xs[idx], ys[idx], ps[idx]

    fig, ax = _new_fig_ax(figsize=(5, 4))
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")

    correct_mask = (ps == ys)
    wrong_mask = ~correct_mask

    c_ok = "tab:green"
    c_bad = "tab:red"
    c_ctr = "k"

    ax.scatter(Xs[correct_mask, 0], Xs[correct_mask, 1], s=point_size, color=c_ok, zorder=2)
    ax.scatter(Xs[wrong_mask,   0], Xs[wrong_mask,   1], s=point_size, color=c_bad, zorder=2)
    ax.scatter(centers[:, 0], centers[:, 1], s=center_size, color=c_ctr, marker="o", zorder=5)

    if show_ticks:
        ax.ticklabel_format(style="sci", axis="both", scilimits=sci_limits, useMathText=True)
        ax.tick_params(axis="both", labelsize=tick_labelsize)
        ax.xaxis.get_offset_text().set_size(tick_labelsize)
        ax.yaxis.get_offset_text().set_size(tick_labelsize)
        ax.set_xlabel(r"$z_1$")
        ax.set_ylabel(r"$z_2$")

    else:
        ax.set_xticks([])
        ax.set_yticks([])

    x_all = np.concatenate([Xs[:, 0], centers[:, 0]])
    y_all = np.concatenate([Xs[:, 1], centers[:, 1]])

    x0, x1 = float(np.nanmin(x_all)), float(np.nanmax(x_all))
    y0, y1 = float(np.nanmin(y_all)), float(np.nanmax(y_all))

    xr = max(x1 - x0, 0.0)
    yr = max(y1 - y0, 0.0)

    px = max(pad_frac * xr, min_span / 2.0)
    py = max(pad_frac * yr, min_span / 2.0)

    if xr < 1e-12:
        xc = 0.5 * (x0 + x1)
        x0, x1 = xc - min_span / 2.0, xc + min_span / 2.0
    else:
        x0, x1 = x0 - px, x1 + px

    if yr < 1e-12:
        yc = 0.5 * (y0 + y1)
        y0, y1 = yc - min_span / 2.0, yc + min_span / 2.0
    else:
        y0, y1 = y0 - py, y1 + py

    ax.set_xlim(x0, x1)
    ax.set_ylim(y0, y1)

    handles = [
        Line2D([0], [0], marker="o", linestyle="none", markerfacecolor=c_ok,  markeredgecolor="none",
               markersize=5.5, label="correct"),
        Line2D([0], [0], marker="o", linestyle="none", markerfacecolor=c_bad, markeredgecolor="none",
               markersize=5.5, label="wrong"),
        Line2D([0], [0], marker="o", linestyle="none", markerfacecolor=c_ctr, markeredgecolor="none",
               markersize=6.5, label="centers"),
    ]
    ax.legend(handles=handles, frameon=False, loc=legend_loc, handlelength=1.0)

    if acc is not None and ari is not None and nmi is not None:
        metrics_text = f"\u2191ACC:{acc:.2f}|ARI:{ari:.2f}|NMI:{nmi:.2f}"
        ax.text(0.5, 0.001, metrics_text, va="bottom", ha="center",
                transform=ax.transAxes, color="black", fontsize=12, fontweight="bold")

    if execution_time is not None:
        time_text = f"\u2193 Time:{execution_time:.2f}s"
        ax.text(0.999, 0.920, time_text, va="bottom", ha="right",
                transform=ax.transAxes, color="black", fontsize=12, fontweight="bold")

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    paths = {}
    if flag_save_img:
        base = os.path.join(out_dir, f"{name}_{kind}")
        pdf_path = f"{base}.{ext_pdf}"
        png_path = f"{base}.png" if save_png else None
        svg_path = f"{base}.svg" if save_svg else None

        _save_standard(fig, pdf_path, path_png=png_path, path_svg=svg_path, dpi_png=300)
        paths = {"pdf": pdf_path, "png": png_path, "svg": svg_path}

    if show:
        plt.show()
    else:
        plt.close(fig)

    return paths

def plot_mnist_batch_sweep(summary_dir=".benchmarks/sweeps", ks=(3, 10, 30),
                           out_dir=".benchmarks/figures", method="group"):
    import os
    import numpy as np, pandas as pd
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    os.makedirs(out_dir, exist_ok=True)

    def _load(K):
        df = pd.read_csv(f"{summary_dir}/batchsweep_MNIST_C{K}_noniid_SUMMARY.csv", header=[0, 1])
        df = df[df.iloc[:, 1].astype(str).str.contains(method)]
        bs = df.iloc[:, 0].astype(float).to_numpy(); o = np.argsort(bs)
        col = lambda i: df.iloc[:, i].astype(float).to_numpy()[o]
        return dict(bs=bs[o], ACC=(col(2), col(3)), ARI=(col(4), col(5)),
                    NMI=(col(6), col(7)), Time=(col(8), col(9)))

    DATA = {K: _load(K) for K in ks}
    palette = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd"]
    COL = {K: palette[i % len(palette)] for i, K in enumerate(ks)}
    panels = [("ACC", "ACC"), ("ARI", "ARI"), ("NMI", "NMI"), ("Time", "Runtime [s]")]

    def _panel(ax, key, ylab, legend=False, label_fs=11, tick_fs=10, leg_fs=10):
        for K in ks:
            m, s = DATA[K][key]
            ax.plot(DATA[K]["bs"], m, "-o", ms=4.0, lw=1.9, color=COL[K], label=f"K = {K}")
            ax.fill_between(DATA[K]["bs"], m - s, m + s, color=COL[K], alpha=0.15, lw=0)
        ax.set_xscale("log", base=2)
        ax.set_xticks([32, 64, 128, 256, 512, 1024]); ax.set_xticklabels([32, 64, 128, 256, 512, 1024])
        ax.set_xlabel(r"Mini-batch size $B$", fontsize=label_fs); ax.set_ylabel(ylab, fontsize=label_fs)
        ax.tick_params(labelsize=tick_fs); ax.grid(True, alpha=0.25, lw=0.4)
        for sp in ("top", "right"):
            ax.spines[sp].set_visible(False)
        if legend:
            ax.legend(frameon=False, fontsize=leg_fs, ncol=1, loc="upper right", handletextpad=0.4)

    for key, ylab in panels:
        fig, ax = plt.subplots(figsize=(5.6, 3.3), constrained_layout=True)
        _panel(ax, key, ylab, legend=(key == "Time"), label_fs=16, tick_fs=14, leg_fs=14)
        ax.set_box_aspect(0.50)
        fig.savefig(f"{out_dir}/MNIST_batchsweep_{key}.pdf", bbox_inches="tight", pad_inches=0.02)
        plt.close(fig)

    fig, axes = plt.subplots(2, 2, figsize=(11.0, 6.0), constrained_layout=True)
    for ax, (key, ylab) in zip(axes.ravel(), panels):
        _panel(ax, key, ylab); ax.set_box_aspect(0.55)
    axes.ravel()[0].legend(frameon=False, loc="lower center", ncol=len(ks), columnspacing=1.0,
                           handletextpad=0.4, bbox_to_anchor=(0.5, 1.02))
    fig.savefig(f"{out_dir}/MNIST_batchsweep_grid.pdf"); plt.close(fig)
    return out_dir

def plot_mnist_alpha_partition(X, y, alphas=(0.1, 0.3, 1.0, 3.0), num_clients=10,
                               num_classes=10, out_dir=".benchmarks/figures", seed=0):
    import os
    import numpy as np, torch
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    os.makedirs(out_dir, exist_ok=True)

    def _split(alpha):
        np.random.seed(seed); torch.manual_seed(seed)
        classes = torch.unique(y).tolist()
        cd = np.random.dirichlet([alpha] * num_clients, size=len(classes))
        parts = [[] for _ in range(num_clients)]
        for ci, c in enumerate(classes):
            idx = torch.where(y == c)[0]; idx = idx[torch.randperm(idx.numel())]
            counts = np.random.multinomial(int(idx.numel()), cd[ci]); s = 0
            for k in range(num_clients):
                cnt = int(counts[k])
                if cnt > 0:
                    parts[k].append(idx[s:s + cnt])
                s += cnt
        return [y[torch.cat(p)] if p else torch.empty(0, dtype=torch.long) for p in parts]

    splits = [_split(a) for a in alphas]
    cmax = max(max(np.bincount(np.asarray(yk).astype(int), minlength=num_classes).max()
                   for yk in [t.cpu().numpy() for t in s]) for s in splits)
    fig, axes = plt.subplots(1, len(alphas), figsize=(3.4 * len(alphas) + 0.1, 3.5),
                             constrained_layout=True)
    for i, (a, s, ax) in enumerate(zip(alphas, splits, np.atleast_1d(axes))):
        plot_noniid_partition(s, num_clients, num_classes, ax=ax, count_max=cmax,
                              show_legend=(i == len(alphas) - 1), label_size=11, tick_size=9)
        ax.set_title(rf"$\alpha = {a:g}$", fontsize=13)
        if i > 0:
            ax.set_ylabel(""); ax.set_yticklabels([])
    fig.savefig(f"{out_dir}/MNIST_noniid_alpha_grid.pdf"); plt.close(fig)
    return out_dir
