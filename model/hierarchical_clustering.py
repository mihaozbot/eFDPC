import torch  # type: ignore
import numpy as np  # type: ignore
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
from matplotlib.cm import get_cmap
import heapq
from matplotlib.collections import LineCollection
import torch.nn as nn
import numpy as np
from matplotlib.cm import get_cmap
from matplotlib.lines import Line2D
from matplotlib.collections import LineCollection
import os
import matplotlib.ticker as mticker

def _hc_new_fig_ax(
    figsize=(5, 4),
    left=0.16,
    right=0.98,
    bottom=0.16,
    top=0.96,
):
    fig, ax = plt.subplots(figsize=figsize)
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")

    fig.subplots_adjust(left=left, right=right, bottom=bottom, top=top)
    return fig, ax

def _hc_apply_sci_axes(ax, sci=True, logx=False, logy=False, scilimits=(-2, 2)):
    if not sci:
        return

    if (not logx) and (not logy):
        ax.ticklabel_format(style="sci", axis="both", scilimits=scilimits, useMathText=True)
        ax.xaxis.get_offset_text().set_size(10)
        ax.yaxis.get_offset_text().set_size(10)
        return

    if not logx:
        ax.ticklabel_format(style="sci", axis="x", scilimits=scilimits, useMathText=True)
        ax.xaxis.get_offset_text().set_size(10)
    if not logy:
        ax.ticklabel_format(style="sci", axis="y", scilimits=scilimits, useMathText=True)
        ax.yaxis.get_offset_text().set_size(10)

    if logx:
        ax.xaxis.set_major_formatter(mticker.LogFormatterSciNotation(base=10))
        ax.xaxis.set_minor_formatter(mticker.NullFormatter())
    if logy:
        ax.yaxis.set_major_formatter(mticker.LogFormatterSciNotation(base=10))
        ax.yaxis.set_minor_formatter(mticker.NullFormatter())

def _hc_set_ticks(ax, show_ticks=True):
    if show_ticks:
        return
    ax.set_xticks([])
    ax.set_yticks([])

def _hc_savefig(fig, out_dir, name, kind, ext="pdf"):
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, f"{name}_{kind}.{ext}")

    fig.savefig(path, format=ext, facecolor=fig.get_facecolor(), edgecolor="none")
    return path

class HierarchicalClustering:
    def __init__(self, parent):
        self.parent = parent
        self.kappa = None

    def _ctx(self):
        C   = int(self.parent.c)
        dev = self.parent.device
        mm  = self.parent.merging_mech
        return C, dev, mm

        
    

            

    @torch.no_grad()
    def _compute_delta_to_higher_on(self, Z: torch.Tensor, tie_tol: float = 0.0):
        C, dev, _ = self._ctx()
        if C <= 0:
            self.nn_higher = torch.empty(0, dtype=torch.long, device=dev)
            self.delta = torch.empty(0, dtype=torch.float32, device=dev)
            self.imp_delta = torch.empty(0, dtype=torch.float32, device=dev)
            return
        Z = Z[:C].to(dev, torch.float64)
        rho = self.importance[:C].to(dev, torch.float64)
        D = torch.cdist(Z, Z) ** 2
        D.fill_diagonal_(float("inf"))
        ri = rho.view(C, 1)
        rj = rho.view(1, C)
        i = torch.arange(C, device=dev).view(C, 1)
        j = torch.arange(C, device=dev).view(1, C)
        if tie_tol > 0.0:
            eq = (rj - ri).abs() <= tie_tol
            Hi = (rj > (ri + tie_tol)) | (eq & (j < i))
        else:
            Hi = (rj > ri) | ((rj == ri) & (j < i))
        D.masked_fill_(~Hi, float("inf"))
        delta, nn = D.min(dim=1)
        no_higher = ~torch.isfinite(delta)
        vals = delta[~no_higher]
        fallback = vals.max() if vals.numel() > 0 else torch.tensor(torch.finfo(delta.dtype).tiny, device=dev, dtype=delta.dtype)
        delta = torch.where(no_higher, fallback, delta)
        delta = delta.clamp_min(torch.finfo(delta.dtype).tiny)
        nn = torch.where(no_higher, torch.full_like(nn, -1), nn)
        self.nn_higher = nn.to(torch.long)
        self.delta = delta.to(torch.float32)
        self.rho = rho.to(torch.float32)
        self.imp_delta = (rho * delta).to(torch.float32)

    def compute_local_density_knn_avg_on(self, Z):
        C = int(self.parent.c)
        mm = self.parent.merging_mech
        dev = self.parent.device

        if C == 0:
            self.outlier = torch.empty(0, device=dev)
            self.density = torch.empty(0, device=dev)
            self.importance = torch.empty(0, device=dev)
            return

        n = self.parent.n[:C].to(dev, torch.float32)

        ki = mm.edge_i
        kj = mm.edge_j
        kk = mm.edge_mah_dist.to(dev, torch.float32)

        msk = (ki >= 0) & (ki < C) & (kj >= 0) & (kj < C)
        ki, kj, kk = ki[msk], kj[msk], kk[msk]

        src = torch.cat([ki, kj], 0)
        dst = torch.cat([kj, ki], 0)
        ksym = torch.cat([kk, kk], 0)

        base = ksym
        w = torch.exp(-base)

        take = (n[dst] > n[src]).to(w.dtype)
        recv = n[dst] * w * take

        mass = n + torch.zeros(C, device=dev, dtype=w.dtype).scatter_add_(0, src, recv)

        self.outlier = 1.0 / mass
        self.density = mass
        self.importance = mass

        
   
    def assign_with_capacity(
        self,
        centers: torch.Tensor,
        lambda_soft: float = 1.0,
        weights: torch.Tensor | None = None
    ):
        import heapq
        import torch

        C = int(self.parent.c)
        centers = centers.long().cpu()
        G = int(centers.numel())
        if G == 0 or C == 0:
            return torch.full((C,), -1, dtype=torch.long)

        if weights is None:
            weights = torch.ones(C, dtype=torch.float32)
        else:
            weights = weights.detach().cpu().float()

        adj = self._adj_list

        total_mass = float(weights[:C].sum().item())
        T = total_mass / float(G)

        def pen(mass_g: float) -> float:
            load = mass_g / T
            return 0.0 if load <= 1.0 else (load - 1.0)

        INF = float("inf")

        best = [INF] * C
        dist = [INF] * C

        own = [-1] * C
        visited = [False] * C
        group_mass = [0.0] * G

        centers_list = centers.tolist()
        centers_set = set(centers_list)

        pq = []
        for g, c in enumerate(centers_list):
            dist[c] = 0.0
            own[c] = g
            group_mass[g] += float(weights[c].item())
            key = 0.0 + lambda_soft * pen(group_mass[g])
            best[c] = min(best[c], key)
            heapq.heappush(pq, (key, 0.0, c, g))

        while pq:
            key, du, u, g = heapq.heappop(pq)
            if visited[u]:
                continue

            visited[u] = True
            own[u] = g
            dist[u] = du

            if u not in centers_set:
                group_mass[g] += float(weights[u].item())

            pg = lambda_soft * pen(group_mass[g])

            for v, w_uv in adj[u]:
                if visited[v]:
                    continue
                nd = du + float(w_uv)
                nkey = nd + pg
                if nkey < best[v]:
                    best[v] = nkey
                    heapq.heappush(pq, (nkey, nd, v, g))

        return torch.tensor(own, dtype=torch.long)
    
    
    @torch.no_grad()
    def _attach_unassigned_to_nearest_assigned(
        self,
        groups: torch.Tensor,
        *,
        use_peaks_only: bool = True,
        chunk: int = 2048,
    ):
        import torch

        C = int(self.parent.c)
        groups = groups.long().cpu()
        Z = self.parent.mu[:C].detach().cpu().to(torch.float32)

        un = (groups < 0).nonzero(as_tuple=False).view(-1)
        if un.numel() == 0:
            return groups

        if use_peaks_only and hasattr(self, "group_center_ids"):
            anchors = self.group_center_ids.detach().cpu().long()
            anchors = anchors[(anchors >= 0) & (anchors < C)]
            if anchors.numel() == 0:
                use_peaks_only = False
        if not use_peaks_only:
            anchors = (groups >= 0).nonzero(as_tuple=False).view(-1)

        if anchors.numel() == 0:
            return groups

        Za = Z[anchors]
        Za_norm = (Za * Za).sum(dim=1).view(1, -1)

        for s in range(0, int(un.numel()), int(chunk)):
            uu = un[s : s + chunk]
            Zu = Z[uu]
            Zu_norm = (Zu * Zu).sum(dim=1).view(-1, 1)
            d2 = Zu_norm + Za_norm - 2.0 * (Zu @ Za.t())
            j = torch.argmin(d2, dim=1)

            groups[uu] = groups[anchors[j]]

        return groups
    
    def group_by_geodesic(self):
        import torch

        C, dev, mm = int(self.parent.c), self.parent.device, self.parent.merging_mech

        if C == 0:
            self.groupings = [{
                "threshold": 0.0,
                "centers": torch.empty(0, dtype=torch.long),
                "groups":  torch.empty(0, dtype=torch.long),
            }]
            return self.groupings

        ei  = mm.edge_i
        ej  = mm.edge_j
        d_e = self.parent.merging_mech.edge_mah_dist

        src = torch.cat([ei, ej]).tolist()
        dst = torch.cat([ej, ei]).tolist()
        w   = torch.cat([d_e, d_e]).tolist()

        adj = [[] for _ in range(C)]
        for s, d, ww in zip(src, dst, w):
            if 0 <= s < C and 0 <= d < C:
                adj[s].append((d, float(ww)))
        self._adj_list = adj

        ki = mm.knn_edge_i.long().cpu()
        kj = mm.knn_edge_j.long().cpu()

        E_knn = int(ki.numel())
        knn_j_table = None
        knn_sets = None
        knn_k_table = None

        if E_knn > 0 and C > 0:
            K = E_knn // C
            if K * C != E_knn:
                knn_out = [[] for _ in range(C)]
                for a, b in zip(ki.tolist(), kj.tolist()):
                    if 0 <= a < C and 0 <= b < C:
                        knn_out[a].append(b)
                knn_sets = [set(nbs) for nbs in knn_out]
                knn_j_table = knn_out
            else:
                order = torch.argsort(ki)
                kj_s = kj[order]
                knn_j_table = kj_s.view(C, K).numpy()
                knn_sets = [set(row.tolist()) for row in knn_j_table]

                k_attr = None
                for name in ("knn_k", "knn_edge_k", "knn_dist", "knn_w", "knn_cost"):
                    if hasattr(mm, name):
                        k_attr = getattr(mm, name)
                        break
                if k_attr is not None:
                    kk = k_attr.detach().cpu().float()
                    if int(kk.numel()) == E_knn:
                        kk_s = kk[order]
                        knn_k_table = kk_s.view(C, K).numpy()

        self._knn_j_table = knn_j_table
        self._knn_sets    = knn_sets
        self._knn_k_table = knn_k_table

        self.groupings = []

        for idx, thr_gamma in enumerate(self.imp_delta_threshold):
            c_idx   = min(idx, len(self.centers) - 1)
            centers = self.centers[c_idx].long().cpu()
            G = int(centers.numel())

            if G == 0:
                groups = torch.full((C,), -1, dtype=torch.long)
                self.groupings.append({
                    "threshold": float(thr_gamma.item()),
                    "centers":  centers,
                    "groups":   groups,
                })
                continue

            groups = self.assign_with_capacity(
                centers,
                lambda_soft=0.0,
                weights=self.importance[:C] if hasattr(self, "importance") else None
            )
            groups = self._attach_unassigned_to_nearest_assigned(groups, use_peaks_only=False)

            self.groupings.append({
                "threshold": float(thr_gamma.item()),
                "centers":   centers,
                "groups":    groups,
            })

        final         = self.groupings[-1]
        groups_final  = final["groups"].clone()
        centers_final = final["centers"].clone()

        self.cluster_to_group = groups_final
        self.group_center_ids = centers_final
        self.n_groups         = int(centers_final.numel())
        self.group_centers    = self.parent.mu[centers_final].detach().clone()
        self.group_members    = [
            (self.cluster_to_group == g).nonzero(as_tuple=False).view(-1)
            for g in range(self.n_groups)
        ]

        return self.groupings

        

    def compute_centers(self, K=None):
        C, dev, _ = self._ctx()
        gamma_higher = self.imp_delta[:C]

        thr0 = float(self.imp_delta_threshold[0])
        seeds = torch.where(gamma_higher >= thr0)[0]
        if seeds.numel() == 0:
            seeds = torch.tensor([int(torch.argmax(gamma_higher).item())], device=dev)

        K0 = int(seeds.numel())

        if K is not None:
            K = max(1, min(int(K), C))
        else:
            K = K0

        if K != K0:
            _, idx = torch.topk(gamma_higher, k=min(K, C))
            seeds = idx

        g = gamma_higher[seeds]

        try:
            order = torch.argsort(g, descending=True, stable=True)
        except TypeError:
            order = torch.argsort(g - 1e-12 * seeds.to(g.dtype), descending=True)

        centers = seeds[order]

        ranks = torch.full((C,), -1, device=dev, dtype=torch.long)
        ranks[centers] = torch.arange(1, centers.numel() + 1, device=dev, dtype=torch.long)

        self.center_gamma = gamma_higher
        self.center_rank = ranks
        self.centers = [centers.detach().cpu()]
        self.initial_centers = [centers.detach().cpu().clone()]

    def compute_imp_delta_threshold(self, num_knees=1, initial_guess=None, enforce_min_keep=True):
        imp_delta = self.imp_delta.detach().cpu().numpy() if hasattr(self.imp_delta, "detach") else np.asarray(self.imp_delta)
        g = np.sort(imp_delta)[::-1]
        n = len(g)

        self.knee_idxs = np.array([n - 1]) if n > 0 else np.array([0])
        self.imp_delta_threshold = torch.tensor([g[-1] ], dtype=torch.float32) if n > 0 else torch.tensor([], dtype=torch.float32)
        self.imp_delta_knee = int(self.knee_idxs[0]) if n > 0 else 0
        self.imp_delta_sorted = torch.tensor(g.copy(), dtype=torch.float32) if n > 0 else torch.tensor([], dtype=torch.float32)
        if n > 0:
            self.imp_delta_line_start = torch.tensor([0, g[0]], dtype=torch.float32)
            self.imp_delta_line_end   = torch.tensor([n - 1, g[-1]], dtype=torch.float32)
        else:
            self.imp_delta_line_start = torch.tensor([0, 0.0], dtype=torch.float32)
            self.imp_delta_line_end   = torch.tensor([0, 0.0], dtype=torch.float32)

        if n == 0:
            return self.imp_delta_threshold, self.imp_delta_knee

        min_keep = None
        if initial_guess is not None:
            min_keep = int(initial_guess)
            min_keep = max(1, min(min_keep, n))

        if min_keep is not None and min_keep >= n:
            self.knee_idxs = np.array([n - 1])
            self.imp_delta_threshold = torch.tensor([g[-1]], dtype=torch.float32)
            self.imp_delta_knee = n - 1
            return self.imp_delta_threshold, self.imp_delta_knee

        knee_idxs = []

        if min_keep is not None:
            split = min_keep - 1
            if 1 <= split < n - 1:
                knee_idxs.append(split)
                segments = [(0, split), (split, n - 1)]
            else:
                segments = [(0, n - 1)]
        else:
            segments = [(0, n - 1)]

        self.segment_lines = []

        while len(knee_idxs) < num_knees and segments:
            start, end = segments.pop(0)
            if end - start < 3:
                continue

            idxs = np.arange(start, end + 1)
            P0 = np.array([start, g[start]])
            P1 = np.array([end,   g[end]])
            vec = P1 - P0
            denom = np.linalg.norm(vec)
            if denom < 1e-12:
                continue

            pts = np.vstack([idxs, g[start:end + 1]]).T
            rel = pts - P0
            dists = np.abs(np.cross(rel, vec)) / denom
            knee = int(idxs[np.argmax(dists)])

            if knee not in knee_idxs:
                knee_idxs.append(knee)
                segments.append((start, knee))
                segments.append((knee, end))
                self.segment_lines.append((P0.copy(), P1.copy()))

        knee_idxs = sorted(knee_idxs)[:num_knees]

        thresholds = torch.tensor(g[knee_idxs], dtype=torch.float32) if knee_idxs else torch.tensor([], dtype=torch.float32)

        if enforce_min_keep and (min_keep is not None):
            kth_val = float(g[min_keep - 1])
            if thresholds.numel() == 0:
                thresholds = torch.tensor([kth_val], dtype=torch.float32)
                knee_idxs = [min_keep - 1]
            else:
                thresholds = torch.minimum(thresholds, torch.tensor([kth_val], dtype=torch.float32))
                knee_idxs[0] = max(knee_idxs[0], min_keep - 1)

        self.knee_idxs = np.array(knee_idxs) if knee_idxs else np.array([0])
        self.imp_delta_threshold = thresholds
        self.imp_delta_knee = int(self.knee_idxs[0]) if self.knee_idxs.size else 0
        self.imp_delta_sorted = torch.tensor(g.copy(), dtype=torch.float32)
        self.imp_delta_line_start = torch.tensor([0, g[0]], dtype=torch.float32)
        self.imp_delta_line_end = torch.tensor([n - 1, g[-1]], dtype=torch.float32)

        return self.imp_delta_threshold, self.imp_delta_knee

    @torch.no_grad()
    def remove_all_except_centers(self):
        C = int(self.parent.c)

        centers = self.centers[0]

        centers = centers

        all_idx = torch.arange(C)
        mask = torch.ones(C)
        mask[centers] = 0
        removed = all_idx[mask > 0]

        mu = self.parent.mu[:C]
        n  = self.parent.n[:C]

        mu_c = mu[centers]
        mu_r = mu[removed]

        rc2 = (mu_r * mu_r).sum(1, keepdim=True)
        cc2 = (mu_c * mu_c).sum(1).unsqueeze(0)
        dist2 = rc2 + cc2 - 2.0 * (mu_r @ mu_c.t())
        pos = torch.argmin(dist2, dim=1)

        n_add = n.new_zeros(centers.numel())
        n_add.scatter_add_(0, pos, n[removed])

        n_old = n[centers].clone()
        n_new = n_old + n_add
        n[centers] = n_new.clamp_min(1e-12)

        if 1:
            D = int(mu.shape[1])
            mu_sum = mu[centers] * n_old.unsqueeze(1)
            add_sum = mu.new_zeros(centers.numel(), D)
            add_sum.scatter_add_(0, pos.unsqueeze(1).expand(-1, D), mu[removed] * n[removed].unsqueeze(1))
            mu_new = (mu_sum + add_sum) / n_new.clamp_min(1e-12).unsqueeze(1)
            mu[centers] = torch.where((n_new > 1e-12).unsqueeze(1), mu_new, mu[centers])

        self.parent.removal_mech.remove_clusters(removed)

        
    def cluster_communities(self, num_knees=1, initial_guess=None):
        self.parent.merging_mech.batch_compute_volume()
        C = int(self.parent.c)
        self.compute_components_global(verbose=False)
        Z_emb = self.parent.mu[:C]
        self.compute_local_density_knn_avg_on(Z_emb)
        self._compute_delta_to_higher_on(Z_emb)
        self.compute_imp_delta_threshold(num_knees, initial_guess)
        self.compute_centers(K=initial_guess)

    def compute_groups(self, initial_guess=None):
        self.group_by_geodesic()
       

    def plot_imp_delta_elbow(
        self,
        name="fig",
        out_dir=".Images",
        ext="pdf",
        show=True,
        show_ticks=True,
        sci=True,
        figsize=(5, 4),
    ):
        g = self.imp_delta_sorted.detach().cpu().numpy().astype(np.float64)
        n = int(g.size)
        if n == 0:
            return None

        r = np.arange(1, n + 1, dtype=np.float64)

        P0s = self.imp_delta_line_start.detach().cpu().numpy().astype(np.float64).copy()
        P1s = self.imp_delta_line_end.detach().cpu().numpy().astype(np.float64).copy()

        P0 = np.array([1.0, float(g[0])], dtype=np.float64)
        P1 = np.array([float(n), float(g[-1])], dtype=np.float64)

        if P0s.size >= 2 and P1s.size >= 2:
            P0 = np.array([float(P0s[0]) + 1.0, float(P0s[1])], dtype=np.float64)
            P1 = np.array([float(P1s[0]) + 1.0, float(P1s[1])], dtype=np.float64)

        fig, ax = _hc_new_fig_ax(figsize=figsize)

        c_curve = "#111111"
        c_base = "#4a4a4a"
        c_thr = "#7a7a7a"
        c_sel = "#2b3a67"
        c_k = "#5a5a5a"

        ax.plot(r, g, color=c_curve, linewidth=1.4)
        ax.plot([P0[0], P1[0]], [P0[1], P1[1]], color=c_base, linewidth=1.2)

        knee_idxs = np.asarray(self.knee_idxs).astype(int).ravel()
        thr = np.asarray(self.imp_delta_threshold.detach().cpu().numpy()).astype(np.float64).ravel()

        v = (P1 - P0)
        vv = float(np.dot(v, v))

        u = v / (np.sqrt(vv) if vv > 0.0 else 1.0)
        nvec = np.array([-u[1], u[0]], dtype=np.float64)
        L = 0.03 * (P1[0] - P0[0])

        for i, k0 in enumerate(knee_idxs):
            if not (0 <= k0 < n):
                continue

            xk = float(k0 + 1)
            yk = float(g[k0])
            pt = np.array([xk, yk], dtype=np.float64)

            if vv > 0.0:
                t = float(np.dot(pt - P0, v) / vv)
                proj = P0 + t * v

                ax.plot([pt[0], proj[0]], [pt[1], proj[1]], color=c_sel, linewidth=1.15)

                p1 = proj
                p2 = proj + L * u
                p3 = p2 + L * nvec
                ax.plot([p1[0], p2[0]], [p1[1], p2[1]], color=c_sel, linewidth=1.0)
                ax.plot([p2[0], p3[0]], [p2[1], p3[1]], color=c_sel, linewidth=1.0)

                ax.scatter([proj[0]], [proj[1]], s=28.0, c=c_sel, zorder=6)

            ax.scatter([pt[0]], [pt[1]], s=60.0, c=c_curve, zorder=7)

            if i < thr.size:
                th = float(thr[i])
                K = int(np.sum(g >= th))
                ax.axhline(th, color=c_thr, linestyle="--", linewidth=1.05, alpha=0.85)
                ax.axvline(float(K), color=c_k, linestyle=":", linewidth=1.05, alpha=0.90)

        ax.set_xlabel(r"Rank $r$")
        ax.set_ylabel(r"Score $\gamma$")
        ax.set_xlim(1.0, float(n))

        _hc_set_ticks(ax, show_ticks=show_ticks)
        _hc_apply_sci_axes(ax, sci=sci, logx=False, logy=False)

        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

        path = _hc_savefig(fig, out_dir, name, "elbow", ext)
        if show:
            plt.show()
        else:
            plt.close(fig)
        return path

    def plot_decision_graph(
        self,
        grouping_idx=0,
        name="fig",
        out_dir=".Images",
        ext="pdf",
        show=True,
        imp_min=0.0,
        point_size=120.0,
        center_size=120.0,
        cmap="tab20",
        alpha_pts=1.0,
        alpha_unassigned=1,
        log_delta=False,
        log_mode="axis",
        show_ticks=True,
        sci=True,
        figsize=(5, 4),
        pad_frac_x=0.05,
        pad_frac_y=0.10,
    ):
        C = int(self.parent.c)
        imp = self.rho[:C].detach().cpu().numpy().astype(np.float64)
        delta = self.delta[:C].detach().cpu().numpy().astype(np.float64)

        alpha_pts = 1.0
        alpha_unassigned = 1.0

        if log_delta:
            if log_mode == "log1p":
                delta_plot = np.log1p(np.maximum(delta, 0.0))
                y_log = False
            elif log_mode == "axis":
                delta_plot = delta
                y_log = True
            else:
                raise ValueError("log_mode must be 'log1p' or 'axis'")
        else:
            delta_plot = delta
            y_log = False

        groups = torch.as_tensor(self.groupings[grouping_idx]["groups"]).cpu().numpy()
        centers = torch.as_tensor(self.groupings[grouping_idx]["centers"]).cpu().numpy()

        mask = (imp > imp_min) & np.isfinite(imp) & np.isfinite(delta_plot)

        uniq = np.unique(groups[groups >= 0])
        cm = plt.cm.get_cmap(cmap, max(len(uniq), 1))

        colors = np.tile(
            np.array([0.7, 0.7, 0.7, alpha_unassigned], dtype=np.float64),
            (C, 1),
        )
        for k, g in enumerate(uniq):
            col = np.array(cm(k), dtype=np.float64)
            col[3] = 1.0
            colors[groups == g] = col

        fig, ax = _hc_new_fig_ax(figsize=figsize)

        ax.scatter(
            imp[mask],
            delta_plot[mask],
            s=point_size,
            c=colors[mask],
            linewidths=0,
            zorder=3,
        )

        cen_mask = np.zeros(C, dtype=bool)
        if centers.size:
            cen_mask[centers] = True
        cen_mask = cen_mask & mask
        if cen_mask.any():
            cen_cols = colors[cen_mask].copy()
            cen_cols[:, 3] = 1.0
            ax.scatter(
                imp[cen_mask],
                delta_plot[cen_mask],
                s=center_size,
                marker="o",
                c=cen_cols,
                edgecolors="k",
                linewidths=1.2,
                zorder=5,
            )

        decision_drawn = False
        if hasattr(self, "imp_delta_threshold") and len(self.imp_delta_threshold) > 0:
            thr = float(self.imp_delta_threshold[min(grouping_idx, len(self.imp_delta_threshold) - 1)].item())
            rr = imp[mask]
            rr = rr[np.isfinite(rr)]
            if rr.size:
                r0, r1 = float(np.min(rr)), float(np.max(rr))
                r = np.linspace(r0, r1, 256)
                y = thr / r
                if log_delta and log_mode == "log1p":
                    y = np.log1p(np.maximum(y, 0.0))
                ax.plot(r, y, color="k", linewidth=1.2, zorder=4)
                decision_drawn = True

        if y_log:
            ax.set_yscale("log")

        ax.set_xlabel(r"Density $\rho$")
        ax.set_ylabel(r"Distance $\delta$")

        x_vals = imp[mask]
        if x_vals.size:
            x_max = float(np.max(x_vals))
            ax.set_xlim(0.0, x_max * (1.0 + float(pad_frac_x)))

        y_vals = delta_plot[mask]
        if y_vals.size:
            y_max = float(np.max(y_vals))
            if y_log:
                ax.set_ylim(0, y_max * (1.0 + float(pad_frac_y)))
            else:
                ax.set_ylim(0.0, y_max * (1.0 + float(pad_frac_y)))

        _hc_set_ticks(ax, show_ticks=show_ticks)
        _hc_apply_sci_axes(ax, sci=sci, logx=False, logy=y_log)

        handles = [
            Line2D(
                [0], [0], marker="o", linestyle="none",
                markerfacecolor="k", markeredgecolor="none",
                markersize=4.5, label=r"Micro $\mathcal{C}$",
            ),
            Line2D(
                [0], [0], marker="o", linestyle="none",
                markerfacecolor="none", markeredgecolor="k",
                markersize=8.5, label=r"Peaks $\mathcal{P}$",
            ),
        ]
        if decision_drawn:
            handles.append(Line2D([0], [0], color="k", lw=1.2, label=r"Decision $\gamma$"))
        ax.legend(handles=handles, frameon=False, loc="best", handlelength=1.2)

        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

        path = _hc_savefig(fig, out_dir, name, "decision", ext)
        if show:
            plt.show()
        else:
            plt.close(fig)
        return path

    def plot_all_cluster_groupings(
        self,
        name="fig",
        out_dir=".Images",
        ext="pdf",
        show=True,
        colormap="tab10",
        peak_size=120.0,
        show_ticks=True,
        sci=True,
        figsize=(5, 4),
        data=None,
        *,
        edge_text: bool = True,
        edge_text_scale: float = 1,
        edge_text_max: int = 2000,
        edge_text_fmt: str = "{:.2g}",
        edge_value_attr=("edge_kappa", "edge_k", "edge_cost", "edge_mah_dist"),
    ):
        C = int(self.parent.c)
        if C <= 0 or not hasattr(self, "groupings") or len(self.groupings) == 0:
            return None

        idx = len(self.groupings) - 1
        result = self.groupings[idx]
        groups = torch.as_tensor(result["groups"]).cpu().numpy()
        centers = torch.as_tensor(result["centers"]).cpu().numpy()
        G = int(centers.size)
        if G <= 0:
            return None

        mu = self.parent.mu[:C].detach().cpu().numpy()
        D = int(mu.shape[1])

        if hasattr(self.parent, "S_glo") and D >= 2:
            Sg = self.parent.S_glo.detach().cpu()
            varg = torch.diag(Sg) if Sg.dim() == 2 else Sg.reshape(-1)
            _, didx = torch.topk(varg, k=2, largest=True)
            d0 = int(didx[0].item())
            d1 = int(didx[1].item())
        else:
            d0, d1 = 0, 1

        X = mu[:C, d0].astype(np.float64)
        Y = mu[:C, d1].astype(np.float64)

        n = self.parent.n[:C].detach().cpu().numpy().astype(np.float64)

        cm = get_cmap(colormap)(np.linspace(0, 1, max(G, 1)))

        fig, ax = _hc_new_fig_ax(figsize=figsize)

        if data is not None:
            data = np.asarray(data)
            _pd = ((data[:, [d0, d1]][:, None, :] - np.stack([X, Y], 1)[None]) ** 2).sum(-1).argmin(1)
            _pg = np.clip(groups.astype(int)[_pd], 0, max(G - 1, 0))
            ax.scatter(data[:, d0], data[:, d1], s=7, c=cm[_pg], alpha=0.35, zorder=0, linewidths=0)

        mm = getattr(self.parent, "merging_mech", None)
        have_kappa = False
        ei = np.empty(0, dtype=np.int64)
        ej = np.empty(0, dtype=np.int64)
        e_val = None

        if mm is not None and hasattr(mm, "edge_i") and getattr(mm.edge_i, "numel", lambda: 0)() > 0:
            ei0 = mm.edge_i.long().detach().cpu().numpy()
            ej0 = mm.edge_j.long().detach().cpu().numpy()

            e_val0 = None
            for nm in edge_value_attr:
                if hasattr(mm, nm):
                    t = getattr(mm, nm)
                    if getattr(t, "numel", lambda: 0)() > 0:
                        tmp = t.detach().cpu().numpy().astype(np.float64)
                        if tmp.shape[0] == ei0.shape[0]:
                            e_val0 = tmp
                            break

            msk = (ei0 >= 0) & (ei0 < C) & (ej0 >= 0) & (ej0 < C)
            ei, ej = ei0[msk], ej0[msk]
            if e_val0 is not None:
                e_val = e_val0[msk]

            E = int(ei.size)
            M = 50_000
            if E > M:
                step = max(1, E // M)
                take = np.arange(0, E, step, dtype=np.int64)[:M]
                ei, ej = ei[take], ej[take]
                if e_val is not None:
                    e_val = e_val[take]

            if ei.size:
                seg = np.empty((ei.size, 2, 2), dtype=np.float32)
                seg[:, 0, 0] = X[ei]
                seg[:, 0, 1] = Y[ei]
                seg[:, 1, 0] = X[ej]
                seg[:, 1, 1] = Y[ej]
                ax.add_collection(LineCollection(seg, linewidths=0.7, colors="k", alpha=0.6, zorder=1))
                have_kappa = True

        plot_sigma = 0.0
        use_unbiased = False
        plot_shrink = 0.5
        use_sqrt_n = False

        for c in range(C):
            g = int(groups[c])
            if g < 0 or g >= G:
                continue

            nc = float(self.parent.n[c].item())
            den = max(nc - 1.0, 1.0) if use_unbiased else max(nc, 1.0)

            S = self.parent.S[c].detach().cpu().numpy()

            if S.ndim == 1:
                cov2 = np.diag(S / den)[[d0, d1]][:, [d0, d1]]
            else:
                cov2 = (S / den)[[d0, d1]][:, [d0, d1]]

            if use_sqrt_n:
                cov2 = cov2 / max(np.sqrt(nc), 1.0)

            vals, vecs = np.linalg.eigh(cov2)

            angle = np.degrees(np.arctan2(*vecs[:, 0][::-1]))
            width, height = (2.0 * plot_sigma * plot_shrink) * np.sqrt(vals)

            col = cm[g].copy()
            col[3] = 0.7

            ax.add_patch(
                Ellipse(
                    xy=(X[c], Y[c]),
                    width=float(width),
                    height=float(height),
                    angle=float(angle),
                    edgecolor=tuple(col),
                    facecolor="none",
                    lw=1.2,
                    zorder=2,
                )
            )

        cols = cm[np.clip(groups.astype(int), 0, max(G - 1, 0))].copy()
        cols[:, 3] = 1.0

        s = 10.0 + 60.0 * np.sqrt(n / np.max(n))
        ax.scatter(X, Y, s=s, c=cols, linewidths=0, zorder=3)

        m = (centers >= 0) & (centers < C)
        if m.any():
            cc = centers[m]
            colc = cm[np.arange(cc.size) % max(G, 1)].copy()
            colc[:, 3] = 1.0
            ax.scatter(
                X[cc],
                Y[cc],
                s=float(peak_size),
                c=colc,
                marker="o",
                edgecolors="k",
                linewidths=1.2,
                zorder=5,
            )

        ax.set_xlabel(rf"$z_{{{d0+1}}}$")
        ax.set_ylabel(rf"$z_{{{d1+1}}}$")

        _hc_set_ticks(ax, show_ticks=show_ticks)
        _hc_apply_sci_axes(ax, sci=sci, logx=False, logy=False)

        handles = [
            Line2D([0], [0], marker="o", linestyle="none", markerfacecolor="k", markeredgecolor="none",
                markersize=4.5, label=r"Micro $\mathcal{C}$"),
            Line2D([0], [0], marker="o", linestyle="none", markerfacecolor="none", markeredgecolor="k",
                markersize=8.5, label=r"Peaks $\mathcal{P}$"),
        ]
        if have_kappa:
            handles.append(Line2D([0], [0], color="k", lw=1.2, label=r"Edge $\kappa$"))

        leg = ax.legend(
            handles=handles,
            frameon=False,
            loc="best",
            borderaxespad=0.0,
            handlelength=1.2,
            labelspacing=0.25,
            handletextpad=0.35,
        )

        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

        path = _hc_savefig(fig, out_dir, name, "grouping", ext)
        if show:
            plt.show()
        else:
            plt.close(fig)
        return path

    def plot_centers_alpha(self):
        C = int(self.parent.c)
        imp = self.importance[:C].cpu().numpy()
        imp = imp / np.max(imp)
        centers = self.parent.mu[:C, :2].cpu().numpy()
        plt.figure(figsize=(6, 6))
        plt.scatter(centers[:, 0], centers[:, 1], alpha=imp, s=100, edgecolors='k')
        plt.xlabel('$x_1$'); plt.ylabel('$x_2$'); plt.title('Cluster centers with alpha proportional into importance')
        plt.show()

    

    def plot_kappa_graph_Sglo(self, use_groups: bool = False,
                              figsize=(6, 6),
                              edge_alpha: float = 0.2,
                              edge_lw: float = 0.4):

        C, dev, mm = self._ctx()
        if C == 0:
            print("[plot_kappa_graph_Sglo] No clusters (C=0).")
            return

        if not hasattr(self.parent, "S_glo"):
            raise RuntimeError("parent.S_glo not found. Needed to pick top-variance dims.")

        S_glo = self.parent.S_glo.detach().cpu()
        if S_glo.dim() == 2:
            var_g = torch.diag(S_glo)
        else:
            var_g = S_glo.reshape(-1)

        D = int(self.parent.feature_dim)
        if D < 2:
            raise RuntimeError(f"Need at least 2 dims to plot, got D={D}.")

        k = min(2, D)
        _, idx = torch.topk(var_g, k=k, largest=True)
        dims = idx.tolist()
        if len(dims) < 2:
            raise RuntimeError("Could not select 2 dimensions from S_glo.")

        d0, d1 = dims[0], dims[1]

        mu = self.parent.mu[:C].detach().cpu()
        X = mu[:, d0].numpy()
        Y = mu[:, d1].numpy()

        if (not hasattr(mm, "edge_i")) or mm.edge_i.numel() == 0:
            print("[plot_kappa_graph_Sglo] No edges in merging_mech.edge_i/j.")
            return

        ei = mm.edge_i.long().detach().cpu().numpy()
        ej = mm.edge_j.long().detach().cpu().numpy()

        mask_valid = (ei >= 0) & (ei < C) & (ej >= 0) & (ej < C)
        ei = ei[mask_valid]
        ej = ej[mask_valid]

        lines = [((X[i], Y[i]), (X[j], Y[j])) for i, j in zip(ei, ej)]

        fig, ax = plt.subplots(figsize=figsize)

        if lines:
            lc = LineCollection(
                lines,
                linewidths=edge_lw,
                colors="k",
                alpha=edge_alpha,
                zorder=1,
            )
            ax.add_collection(lc)

        if use_groups and hasattr(self, "cluster_to_group") and self.cluster_to_group is not None:
            g = self.cluster_to_group[:C].detach().cpu().numpy()
            sc = ax.scatter(
                X, Y,
                c=g,
                s=20,
                cmap="tab20",
                edgecolors="k",
                linewidths=0.3,
                zorder=3,
            )
            plt.colorbar(sc, ax=ax, label="group id")
        else:
            ax.scatter(
                X, Y,
                s=20,
                c="tab:blue",
                edgecolors="k",
                linewidths=0.3,
                zorder=3,
            )

        ax.set_xlabel(f"μ[dim {d0}] (top S_glo)")
        ax.set_ylabel(f"μ[dim {d1}] (2nd S_glo)")
        ax.set_title("κ-graph on top-variance S_glo dimensions")
        
        plt.tight_layout()
        plt.show()

    @torch.no_grad()
    def compute_components_global(self, verbose: bool = True, sort_by: str = "samples", max_iter: int = 50):
        parent = self.parent
        dev = parent.mu.device
        C = int(getattr(parent, "c", 0))

        if C <= 0:
            if verbose:
                print("[compute_components_global] No clusters (C=0).")
            self.cluster_to_component = torch.empty(0, dtype=torch.long, device=dev)
            self.component_n_clusters = torch.empty(0, dtype=torch.long, device=dev)
            self.component_n_samples  = torch.empty(0, dtype=parent.n.dtype, device=dev)
            self.n_components = 0
            self.kappa_component_id = self.cluster_to_component
            self.kappa_component_stats = []
            return

        mm = parent.merging_mech

        if (not hasattr(mm, "edge_i")) or mm.edge_i.numel() == 0:
            comp = torch.arange(C, device=dev, dtype=torch.long)
            comp_clusters = torch.ones(C, device=dev, dtype=torch.long)
            comp_samples  = parent.n[:C].detach()

            self.cluster_to_component = comp
            self.component_n_clusters = comp_clusters
            self.component_n_samples  = comp_samples
            self.n_components = C

            self.kappa_component_id = comp
            self.kappa_component_stats = [
                {"component": k, "clusters": 1, "samples": float(comp_samples[k].item())}
                for k in range(C)
            ]
            if verbose:
                for r in self.kappa_component_stats:
                    print(f"  comp {r['component']:4d}  clusters={r['clusters']:6d}  samples={r['samples']:12.1f}")
            return

        ei = mm.edge_i.to(device=dev, dtype=torch.long)
        ej = mm.edge_j.to(device=dev, dtype=torch.long)

        mask = (ei >= 0) & (ei < C) & (ej >= 0) & (ej < C)
        ei = ei[mask]
        ej = ej[mask]

        if ei.numel() == 0:
            comp = torch.arange(C, device=dev, dtype=torch.long)
            comp_clusters = torch.ones(C, device=dev, dtype=torch.long)
            comp_samples  = parent.n[:C].detach()
            self.cluster_to_component = comp
            self.component_n_clusters = comp_clusters
            self.component_n_samples  = comp_samples
            self.n_components = C
            self.kappa_component_id = comp
            self.kappa_component_stats = [
                {"component": k, "clusters": 1, "samples": float(comp_samples[k].item())}
                for k in range(C)
            ]
            return

        label = torch.arange(C, device=dev, dtype=torch.long)

        for _ in range(max_iter):
            lu = label[ei]
            lv = label[ej]
            m  = torch.minimum(lu, lv)

            new_label = label.clone()
            new_label.scatter_reduce_(0, ei, m, reduce="amin")
            new_label.scatter_reduce_(0, ej, m, reduce="amin")

            new_label = new_label[new_label]

            if torch.equal(new_label, label):
                break
            label = new_label

        _, comp = torch.unique(label, return_inverse=True)
        K = int(comp.max().item()) + 1

        n = parent.n[:C].detach().to(dev)
        comp_samples = torch.zeros(K, device=dev, dtype=n.dtype).scatter_add_(0, comp, n)
        comp_clusters = torch.bincount(comp, minlength=K)

        if sort_by in ("samples", "clusters"):
            key = comp_samples if sort_by == "samples" else comp_clusters.to(comp_samples.dtype)
            order = torch.argsort(key, descending=True)
            new_id = torch.empty(K, device=dev, dtype=torch.long)
            new_id[order] = torch.arange(K, device=dev, dtype=torch.long)

            comp = new_id[comp]
            comp_samples = comp_samples[order]
            comp_clusters = comp_clusters[order]
            K = int(comp_samples.numel())

        self.cluster_to_component = comp
        self.component_n_samples  = comp_samples
        self.component_n_clusters = comp_clusters
        self.n_components = int(K)

        self.kappa_component_id = comp
        self.kappa_component_stats = [
            {"component": k,
            "clusters": int(comp_clusters[k].item()),
            "samples":  float(comp_samples[k].item())}
            for k in range(self.n_components)
        ]

        if verbose:
            for r in self.kappa_component_stats:
                print(f"  comp {r['component']:4d}  clusters={r['clusters']:6d}  samples={r['samples']:12.1f}")

    def plot_kappa_components_Sglo(
        self,
        draw_edges: bool = True,
        edge_alpha: float = 0.2,
        edge_lw: float = 0.4,
        marker_size: float = 25.0,
        alpha_pts: float = 0.99,
        cmap: str = "gist_ncar",
        sort_by: str = "clusters",
    ):

        import numpy as np
        import torch
        import matplotlib.pyplot as plt
        from matplotlib.cm import get_cmap
        from matplotlib.collections import LineCollection
        from matplotlib.lines import Line2D

        C = int(self.parent.c)
        if C == 0:
            print("[plot_kappa_components_Sglo] No clusters (C=0).")
            return

        if not hasattr(self, "kappa_component_id") or not hasattr(self, "kappa_component_stats"):
            raise RuntimeError("Call compute_components_global() first (plot does not compute components).")

        labels = self.kappa_component_id[:C].detach().cpu().numpy().astype(np.int64)

        rows = list(self.kappa_component_stats)
        if sort_by == "clusters":
            rows.sort(key=lambda r: r["clusters"], reverse=True)
        else:
            rows.sort(key=lambda r: r["samples"], reverse=True)

        print("===== κ Connected Components =====")
        print(f"num_components = {len(rows)}")
        for r in rows:
            print(f"  comp {r['component']:4d}  clusters={r['clusters']:6d}  samples={r['samples']:12.1f}")
        print("=================================")

        if not hasattr(self.parent, "S_glo"):
            raise RuntimeError("parent.S_glo not found. Needed to pick top-variance dims.")

        S_glo = self.parent.S_glo.detach().cpu()
        var_g = torch.diag(S_glo) if S_glo.dim() == 2 else S_glo.reshape(-1)

        D = int(self.parent.feature_dim)
        if D < 2:
            raise RuntimeError(f"Need at least 2 dims to plot, got D={D}.")

        _, idx = torch.topk(var_g, k=2, largest=True)
        d0, d1 = int(idx[0].item()), int(idx[1].item())

        mu = self.parent.mu[:C].detach().cpu()
        X = mu[:, d0].numpy()
        Y = mu[:, d1].numpy()

        K = int(labels.max() + 1) if labels.size else 1
        pal = get_cmap(cmap, max(K, 1))
        colors = pal(labels)
        colors[:, 3] = alpha_pts

        def _segments_from_edges(ei, ej, max_edges: int):
            if ei.size == 0:
                return None
            E = ei.size
            if E > max_edges:
                step = max(1, E // max_edges)
                take = np.arange(0, E, step, dtype=np.int64)[:max_edges]
                ei = ei[take]
                ej = ej[take]
            seg = np.empty((ei.size, 2, 2), dtype=np.float32)
            seg[:, 0, 0] = X[ei]; seg[:, 0, 1] = Y[ei]
            seg[:, 1, 0] = X[ej]; seg[:, 1, 1] = Y[ej]
            return seg

        seg_kappa = None
        seg_knn = None
        if draw_edges:
            mm = self.parent.merging_mech

            MAX_KAPPA_EDGES = 50_000
            MAX_KNN_EDGES   = 50_000

            if hasattr(mm, "edge_i") and mm.edge_i.numel() > 0:
                ei = mm.edge_i.long().detach().cpu().numpy()
                ej = mm.edge_j.long().detach().cpu().numpy()
                mask = (ei >= 0) & (ei < C) & (ej >= 0) & (ej < C)
                ei = ei[mask]; ej = ej[mask]
                seg_kappa = _segments_from_edges(ei, ej, MAX_KAPPA_EDGES)

            if hasattr(mm, "knn_edge_i") and mm.knn_edge_i.numel() > 0:
                ki = mm.knn_edge_i.long().detach().cpu().numpy()
                kj = mm.knn_edge_j.long().detach().cpu().numpy()
                mask = (ki >= 0) & (ki < C) & (kj >= 0) & (kj < C)
                ki = ki[mask]; kj = kj[mask]
                seg_knn = _segments_from_edges(ki, kj, MAX_KNN_EDGES)

        fig, ax = plt.subplots(figsize=(6, 6))

        if seg_knn is not None:
            lc_knn = LineCollection(seg_knn, linewidths=edge_lw, colors="r", alpha=max(0.05, edge_alpha * 0.35), zorder=1)
            ax.add_collection(lc_knn)

        if seg_kappa is not None:
            lc_kap = LineCollection(seg_kappa, linewidths=edge_lw, colors="k", alpha=edge_alpha, zorder=2)
            ax.add_collection(lc_kap)

        ax.scatter(X, Y, s=marker_size, c=colors, edgecolors="k", linewidths=0.2, zorder=3)

        handles = []
        if seg_kappa is not None:
            handles.append(Line2D([0], [0], color="k", lw=1.5, label="κ-pruned edges"))
        if seg_knn is not None:
            handles.append(Line2D([0], [0], color="r", lw=1.5, label="kNN edges (unpruned)"))
        if handles:
            ax.legend(handles=handles, loc="best", frameon=True)

        ax.set_xlabel(f"μ[dim {d0}] (top S_glo)")
        ax.set_ylabel(f"μ[dim {d1}] (2nd S_glo)")
        ax.set_title("κ connected components + κ edges + kNN edges")
        ax.set_xticks([]); ax.set_yticks([])
        plt.tight_layout()
        plt.show()
   

    @torch.no_grad()
    def plot_decision_panels(
        self,
        grouping_idx: int = -1,
        colormap="tab10",
        alpha_floor: float = 0.05,
        show_nn: bool = True,
        center_size: float = 180.0,
        point_size: float = 28.0,
    ):
        C = int(self.parent.c)
        if C <= 0:
            print("No clusters.")
            return

        if not hasattr(self, "delta") or not hasattr(self, "imp_delta") or not hasattr(self, "nn_higher"):
            self._compute_delta_to_higher()

        if not hasattr(self, "groupings") or len(self.groupings) == 0:
            raise RuntimeError("self.groupings is empty; run your grouping computation first.")
        result = self.groupings[grouping_idx]
        groups  = result["groups"].long()
        centers = result["centers"].long()

        mu2 = self.parent.mu[:C, :2].detach().cpu().numpy()

        grp = groups.detach().cpu().numpy().astype(np.int64)
        G = int(centers.numel())
        cm = get_cmap(colormap)(np.linspace(0, 1, max(G, 1)))

        imp  = self.importance[:C].detach().cpu().numpy().astype(np.float64)
        delt = self.delta[:C].detach().cpu().numpy().astype(np.float64)
        impd = self.imp_delta[:C].detach().cpu().numpy().astype(np.float64)

        nn = self.nn_higher[:C].detach().cpu().numpy().astype(np.int64)

        def norm_to_alpha(x: np.ndarray) -> np.ndarray:
            xmin = np.nanmin(x)
            xmax = np.nanmax(x)
            if not np.isfinite(xmin) or not np.isfinite(xmax) or xmax == xmin:
                a = np.ones_like(x, dtype=np.float64)
            else:
                a = (x - xmin) / (xmax - xmin)
                a = np.clip(a, 0.0, 1.0)
            return alpha_floor + (1.0 - alpha_floor) * a

        A_imp  = norm_to_alpha(imp)
        A_del  = norm_to_alpha(delt)
        A_impd = norm_to_alpha(impd)

        panels = [
            ("importance (alpha=min-max)", A_imp,  imp),
            ("delta (alpha=min-max)",      A_del,  delt),
            ("imp_delta (alpha=min-max)",  A_impd, impd),
        ]

        fig, axes = plt.subplots(1, 3, figsize=(16, 5), sharex=True, sharey=True)

        for ax, (title, A, raw) in zip(axes, panels):
            for c in range(C):
                g = grp[c]
                col = cm[g % len(cm)]

                a = float(A[c])
                pt = (col[0], col[1], col[2], a)

                ax.scatter(
                    mu2[c, 0], mu2[c, 1],
                    s=point_size, c=[pt], linewidths=0, zorder=3
                )

            centers_cpu = centers.detach().cpu().numpy().astype(np.int64)
            centers_mu  = self.parent.mu[centers, :2].detach().cpu().numpy()

            center_cols = cm[(np.arange(G) % len(cm))].copy()
            center_cols[:, 3] = A[centers_cpu]

            ax.scatter(
                centers_mu[:, 0], centers_mu[:, 1],
                c=center_cols,
                s=center_size,
                marker="o",
                edgecolors="k",
                linewidths=2.2,
                zorder=7,
            )

            if show_nn:
                for i in range(C):
                    j = nn[i]
                    if j >= 0:
                        ax.plot(
                            [mu2[i, 0], mu2[j, 0]],
                            [mu2[i, 1], mu2[j, 1]],
                            color=(0, 0, 0, 0.10),
                            lw=1.0,
                            zorder=2,
                        )

            ax.set_title(title)
            ax.set_xlabel("mu[0]")

        axes[0].set_ylabel("mu[1]")
        plt.tight_layout()
        plt.show()

        def stats(x):
            return f"min={np.min(x):.6g}, mean={np.mean(x):.6g}, max={np.max(x):.6g}"

        print(f"importance: {stats(imp)}")
        print(f"delta:      {stats(delt)}")
        print(f"imp_delta:  {stats(impd)}")

