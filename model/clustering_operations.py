import torch
import math

class ClusteringOps:
    def __init__(self, parent):
        self.parent = parent
        self.feature_dim = parent.feature_dim
        self.Gamma_max = torch.exp(-self.parent.min_d2/self.feature_dim)
        

    def fps_mahalanobis(self, X: torch.Tensor, n_clusters: int,
                        start_idx: int | None = None) -> tuple[torch.Tensor, torch.Tensor]:

        n, d   = X.shape
        dev    = X.device
        dtype  = X.dtype

        S0     = self.parent.S_0.to(dev).to(dtype)
        if S0.dim() > 1:
            S0 = torch.diag(S0)
        S0_inv = 1.0 / S0

        C      = torch.empty((n_clusters, d), device=dev, dtype=dtype)
        idxs   = torch.empty(n_clusters,    device=dev, dtype=torch.long)

        if start_idx is None:
            start_idx = torch.argmin(X[:, 0])
        C[0]   = X[start_idx]
        idxs[0] = start_idx

        diff    = X - C[0]
        min_d2  = (diff**2 * S0_inv).sum(dim=1)

        for i in range(1, n_clusters):
            next_idx      = torch.argmax(min_d2)
            C[i]          = X[next_idx]
            idxs[i]       = next_idx

            diff          = X - C[i]
            new_d2        = (diff**2 * S0_inv).sum(dim=1)
            min_d2        = torch.minimum(min_d2, new_d2)

        return C, idxs

    @torch.no_grad()
    def add_new_clusters(self, Z, n, y, min_pts: int = 2):
        dev = self.parent.device
        Z, n, y = Z.to(dev), n.to(dev), y.to(dev)
        m, d = Z.shape
        if m == 0:
            return

        k_try = max(1, int(math.ceil(m / float(self.parent.n_cap))))

        seeds, _ = self.fps_mahalanobis(Z, k_try)

        S0 = self.parent.S_0.to(dev, dtype=Z.dtype)
        if S0.dim() > 1:
            S0 = torch.diag(S0)
        S0_inv = S0.reciprocal()

        z2  = (Z.square() * S0_inv).sum(1, keepdim=True)
        c2  = (seeds.square() * S0_inv).sum(1).unsqueeze(0)
        d2  = z2 + c2 - 2.0 * ((Z * S0_inv) @ seeds.t())

        minv, win = d2.min(1)
        ok = minv <= float(self.parent.min_d2)
        if not ok.any():
            self.batch_add_cluster(Z, y, n)
            return

        cnt = torch.zeros(k_try, device=dev, dtype=torch.float32)
        cnt.index_add_(0, win[ok], torch.ones_like(minv[ok], dtype=torch.float32))
        keep = cnt >= float(min_pts)

        ok = ok & keep[win]
        if not ok.any():
            self.batch_add_cluster(Z, y, n)
            return

        kept = keep.nonzero(as_tuple=False).view(-1)
        centers = seeds[kept]

        c_before = self.parent.c
        self.batch_add_cluster(
            centers,
            torch.zeros(centers.size(0), device=dev, dtype=torch.long),
            torch.zeros(centers.size(0), device=dev, dtype=n.dtype),
        )

        remap = -torch.ones(k_try, device=dev, dtype=torch.long)
        remap[kept] = torch.arange(kept.numel(), device=dev, dtype=torch.long)
        winners_global = c_before + remap[win[ok]]

        self.batch_increment_clusters(Z[ok], y[ok], n[ok], winners_global)

        left = ~ok
        if left.any():
            self.batch_add_cluster(Z[left], y[left], n[left])

    def update_S_0(self, decay=1):
        s0_new = self.parent.var_glo / self.parent.N_r
        s0_new.clamp_(min=self.parent.S_0_init)

        self.parent.S_0 = s0_new
        C = int(self.parent.c)
        nC = self.parent.n[:C].unsqueeze(1)
        S_tgt = nC * self.parent.S_0.unsqueeze(0)

        self.parent.S[:C].mul_(1.0 - decay).add_(S_tgt, alpha=decay)
        self.parent.S[:C] = torch.maximum(self.parent.S[:C], S_tgt)

        self.parent.S_inv[:C] = 1.0 / (self.parent.S_0)

    @torch.no_grad()
    def batch_add_cluster(self, Z, y, n, S=None):
        dev   = self.parent.device
        dtype = Z.dtype

        num_added = int(Z.shape[0])
        if num_added == 0:
            return

        reserve    = 2
        cap_target = int(self.parent.c) + num_added + reserve
        self.parent.overseer.ensure_capacity(cap_target)

        c0  = int(self.parent.c)
        idx = slice(c0, c0 + num_added)

        self.parent.mu[idx]    = Z
        self.parent.mu_og[idx] = Z
        self.parent.n[idx]     = n

        base_S = self.parent.S_0 if S is None else S
        if not torch.is_tensor(base_S):
            base_S = torch.as_tensor(base_S, device=dev, dtype=dtype)
        else:
            base_S = base_S.to(device=dev, dtype=dtype)

        if base_S.dim() > 1:
            base_S = torch.diag(base_S)

        self.parent.S[idx]     = n.unsqueeze(1) * base_S
        self.parent.S_inv[idx] = 1.0 / torch.clamp(base_S, min=self.parent.epsilon)

        self.parent.cluster_labels[idx] = self.parent.one_hot_labels[y]

        if c0 == 0:
            self.parent.d2 = torch.ones(Z.size(0), num_added, device=dev, dtype=dtype)
        else:
            d2_old = self.parent.d2
            d2_new = torch.ones(d2_old.size(0), c0 + num_added, device=dev, dtype=d2_old.dtype)
            d2_new[:, :c0] = d2_old
            self.parent.d2 = d2_new

        self.parent.c = c0 + num_added

    def batch_update_clusters(self, Z, y, n):
        C = int(self.parent.c)
        if C == 0 or self.parent.d2.numel() == 0:
            self.batch_add_cluster(Z, y, n)
            return

        d2 = self.parent.d2[:, :C]
        full = self.parent.n[:C] >= self.parent.n_cap

        d2_eff = d2.masked_fill(full.unsqueeze(0), float('inf'))
        d2_eff = d2_eff.masked_fill(~torch.isfinite(d2_eff), float('inf'))

        min_vals, min_idx = d2_eff.min(dim=1)
        nearest_full = torch.isinf(min_vals)

        assign = (min_vals <= self.parent.min_d2)

        self.batch_increment_clusters(Z[assign], y[assign], n[assign], min_idx[assign])
        self.batch_add_cluster(Z[~assign], y[~assign], n[~assign])

    @torch.no_grad()
    def batch_increment_clusters(self, Z, y, n, jm):
        if Z.numel() == 0:
            return
        dev = Z.device
        D = Z.shape[1]

        n_f = n.to(torch.float32)

        uniq, inv = torch.unique(jm, return_inverse=True)
        U = uniq.numel()

        Zw = Z * n_f.unsqueeze(1)
        Z2w = Z.square() * n_f.unsqueeze(1)

        num_added   = torch.zeros(U, device=dev).index_add_(0, inv, n_f)
        sum_Z       = torch.zeros((U, D), device=dev, dtype=Z.dtype).index_add_(0, inv, Zw)
        sum_Z2_raw  = torch.zeros((U, D), device=dev, dtype=Z.dtype).index_add_(0, inv, Z2w)

        num_pre = self.parent.n[uniq].to(torch.float32)
        mu_pre  = self.parent.mu[uniq]
        S_pre   = self.parent.S[uniq]

        sum_Z2 = sum_Z2_raw - 2.0 * mu_pre * sum_Z + (mu_pre.square() * num_added.unsqueeze(1))

        num_post = num_pre + num_added
        mu_post  = (num_pre.unsqueeze(1) * mu_pre + sum_Z) / num_post.unsqueeze(1)
        S_post   = S_pre + sum_Z2

        self.parent.n[uniq]    = num_post.to(self.parent.n.dtype)
        self.parent.mu[uniq]   = mu_post
        self.parent.S[uniq]    = S_post
        self.parent.S_inv[uniq] = 1.0 / torch.clamp(
            S_post / (num_post.unsqueeze(1) + D),
            min=self.parent.epsilon
        )

    @torch.no_grad()
    def batch_update_global_statistics(self, Z, y):
        C = int(self.parent.num_classes)
        device = self.parent.device
        B, F = Z.shape

        counts = torch.bincount(y, minlength=C).to(Z.dtype)

        sums  = torch.zeros(C, F, device=device, dtype=Z.dtype)
        sumsq = torch.zeros(C, F, device=device, dtype=Z.dtype)
        sums.index_add_(0, y, Z)
        sumsq.index_add_(0, y, Z * Z)

        n_prev = self.parent.n_glo.clone()
        N_prev = n_prev.sum()
        n_b    = counts.sum()

        m_b = Z.mean(0)
        Zc  = Z - m_b
        M2_b_diag = (Zc * Zc).sum(0)

        self.parent.n_glo[:C] += counts
        N_new   = self.parent.n_glo.sum()
        mu_prev = self.parent.mu_glo
        delta   = m_b - mu_prev
        self.parent.mu_glo = mu_prev + delta * (n_b / N_new)

        diag = torch.diagonal(self.parent.S_glo)
        diag += M2_b_diag + (delta * delta) * (N_prev * n_b / N_new)

        self.parent.var_glo = diag / N_new

        self.update_S_0()

        cnt    = counts.clamp_min(1.0).unsqueeze(1)
        m_b_c  = sums / cnt
        M2_b_c = sumsq - cnt * (m_b_c * m_b_c)

        mu_prev_c = self.parent.mu_cls
        n_prev_c  = n_prev[:C].unsqueeze(1)
        n_new_c   = self.parent.n_glo[:C].unsqueeze(1)
        delta_c   = m_b_c - mu_prev_c

        self.parent.mu_cls = mu_prev_c + delta_c * (cnt / n_new_c)
        self.parent.S_cls  = self.parent.S_cls + M2_b_c + (delta_c * delta_c) * (n_prev_c * cnt / n_new_c)

    def update_clustering_condition(self):
    
        self.parent.cond_cov = self.parent.feature_dim < 10
