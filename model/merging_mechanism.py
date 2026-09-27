
import math
import torch

class MergingMechanism:
    def __init__(self, parent):
        self.parent = parent
        self.feature_dim = parent.feature_dim
        self.ln2 = torch.log(torch.tensor(2))

    def find_pairs_greedy(self):
        dev = self.parent.device
        C = int(self.parent.c)
        E = int(self.edge_i.numel())
        if E == 0 or C <= 1:
            return torch.empty((0, 2), device=dev, dtype=torch.long)

        kj = self.parent.kappa_join
        kj = float(kj.item()) if torch.is_tensor(kj) else float(kj)
        if kj <= 0.0:
            return torch.empty((0, 2), device=dev, dtype=torch.long)

        mask = (self.edge_i != self.edge_j) & (self.edge_kappa <= kj)
        ei = self.edge_i[mask]
        ej = self.edge_j[mask]
        ek = self.edge_kappa[mask]
        if ei.numel() == 0:
            return torch.empty((0, 2), device=dev, dtype=torch.long)

        a = torch.minimum(ei, ej)
        b = torch.maximum(ei, ej)

        key = a.to(torch.int64) * C + b.to(torch.int64)
        order = torch.argsort(key)
        key_s = key[order]
        a_s = a[order]
        b_s = b[order]
        ek_s = ek[order]

        uniq_key, inv = torch.unique_consecutive(key_s, return_inverse=True)
        best_k = torch.full((uniq_key.numel(),), float("inf"), device=dev, dtype=ek_s.dtype)
        best_k.scatter_reduce_(0, inv, ek_s, reduce="amin", include_self=True)

        first = torch.ones_like(inv, dtype=torch.bool)
        first[1:] = inv[1:] != inv[:-1]
        a_u = a_s[first].long()
        b_u = b_s[first].long()
        ek_u = best_k

        ord2 = torch.argsort(ek_u)
        a2 = a_u[ord2]
        b2 = b_u[ord2]

        used = torch.zeros((C,), device=dev, dtype=torch.bool)
        take = torch.zeros((a2.numel(),), device=dev, dtype=torch.bool)

        for t in range(a2.numel()):
            i = int(a2[t].item())
            j = int(b2[t].item())
            if (not used[i]) and (not used[j]):
                used[i] = True
                used[j] = True
                take[t] = True

        pairs = torch.stack([a2[take], b2[take]], dim=1)
        return pairs

            
    def perform_merge(self):
        pairs = self.find_pairs_greedy()
        if pairs.numel() == 0:
            return torch.tensor([], device=self.parent.device, dtype=torch.long)

        pairs = pairs[pairs[:, 0] != pairs[:, 1]]
        if pairs.numel() == 0:
            return torch.tensor([], device=self.parent.device, dtype=torch.long)
        
        pairs = pairs[pairs[:, 0] < pairs[:, 1]]
        if pairs.numel() == 0:
            return torch.tensor([], device=self.parent.device, dtype=torch.long)

        i_all, j_all = pairs[:, 0], pairs[:, 1]
        n_i, n_j  = self.parent.n[i_all], self.parent.n[j_all]
        n_ij      = n_i + n_j
        mu_i      = self.parent.mu[i_all]
        mu_j      = self.parent.mu[j_all]
        mu_ij     = (n_i.unsqueeze(1) * mu_i + n_j.unsqueeze(1) * mu_j) / n_ij.unsqueeze(1)

        S_i, S_j  = self.parent.S[i_all], self.parent.S[j_all]
        mu_diff2  = (mu_i - mu_j) ** 2
        cross     = (n_i * n_j / n_ij).unsqueeze(1) * mu_diff2 / self.parent.merging_weight
        S_ij      = S_i + S_j + cross

        self.parent.mu[i_all]    = mu_ij
        self.parent.S[i_all]     = S_ij
        self.parent.n[i_all]     = n_ij

        self.parent.S_inv[i_all] = 1.0 / (S_ij / (n_ij.unsqueeze(1) + self.parent.feature_dim))

        unlabeled_i = (self.parent.cluster_labels[i_all, 1:].sum(1) == 0)
        labeled_j   = (self.parent.cluster_labels[j_all, 1:].sum(1)  > 0)
        self.parent.cluster_labels[i_all[unlabeled_i & labeled_j]] = \
            self.parent.cluster_labels[j_all[unlabeled_i & labeled_j]]

        self.parent.removal_mech.remove_clusters(j_all)

        changed = torch.sort(torch.cat((i_all, j_all)))[0]
        changed = changed[changed < self.parent.c]

        return changed
    

    @torch.no_grad()
    def batch_compute_volume(
        self,
        eps: float = 1e-12,
        row_chunk: int = 0,
    ):
        C = int(self.parent.c)
        D = int(self.parent.feature_dim)
        dev = self.parent.device

        self.edge_i = torch.empty(0, dtype=torch.long, device=dev)
        self.edge_j = torch.empty(0, dtype=torch.long, device=dev)
        self.edge_kappa = torch.empty(0, dtype=torch.float32, device=dev)
        self.edge_mah_dist = torch.empty(0, dtype=torch.float32, device=dev)
        self._have_sparse = True

        self.knn_edge_i = torch.empty(0, dtype=torch.long, device=dev)
        self.knn_edge_j = torch.empty(0, dtype=torch.long, device=dev)
        self.knn_edge_kappa = torch.empty(0, dtype=torch.float32, device=dev)
        self.knn_mah_dist = torch.empty(0, dtype=torch.float32, device=dev)

        if C <= 1 or D <= 0:
            return

        mu = self.parent.mu[:C].to(dev)
        S  = self.parent.S[:C].to(dev)
        n  = self.parent.n[:C].to(dev)

        k_keep = math.ceil(math.sqrt(C))
        k_keep = max(1, min(C - 1, k_keep))

        if row_chunk is None or row_chunk <= 0:
            target_bytes = 128 * 1024 * 1024
            bytes_per_row = C * mu.element_size()
            row_chunk = max(32, min(1024, int(target_bytes // max(1, bytes_per_row))))
        row_chunk = int(row_chunk)

        mu2_all = (mu * mu).sum(1)
        muT = mu.t().contiguous()
        inf = torch.tensor(float("inf"), device=dev, dtype=mu.dtype)

        knn_j = torch.empty((C, k_keep), device=dev, dtype=torch.long)
        for s in range(0, C, row_chunk):
            e = min(C, s + row_chunk)
            B = e - s
            if B <= 0:
                continue

            mu2_blk = mu2_all[s:e]
            dist2 = (mu2_blk.unsqueeze(1) + mu2_all.unsqueeze(0) - 2.0 * (mu[s:e] @ muT)).clamp_min_(0.0)
            dist2[torch.arange(B, device=dev), torch.arange(s, e, device=dev)] = inf

            knn_j[s:e] = dist2.topk(k_keep, largest=False, dim=1).indices

        ii = torch.arange(C, device=dev, dtype=torch.long).unsqueeze(1)
        jj = knn_j

        n_i = n[ii]
        n_j = n[jj]
        nsum = (n_i + n_j).clamp_min(eps)

        Sigma12 = (S[ii] + S[jj]) / nsum.unsqueeze(-1)
        Sigma12 = Sigma12.clamp_min(eps)

        diff = mu[ii] - mu[jj]
        dM2  = (diff * diff / Sigma12).sum(-1)
        alpha = (n_i * n_j) / (nsum * nsum)
        kappa = 1.0 + alpha * dM2

        src = torch.arange(C, device=dev, dtype=torch.long).repeat_interleave(k_keep)
        dst = knn_j.reshape(-1)
        kap = kappa.to(torch.float32).reshape(-1)
        m2  = (dM2 / self.feature_dim).to(torch.float32).reshape(-1)

        self.knn_edge_i = src
        self.knn_edge_j = dst
        self.knn_edge_kappa = kap
        self.knn_mah_dist = m2

        key = src * C + dst
        rev = dst * C + src

        rev_sorted = torch.sort(rev).values.contiguous()
        pos = torch.searchsorted(rev_sorted, key)

        mutual = torch.zeros_like(key, dtype=torch.bool)
        ok = pos < rev_sorted.numel()
        pos_ok = pos[ok]
        mutual[ok] = (rev_sorted[pos_ok] == key[ok])

        keep = mutual & (src < dst)

        self.edge_i = src[keep].contiguous()
        self.edge_j = dst[keep].contiguous()
        self.edge_kappa = kap[keep].contiguous()
        self.edge_mah_dist = m2[keep].contiguous()

        self._have_sparse = True

        return
    

    def batch_merging_mechanism(self): 
            
            self.batch_compute_volume()
            i_merged = self.perform_merge()

            return i_merged
        

