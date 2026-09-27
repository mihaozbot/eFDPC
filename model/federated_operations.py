import torch

class FederalOps:
    def __init__(self, parent):
        self.parent = parent
        self.parent.dp_on = False
        self.parent.dp_eps = 30.0
        self.parent.dp_delta = 1e-5
        self.parent.dp_R = 1.0
        self.parent.dp_n_min = 1.0
        self.min_d2 = ((self.parent.num_sigma) ** 2)
        self.shuffler_debug = []

    @torch.no_grad()
    def renormalize_mass(self, eps: float = 1e-12):
        p = self.parent
        if int(p.c) == 0:
            return

        nmax = p.n[:p.c].max().clamp_min(eps)
        p.n.div_(nmax)
        p.S.div_(nmax)

        ngmax = p.n_glo.max().clamp_min(eps)
        p.n_glo.div_(ngmax)
        p.S_glo.div_(ngmax)
        p.S_cls.div_(ngmax)

        p.var_glo = p.S_glo / p.n_glo.sum().clamp_min(eps)

    def merge_model_statistics(self, model):
            n_fed = torch.sum(self.parent.n_glo)
            n_local = torch.sum(model.n_glo.clone())

            self.parent.n_glo = (self.parent.n_glo + model.n_glo.clone())

            self.parent.S_glo = (self.parent.S_glo + model.S_glo.clone()) + \
                                ((n_fed * n_local) / (n_fed + n_local)) * (self.parent.mu_glo - model.mu_glo.clone()) ** 2
            self.parent.mu_glo = (self.parent.mu_glo * n_fed + model.mu_glo.clone() * n_local) / (n_fed + n_local)
            self.parent.var_glo = torch.diag(self.parent.S_glo) / (n_fed + n_local)

            self.parent.N_r = (float(self.parent.N_r) * (n_fed) + model.N_r * (n_local)) / ((n_fed + n_local))

            self.parent.clusterer.update_S_0()

            n_local = self.parent.n_glo.clone()
            n_fed = model.n_glo.clone()

            n_local_exp = n_local.unsqueeze(1)
            n_fed_exp = n_fed.unsqueeze(1)
            n_sum_exp = (n_local + n_fed).unsqueeze(1)

            self.parent.mu_cls = (self.parent.mu_cls * n_fed_exp + model.mu_cls * n_local_exp) / n_sum_exp

            mu_diff = self.parent.mu_cls - model.mu_cls
            scatter_update = (n_fed_exp * n_local_exp / n_sum_exp) * mu_diff ** 2

            self.parent.S_cls += model.S_cls + scatter_update
            
            
    @torch.no_grad()
    def merge_model_privately(self, model, n_min, pred_min):
        valid_clusters = (model.n[:model.c] > self.parent.kappa_n)
        valid_indices = torch.where(valid_clusters)[0]
        if int(valid_indices.numel()) == 0:
            return

        self.merge_model_statistics(model)
        self.parent.clusterer.update_clustering_condition()

        before_size = int(self.parent.c)
        after_size  = before_size + int(valid_indices.numel())
        self.parent.overseer.ensure_capacity(after_size)

        new_indices = torch.arange(before_size, before_size + int(valid_indices.numel()), device=model.mu.device)

        self.parent.mu.data[new_indices] = model.mu.data[valid_indices].clone()
        self.parent.mu_og.data[new_indices] = model.mu_og.data[valid_indices].clone()
        self.parent.S.data[new_indices] = model.S.data[valid_indices].clone()
        self.parent.n.data[new_indices] = model.n.data[valid_indices].clone()
        self.parent.S_inv.data[new_indices] = model.S_inv.data[valid_indices].clone()
        self.parent.cluster_labels[new_indices] = model.cluster_labels[valid_indices].clone()

        self.parent.d2 = torch.zeros(after_size, dtype=torch.float32, device=self.parent.device, requires_grad=False)
        self.parent.c = after_size

        self.parent.clusterer.update_S_0()

    @torch.no_grad()
    def _ensure_S_0_from_centers(self):
        C = int(self.parent.c)
        mu = self.parent.mu[:C].detach()
        dev = mu.device
        dtype = mu.dtype
        X = mu.to(torch.float64)
        X = X - X.mean(dim=0, keepdim=True)
        S0 = (X.T @ X) / float(C - 1)
        S0 = torch.diag(S0) + 1e-9
        self.parent.S_0 = S0.to(device=dev, dtype=dtype) / self.parent.N_r

    def reset_covariances_to_S0(self):
        self._ensure_S_0_from_centers()
        C = int(self.parent.c)
        self.parent.S[:C].copy_(self.parent.n[:C, None] * self.parent.S_0)
        self.parent.S_inv[:C].copy_(1.0 / self.parent.S[:C])

    def federated_merging(self):
        with torch.no_grad():
            cond_fed_merge = True
            while cond_fed_merge:
                ind_merged = self.parent.merging_mech.batch_merging_mechanism()
                cond_fed_merge = (ind_merged.shape[0] != 0)

