import torch

class ModelOps:
    def __init__(self, parent):
        self.parent = parent
        
        self.parent.debug_flag = 0
        self.parent.enable_debugging = False
        self.parent.enable_adding = True
        self.parent.enable_merging = True
            
        self.par_lim = {
                'kappa_n': [1, self.parent.feature_dim],
                'num_sigma': [1, 2 * self.parent.feature_dim],
                'kappa_join': [0.3, 10],
                'N_r': [1, 2 * self.parent.feature_dim],
                'kappa_features': [0, 1],
                'c_max': [self.parent.num_classes, 100*self.parent.num_classes],
            }
        
    def _next_pow2(self, n: int) -> int:
        n = 1 if n <= 1 else int(n)
        p = 1
        while p < n:
            p <<= 1
        return p

    def ensure_capacity(self, new_c_host: int):
        """Expand capacity if needed. Uses only host ints (no GPU->CPU sync)."""
        with torch.no_grad():
            need = int(new_c_host)
            if need > self.parent.current_capacity:
                self.modify_capacity(need)

    def modify_capacity(self, new_c_host: int):
        with torch.no_grad():
            c_host   = int(self.parent.c)
            need     = max(int(new_c_host), c_host, 1)
            new_cap  = self._next_pow2(need)

            F        = int(self.parent.feature_dim)
            n_phi    = int(getattr(self.parent, "n_phi", 1))
            Cn       = int(self.parent.num_classes)

            def _resize(old: torch.Tensor, size):
                out = torch.zeros(size, dtype=old.dtype, device=old.device, requires_grad=False)
                rows = min(c_host, old.shape[0], out.shape[0])
                if rows:
                    out[:rows].copy_(old[:rows])
                return out

            self.parent.mu     = _resize(self.parent.mu,     (new_cap, F))
            self.parent.mu_og  = _resize(self.parent.mu_og,  (new_cap, F))
            self.parent.S      = _resize(self.parent.S,      (new_cap, F))
            self.parent.S_inv  = _resize(self.parent.S_inv,  (new_cap, F))
            self.parent.n      = _resize(self.parent.n,      (new_cap,))

            self.parent.cluster_labels = _resize(self.parent.cluster_labels, (new_cap, Cn))

            if hasattr(self.parent, "key_hi") and (self.parent.key_hi is not None):
                self.parent.key_hi = _resize(self.parent.key_hi, (new_cap,))
            if hasattr(self.parent, "key_lo") and (self.parent.key_lo is not None):
                self.parent.key_lo = _resize(self.parent.key_lo, (new_cap,))

            self.parent.current_capacity = int(new_cap)
    
    def toggle_adding(self, enable=None, print_flag=False):
        if enable is None:
            self.parent.enable_adding = not self.parent.enable_adding
            state = "enabled" if self.parent.enable_adding else "disabled"
        else:
            self.parent.enable_adding = enable
            state = "enabled" if enable else "disabled"
        if print_flag:
            print(f"Cluster adding has been {state}.")

    def toggle_merging(self, enable=None, print_flag=False):
        if enable is None:
            self.parent.enable_merging = not self.parent.enable_merging
            state = "enabled" if self.parent.enable_merging else "disabled"
        else:
            self.parent.enable_merging = enable
            state = "enabled" if enable else "disabled"
        if print_flag:
            print(f"Cluster merging has been {state}.")

    def toggle_debugging(self, enable=None, print_flag=False):
        if enable is None:
            self.parent.enable_debugging = not self.parent.enable_debugging
            state = "enabled" if self.parent.enable_debugging else "disabled"
        else:
            self.parent.enable_debugging = enable
            state = "enabled" if enable else "disabled"
        if print_flag:
            print(f"Debugging has been {state}.")
        
    def toggle_evolving(self, enable=None, print_flag=False):
        
        if enable is None:
            self.parent.evolving = not self.parent.evolving
        else:
            self.parent.evolving = enable
        
        self.parent.enable_adding = self.parent.evolving
        self.parent.enable_merging = self.parent.evolving

        state = "enabled" if self.parent.evolving else "disabled"
        if print_flag:
            print(f"Evolving has been {state}.")

    

