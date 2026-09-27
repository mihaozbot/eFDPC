import torch

class MathOps():
    def __init__(self, parent):
        self.parent = parent
        self.feature_dim = parent.feature_dim
        
    def compute_batched_normalized_gamma(self):
        
        gamma_per_class = self.parent.Gamma[:, :self.parent.c].unsqueeze(2) * self.parent.cluster_labels[:self.parent.c, :].unsqueeze(0)
        gamma_per_class = torch.nan_to_num(gamma_per_class, nan=0.0)
                
        sum_per_class = gamma_per_class.sum(dim=1, keepdim=True)
        sum_per_class = torch.nan_to_num(sum_per_class, nan=0.0)
        sum_per_class[sum_per_class == 0] = 1
        
        norm_gamma_per_class = (gamma_per_class / sum_per_class).sum(dim=2)
        
        sum_norm = (norm_gamma_per_class * self.parent.Gamma[:, :self.parent.c]).sum(dim=1, keepdim=True)
        sum_norm = torch.nan_to_num(sum_norm, nan=0.0)
        sum_norm[sum_norm == 0] = 1
        
        norm_gamma = (norm_gamma_per_class * self.parent.Gamma[:, :self.parent.c]) / sum_norm
        
        if self.parent.debug_flag == 4: 
            if torch.isnan(norm_gamma).any():
                print("Error normalization")
            if any(abs(torch.sum(norm_gamma, dim = 1) -1) > 1e-4 ):
                print( f"Sum gamma norm is equal to {torch.sum(norm_gamma, dim = 1)[abs(torch.sum(norm_gamma, dim = 1) -1) > 1e-1]}") 
        return norm_gamma
   

        

    @torch.no_grad()
    def compute_batched_activation_diagonal(self, Z):
       
        C, dev = self.parent.c, self.parent.device

        Fm = self.parent.feature_mask
        if bool(Fm.all()):
            Zs, mu, Sinv = Z, self.parent.mu[:C], self.parent.S_inv[:C]
        else:
            Zs   = Z[:, Fm]
            mu   = self.parent.mu[:C][:, Fm]
            Sinv = self.parent.S_inv[:C][:, Fm]

        Z2     = Zs.pow(2)
        beta   = mu * Sinv
        bconst = (mu.pow(2) * Sinv).sum(1)

        d2 = (Z2 @ Sinv.t()) - 2.0 * (Zs @ beta.t()) + bconst.unsqueeze(0)

        self.parent.d2 = torch.log1p(d2)

       
    def compute_batched_gamma_diagonal(self):    
        
        C = int(self.parent.c)
        Fm = self.parent.feature_mask
        d2 = self.parent.d2
        Sigma_diag = (self.parent.S[:C, Fm] / self.parent.n[:C, None]).clamp_min(1e-12)
        log_det = torch.log(Sigma_diag).sum(-1)
        log_score = -(d2 / self.parent.feature_dim) - log_det.unsqueeze(0)
        log_score = log_score - torch.amax(log_score, dim=1, keepdim=True)
        gamma = torch.exp(log_score)
        gamma = gamma / gamma.sum(dim=1, keepdim=True).clamp_min(1e-12)
        self.parent.Gamma = gamma

