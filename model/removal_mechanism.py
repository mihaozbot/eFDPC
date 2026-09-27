import torch

class RemovalMechanism:
    def __init__(self, parent):
        self.parent = parent
        self.push_th = (self.parent.num_sigma)**2
    

                    
                    

                    
    def remove_clusters(self, indices_to_remove):        

        self.parent.c  = self.parent.c - indices_to_remove.shape[0]

        indices_to_remove_filtered = indices_to_remove[indices_to_remove < self.parent.c ]

        valid_indices = torch.ones(self.parent.c + indices_to_remove.shape[0], dtype=torch.bool, device=self.parent.device)
        valid_indices[indices_to_remove] = 0
        remaining_indices = torch.nonzero(valid_indices).squeeze(1)

        replacement_indices = remaining_indices[remaining_indices >= self.parent.c]

        self.parent.mu.data[indices_to_remove_filtered] = self.parent.mu.data[replacement_indices]
        self.parent.mu_og.data[indices_to_remove_filtered] = self.parent.mu_og.data[replacement_indices]
        self.parent.S.data[indices_to_remove_filtered] = self.parent.S.data[replacement_indices]
        self.parent.n.data[indices_to_remove_filtered] = self.parent.n.data[replacement_indices]
        self.parent.S_inv.data[indices_to_remove_filtered] = self.parent.S_inv.data[replacement_indices]
        self.parent.cluster_labels[indices_to_remove_filtered] = self.parent.cluster_labels[replacement_indices]
        
      
    
        
        return indices_to_remove_filtered
    

    

        
        
        
        