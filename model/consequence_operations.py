import torch

class ConsequenceOps():
    def __init__(self, parent):
        self.parent = parent

        
    

    def safe_argmax(self, one_hot_max_labels, dim):
        if one_hot_max_labels.size(dim) == 0:
            new_shape = list(one_hot_max_labels.size())
            new_shape[dim] = 0
            return torch.empty(new_shape, dtype=torch.long, device=one_hot_max_labels.device)
        else:
            return torch.argmax(one_hot_max_labels, dim=dim)
        
    def defuzzify_batch(self, Z):
        
        normalized_gamma = self.parent.mathematician.compute_batched_normalized_gamma()

        expanded_cluster_labels = self.parent.cluster_labels[:self.parent.c].unsqueeze(0).expand(normalized_gamma.shape[0], -1, -1)

        label_scores = torch.sum(normalized_gamma.unsqueeze(-1) * expanded_cluster_labels[:,:,1:], dim=1)
                 
        if self.parent.class_type == 0:

            max_indices = torch.argmax(normalized_gamma, dim=1)
            
            one_hot_max_labels = expanded_cluster_labels[torch.arange(normalized_gamma.shape[0]), max_indices, 1:]
            
            max_labels = self.safe_argmax(one_hot_max_labels, dim=1)

        else:
            
            phi = torch.ones((self.parent.c, Z.shape[0], self.parent.n_phi), device=self.parent.device)

            for i in range(1, self.parent.order + 1):
                phi[:, :, i:(i + Z.shape[1])] = Z.unsqueeze(0).repeat(self.parent.c, 1, 1) ** i

            self.parent.all_scores = torch.softmax(torch.einsum('cbf, cfo -> boc', phi, self.parent.theta[:self.parent.c])[:,1:], dim=1)
            
            label_scores = torch.sum(self.parent.all_scores.transpose(1, 2)*normalized_gamma.unsqueeze(-1), dim=1)
            
            
            max_labels = torch.argmax(label_scores, dim=1)         
            
        return label_scores, max_labels

            
                
