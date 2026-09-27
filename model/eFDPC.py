import torch

from model.clustering_operations import ClusteringOps
from model.removal_mechanism import RemovalMechanism 
from model.merging_mechanism import MergingMechanism
from model.math_operations import MathOps
from model.consequence_operations import ConsequenceOps
from model.model_operations import ModelOps
from model.federated_operations import FederalOps
from model.hierarchical_clustering import HierarchicalClustering
from model.shuffler_privacy import ShufflerPrivacy

class eFDPC(torch.nn.Module):
    def __init__(self, feature_dim, num_classes, kappa_n, num_sigma, kappa_join, N_r, c_max, kappa_features, device, thr_relevance):
        super(eFDPC, self).__init__()
        self.device = device
        self.feature_dim = torch.tensor(feature_dim, device=self.device)
        self.kappa_n = torch.tensor(kappa_n, device=self.device)
        self.num_sigma = torch.tensor(num_sigma, device=self.device)
        self.kappa_join = torch.tensor(kappa_join, device=self.device)
        self.S_0_init = 10 ** (-64 / torch.sqrt(self.feature_dim)) * torch.ones(feature_dim, device=self.device)
        self.S_0 = self.S_0_init.clone()
        self.N_r = torch.tensor(N_r, device=self.device)
        self.num_classes = torch.tensor(num_classes, device=self.device)
        self.c_max = torch.tensor(c_max, device=self.device)
        self.order = torch.tensor(1, device=self.device)
        self.thr_relevance = torch.tensor(thr_relevance, device=self.device)
        self.n_phi = self.order * self.feature_dim + 1
        self.class_type = torch.tensor(0, device=self.device)
        self.kappa_features = torch.tensor(kappa_features, device=self.device)
        self.merging_weight = torch.tensor(1, device=self.device)
        self.epsilon = torch.tensor(1e-8, device=self.device)
        self.scale_unlabeled = torch.tensor((1 / 2) ** 2, device=self.device)
        self.student_factor = torch.tensor(1, device=self.device)
        self.eta = torch.tensor(1e-8, device=self.device)
        self.min_d2 = torch.log1p((self.num_sigma) ** 2)
        self.cap_sigma = 3
        self.n_cap = self.feature_dim + 1

        self.c = torch.tensor(0, dtype=torch.int32, device=self.device)
        self.current_capacity = c_max
        self.cluster_labels = torch.zeros((self.current_capacity, num_classes), dtype=torch.int32, device=device)
        self.score = torch.zeros((self.current_capacity,), dtype=torch.float32, device=device)
        self.num_pred = torch.zeros((self.current_capacity,), dtype=torch.float32, device=device)
        self.age = torch.zeros((self.current_capacity,), dtype=torch.float32, device=device)
        self.one_hot_labels = torch.eye(num_classes, dtype=torch.float16, device=device)
        self.feature_mask = torch.ones(feature_dim, dtype=torch.bool, device=device)

        self.n = torch.zeros(self.current_capacity, dtype=torch.float32, device=device, requires_grad=False)
        self.mu = torch.zeros(self.current_capacity, feature_dim, dtype=torch.float32, device=device, requires_grad=False)
        self.mu_og = torch.zeros(self.current_capacity, feature_dim, dtype=torch.float32, device=device, requires_grad=False)
        self.S = torch.zeros(self.current_capacity, feature_dim, dtype=torch.float32, device=device, requires_grad=False)
        self.S_inv = torch.zeros_like(self.S, requires_grad=False)
        self.L = None
        self.log_likelihood = None

        self.theta = None

        self.n_glo = torch.ones((num_classes), dtype=torch.float32, device=device)
        self.mu_glo = torch.zeros((feature_dim), dtype=torch.float32, device=device)
        self.mu_cls = torch.zeros((num_classes, feature_dim), dtype=torch.float32, device=device)

        self.S_glo = 10 ** (-64 / torch.sqrt(self.feature_dim)) * torch.ones((feature_dim, feature_dim), dtype=torch.float32, device=device)
        self.S_glo_chol = None

        self.S_cls = 10 ** (-64 / torch.sqrt(self.feature_dim)) * torch.ones((num_classes, feature_dim), dtype=torch.float32, device=device)

        self.overseer = ModelOps(self)
        self.mathematician = MathOps(self)
        self.clusterer = ClusteringOps(self)
        self.merging_mech = MergingMechanism(self)
        self.removal_mech = RemovalMechanism(self)
        self.consequence = ConsequenceOps(self)
        self.federal_agent = FederalOps(self)
        self.hierarch = HierarchicalClustering(self)
        self.shuffler = ShufflerPrivacy(self)

            
    def toggle_evolving(self, enable=None):
        self.overseer.toggle_evolving(enable)

    def toggle_adding(self, enable=None):
        self.overseer.toggle_adding(enable)

    def toggle_merging(self, enable=None):
        self.overseer.toggle_merging(enable)

    def toggle_debugging(self, enable=None):
        self.overseer.toggle_debugging(enable)

    def set_debugging_flag(self, debug_flag):
        self.debug_flag = debug_flag

    def federated_merging(self):
        self.federal_agent.federated_merging()

    def recluster_outliers(self, frac=0.9, max_remove=None):
        C = int(self.c)
        n = self.n[:C]

        i_single = torch.where(n == 1)[0]
        if i_single.numel() == 0:
            return

        k = int((frac * i_single.numel()) + 0.9999)
        if max_remove is not None:
            k = min(k, int(max_remove))
        k = max(1, min(k, i_single.numel()))

        perm = torch.randperm(i_single.numel(), device=i_single.device)
        i_remove = i_single[perm[:k]]

        Z = self.mu[i_remove]
        y = torch.zeros(Z.shape[0], device=Z.device, dtype=torch.long)

        self.removal_mech.remove_clusters(i_remove)
        if i_remove.numel() > 0:
            self.batch_clustering(Z, y)

        if C > self.c_max:
            self.hierarch.cluster_communities(num_knees=1, initial_guess=self.c_max ) 
            self.hierarch.remove_all_except_centers()
                

    def batch_clustering(self, Z, y):
        
            
            self.clusterer.update_clustering_condition()
            
            self.fisher_scores = torch.ones(self.feature_dim, dtype=torch.bool, device=self.device)
            self.clusterer.batch_update_global_statistics(Z, y)
            

            self.mathematician.compute_batched_activation_diagonal(Z)
                

            self.clusterer.batch_update_clusters(Z, y, torch.ones(len(Z), device=self.device))
            

    def forward(self, Z):
        self.fisher_scores = torch.ones(self.feature_dim, dtype=torch.bool, device=self.device)
        self.feature_mask = (self.fisher_scores >= self.kappa_features).to(self.device)
        
        self.mathematician.compute_batched_activation_diagonal(Z)
        self.mathematician.compute_batched_gamma_diagonal()                                       
        

        soft_scores, preds_max = self.consequence.defuzzify_batch(Z) 

        clusters = self.Gamma.argmax(dim=1)
        return soft_scores, preds_max, clusters
    