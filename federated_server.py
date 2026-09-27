import copy
import numpy as np
import torch
from scipy.stats import chi2
from torch.utils.data import TensorDataset

from model.eFDPC import eFDPC
from utils.utils_train import (
    train_model_sequentially_loader,
    train_model_sequentially_prealocated,
)

def efdpc_server(train_datasets, train_loaders, num_clients, num_clusters, rounds,
                 batch_size, local_model_params, federated_model_params, flag_use_loader,
                 debug_flag=0, flag_history=False, num_shufflers=3):
    federated_model = eFDPC(**federated_model_params)
    federated_model.set_debugging_flag(debug_flag)
    local_models = [eFDPC(**local_model_params) for _ in range(num_clients)]
    history = [] if flag_history else None

    for r in range(rounds):
        if flag_use_loader:
            local_models = train_model_sequentially_loader(
                local_models, train_loaders, round=r, debug_flag=debug_flag)
        else:
            local_models = train_model_sequentially_prealocated(
                local_models, train_datasets, batch_size, round=r, debug_flag=debug_flag)

        shufflers = [eFDPC(**federated_model_params) for _ in range(num_shufflers)]
        for sm in shufflers:
            sm.set_debugging_flag(debug_flag)
            sm.c = 0
            sm.shuffler._ensure_key_buffers(sm)

        for cm in local_models:
            cm.federal_agent.federated_merging()
            cm.hierarch.cluster_communities(num_knees=1, initial_guess=num_clusters)
            cm.S[:cm.c] /= cm.n[:cm.c, None]
            nmin, nmax = torch.min(cm.n[:cm.c]), torch.max(cm.n[:cm.c])
            cm.n[:cm.c] = (cm.n[:cm.c] - nmin) / (nmax - nmin)
            cm.hierarch.remove_all_except_centers()

            payloads = cm.shuffler.client_make_shuffler_payloads(
                num_shufflers=num_shufflers, federated_model_params=federated_model_params)
            for s in range(num_shufflers):
                shufflers[s].shuffler.shuffler_receive_payload(payloads[s])

        for sm in shufflers:
            sm.shuffler.shuffle()

        reconstructed_model = eFDPC(**federated_model_params)
        reconstructed_model.set_debugging_flag(debug_flag)
        reconstructed_model.c = 0
        reconstructed_model.shuffler._ensure_key_buffers(reconstructed_model)
        reconstructed_model.shuffler.aggregator_reconstruct_from_shufflers_inplace(
            shufflers=shufflers, require_all=True)

        federated_model.federal_agent.merge_model_privately(reconstructed_model, 0, 0)
        federated_model.federal_agent.reset_covariances_to_S0()
        federated_model.federal_agent.renormalize_mass()

        consolidated_model = copy.deepcopy(federated_model)
        consolidated_model.hierarch.cluster_communities(num_knees=1, initial_guess=num_clusters)
        consolidated_model.hierarch.remove_all_except_centers()

        if flag_history:
            federated_model.hierarch.cluster_communities(num_knees=1, initial_guess=num_clusters)
            federated_model.hierarch.compute_groups(initial_guess=num_clusters)
            history.append(copy.deepcopy(federated_model))

        if r < (rounds - 1):
            for icm in range(len(local_models)):
                local_models[icm].federal_agent.merge_model_privately(consolidated_model, 0, 0)

    federated_model.hierarch.cluster_communities(num_knees=1, initial_guess=num_clusters)
    federated_model.hierarch.compute_groups(initial_guess=num_clusters)
    return history if flag_history else federated_model


def efdpc_params(d, P, batch_size, device="cpu"):
    return {
        "feature_dim": d,
        "num_classes": 1,
        "kappa_n": 0,
        "num_sigma": d,
        "kappa_join": float(1.0 + 0.25 * chi2.ppf(0.5, d) / d),
        "N_r": max(4 * P, d),
        "c_max": 2 * batch_size,
        "kappa_features": 1,
        "device": torch.device(device),
        "thr_relevance": 1,
    }


def fit_efdpc(client_data, P, batch_size=None, rounds=1, num_shufflers=3, device="cpu"):
    clients = [torch.as_tensor(np.asarray(x), dtype=torch.float32) for x in client_data]
    d = clients[0].shape[1]
    batch_size = batch_size or int(np.sqrt(sum(len(x) for x in clients)))
    datasets = [TensorDataset(x, torch.zeros(len(x), dtype=torch.int64)) for x in clients]
    params = efdpc_params(d, P, batch_size, device)
    return efdpc_server(datasets, None, len(datasets), P, rounds, batch_size, params, params,
                        flag_use_loader=False, num_shufflers=num_shufflers)


def predict_efdpc(model, X):
    X = torch.as_tensor(np.asarray(X), dtype=torch.float32)
    c = int(model.c)
    mu = model.mu[:c].detach().cpu()
    peaks = model.hierarch.centers[0].long().cpu()
    groups = model.hierarch.cluster_to_group[:c].long().cpu()
    return {"peaks": torch.cdist(X, mu[peaks]).argmin(1).numpy(),
            "groups": groups[torch.cdist(X, mu).argmin(1)].numpy()}
