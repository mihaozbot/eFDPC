import torch
import numpy as np

from torch.utils.data import DataLoader, TensorDataset


def prepare_iid_federated_dataset_for_clustering(X, y, num_clients, batch_size=32):

    X = X.clone().detach()
    y = y.clone().detach()

    y_train = torch.zeros_like(y)

    X_train_split = torch.split(X, int(len(X) / num_clients))
    y_train_split = torch.split(y_train, int(len(y_train) / num_clients))

    train_datasets = [TensorDataset(x[(idx := torch.randperm(x.size(0)))], y[idx])
                    for x, y in zip(X_train_split, y_train_split)]
    
    train_loaders = [DataLoader(train_datasets[i], batch_size=batch_size, shuffle=True, pin_memory=True) for i in range(num_clients)]

    test_dataset = TensorDataset(X, y)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, pin_memory=True)

    all_data = (X, y)
    train_data = (X_train_split, y_train_split)
    
    return train_loaders, train_datasets, test_loader, test_dataset, all_data, train_data


def prepare_non_iid_federated_dataset_for_clustering(
    X,
    y,
    num_clients,
    alpha=0.9,
    batch_size=32,
    min_samples_per_client: int | None = None,
):

    if not torch.is_tensor(X):
        X = torch.tensor(X)
    if not torch.is_tensor(y):
        y = torch.tensor(y)
    y = y.clone().detach().squeeze()

    N = int(y.numel())

    classes = torch.unique(y).tolist()
    num_classes = len(classes)

    if min_samples_per_client is not None:
        min_samples_per_client = int(min_samples_per_client)
        if N < num_clients * min_samples_per_client:
            raise ValueError(
                f"Not enough samples N={N} to guarantee "
                f"{min_samples_per_client} per client for {num_clients} clients "
                f"(need at least {num_clients * min_samples_per_client})."
            )

    class_distribution = np.random.dirichlet([alpha] * num_clients, size=num_classes)

    client_idx_lists = [[] for _ in range(num_clients)]
    for ci, c in enumerate(classes):
        idx = torch.where(y == c)[0]
        idx = idx[torch.randperm(idx.numel())]

        counts = np.random.multinomial(int(idx.numel()), class_distribution[ci])

        start = 0
        for k in range(num_clients):
            cnt = int(counts[k])
            if cnt > 0:
                client_idx_lists[k].append(idx[start : start + cnt])
            start += cnt

    client_indices = []
    for k in range(num_clients):
        if len(client_idx_lists[k]) == 0:
            client_indices.append(torch.empty((0,), dtype=torch.long))
        else:
            client_indices.append(torch.cat(client_idx_lists[k], dim=0))

    if min_samples_per_client is not None:
        client_indices_py = [ci.cpu().tolist() for ci in client_indices]
        sizes = [len(ci) for ci in client_indices_py]

        def donor_candidate():
            donor = int(np.argmax(sizes))
            if sizes[donor] <= min_samples_per_client:
                return None
            return donor

        while True:
            smallest = int(np.argmin(sizes))
            if sizes[smallest] >= min_samples_per_client:
                break

            donor = donor_candidate()
            if donor is None:
                raise RuntimeError(
                    "Could not repair split to satisfy min_samples_per_client. "
                    f"Sizes: {sizes}, min={min_samples_per_client}"
                )

            j = np.random.randint(0, sizes[donor])
            moved = client_indices_py[donor].pop(j)
            client_indices_py[smallest].append(moved)
            sizes[donor] -= 1
            sizes[smallest] += 1

        client_indices = [
            torch.tensor(ci, dtype=torch.long, device=X.device) for ci in client_indices_py
        ]

    X_train_split = [X[ci] for ci in client_indices]
    y_train_split = [
        torch.zeros((x.shape[0],), dtype=y.dtype, device=y.device) for x in X_train_split
    ]
    y_true_split = [y[ci] for ci in client_indices]

    train_datasets = []
    for xk, yk in zip(X_train_split, y_train_split):
        perm = torch.randperm(xk.size(0), device=xk.device)
        train_datasets.append(TensorDataset(xk[perm], yk[perm]))

    train_loaders = [
        DataLoader(train_datasets[i], batch_size=batch_size, shuffle=True, pin_memory=True)
        for i in range(num_clients)
    ]

    X_test = torch.cat(X_train_split, dim=0)
    y_test = torch.cat(y_true_split, dim=0)
    test_dataset = TensorDataset(X_test, y_test)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, pin_memory=True)

    all_data = (X, y)
    train_data = (X_train_split, y_train_split)

    return train_loaders, train_datasets, test_loader, test_dataset, all_data, train_data
