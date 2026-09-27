import torch
import matplotlib.pyplot as plt
from utils.utils_plots import plot_pca_reduction

def train_model_prealocated(model,
                            dataset,
                            index,
                            batch_size,
                            round = 0,
                            debug_flag = 0):

    model.toggle_evolving(True)
    model.train()
    model.set_debugging_flag(debug_flag)

    X_buf, y_buf = dataset.tensors

    for batch_idx, start in enumerate(range(0, X_buf.size(0), batch_size )):
        data   = X_buf[start : start + batch_size]
        labels = y_buf[start : start + batch_size]
        
        data = data.to(model.device, non_blocking=True)
        labels = labels.to(model.device, non_blocking=True)
   
        model.batch_clustering(data, labels)
           
        if debug_flag:
            print(f"Batch size {data.shape[0]}({batch_size})/{X_buf.shape[0]}")
            print(f"Number of clusters is {model.c}")
            print(f"Covariance size {torch.sum(model.S_0)}")
            
        if debug_flag in (8, 9):
            fig = plot_pca_reduction((data, labels), model=model,
                                     num_sigma=2, N_max=model.kappa_n)
            plt.show()
      

        if batch_idx>1:
            model.recluster_outliers()

        if debug_flag:
            print(f"Number of clusters after reclustering is {model.c}")

            

        if debug_flag in (8, 9):
            fig = plot_pca_reduction((data, labels), model=model,
                                     num_sigma=2, N_max=model.kappa_n)
            if debug_flag == 9:
                plt.show()
            else:
                base = f".Images/Train/Batch/Client_{index}"
                fig.savefig(f"{base}/round_{round}_batch_{batch_idx}.svg",
                            bbox_inches='tight')
                fig.savefig(f"{base}/round_{round}_batch_{batch_idx}.png",
                            bbox_inches='tight')
                fig.savefig(f"{base}/round_{round}_batch_{batch_idx}.pdf",
                            bbox_inches='tight')

    return model

def train_model_loader(model, data_loader, index, round=0, debug_flag=0):
    model.toggle_evolving(True)
    model.train()
    model.set_debugging_flag(debug_flag)
    x_list = []
    y_list = []

    for batch, (data, labels) in enumerate(data_loader):
        
        data = data.to(model.device, non_blocking=True)
        labels = labels.to(model.device, non_blocking=True)
        model.batch_clustering(data, labels)
        model.recluster_outliers()
        x_list.append(data.detach().cpu())
        y_list.append(labels.detach().cpu())

        if debug_flag in [8, 9]:
            fig = plot_pca_reduction((data, labels), model=model, num_sigma=2, N_max=model.kappa_n)
            if debug_flag == 9:
                plt.show()
            else:
                fig.savefig(f".Images/Train/Batch/Client_{index}/round_{round}_batch_{batch}.svg", bbox_inches='tight')
                fig.savefig(f".Images/Train/Batch/Client_{index}/round_{round}_batch_{batch}.png", bbox_inches='tight')
                fig.savefig(f".Images/Train/Batch/Client_{index}/round_{round}_batch_{batch}.pdf", bbox_inches='tight')
    
    

    if debug_flag in [16]:
        fig = plot_pca_reduction((data, labels), model=model, num_sigma=2, N_max=model.kappa_n)
        plt.show()

    return model

def train_model_sequentially_loader(models, data_loaders, round=0, debug_flag=0):
    trained_models = []
    for index, (model, data_loader) in enumerate(zip(models, data_loaders)):
        trained_model = train_model_loader(model, data_loader, index, round, debug_flag)
        trained_model.recluster_outliers()
        trained_models.append(trained_model)

    return trained_models

def train_model_sequentially_prealocated(models,
                                         train_datasets, 
                                         batch_size,
                                         round= 0,
                                         debug_flag= 0):

    trained_models = []
    for idx, (model, ds) in enumerate(zip(models, train_datasets)):
        trained = train_model_prealocated(model, ds, idx, batch_size, round, debug_flag)
        trained.recluster_outliers()
        trained_models.append(trained)
    return trained_models
