import torch 
import numpy as np
from sklearn import metrics
from utils import attention_knn
from sklearn.cluster import KMeans
from torch.utils.data import DataLoader
from scipy.optimize import linear_sum_assignment


def valid(args, model, dataset, view, data_size, class_num):
    """
    Perform validation with clustering evaluation on learned representations
    
    Parameters
    ----------
    args: argparse.Namespace
        Configuration parameters containing batch_size etc.
    model: torch.nn.Module
        Trained neural network model for feature extraction
    dataset: torch.utils.data.Dataset
        Validation dataset containing samples
    view: int
        Identifier for data modality/view (used in multi-view learning)
    data_size: tuple
        Dimensionality of input data samples
    class_num: int
        Number of target clusters/classes
    
    Returns
    -------
    tuple
        Clustering metrics including (NMI, ARI, ACC, PUR, F-score, Precision, Recall)
    """
    # Disable gradient computation
    with torch.no_grad():
        # Create non-shuffled dataloader for ordered inference
        test_loader = DataLoader(dataset, args.batch_size, shuffle=False)
        
        # Extract features and labels from trained model
        labels_vector, hS = inference(args, test_loader, model, view, data_size)
    

    print('---------train over---------')
    print('Clustering results:')
    # Cluster features using K-means
    kmeans = KMeans(n_clusters=class_num, n_init=100)   # Multiple initializations for stability
    y_pred = kmeans.fit_predict(hS)                     # Get cluster assignments

    # Compute evaluation metrics
    nmi, ari, acc, pur, f, precision, recall = evaluate(labels_vector, y_pred)

    # Print formatted results
    print('ACC = {:.4f} NMI = {:.4f} PUR = {:.4f} ARI = {:.4f} F-Score = {:.4f} Precision = {:.4f} Recall = {:.4f}'
          .format(acc, nmi, pur, ari, f, precision, recall))
    return nmi, ari, acc, pur, f, precision, recall 

def cluster_acc(y_true, y_pred):
    """
    Compute clustering accuracy through optimal label alignment using Hungarian algorithm
    
    Parameters
    ----------
    y_true: array-like
        Ground truth cluster labels, shape [n_samples]
    y_pred: array-like
        Predicted cluster labels, shape [n_samples]
        
    Returns
    -------
    float
        Best matching accuracy between predicted and true labels
    """
    # Convert labels to integer type for indexing
    y_true = y_true.astype(np.int64)
    assert y_pred.size == y_true.size  # Validate matching dimensions

    # Create confusion matrix dimensions (covers all possible labels)
    D = max(y_pred.max(), y_true.max()) + 1  # +1 for zero-based indexing

    # Build confusion matrix where w[i,j] counts samples in pred i and true j
    w = np.zeros((D, D), dtype=np.int64)
    for i in range(y_pred.size):
        w[y_pred[i], y_true[i]] += 1  # Populate co-occurrence counts

    # Find optimal label permutation using Hungarian algorithm
    cost_matrix = w.max() - w  # Convert counts to minimization problem
    row_ind, col_ind = linear_sum_assignment(cost_matrix)  # Optimal assignment

    # Extract matched index pairs
    ind = np.concatenate([row_ind.reshape(-1,1), col_ind.reshape(-1,1)], axis=1)
    
    # Calculate accuracy: sum matched pairs / total samples
    return sum(w[i,j] for i,j in ind) / y_pred.size

def purity(y_true, y_pred):
    """
    Compute clustering purity by assigning each cluster to its most frequent class

    Parameters
    ----------
    y_true: array-like of shape (n_samples,)
        Ground truth class labels (can be non-consecutive integers)
    y_pred: array-like of shape (n_samples,)
        Cluster assignments obtained from clustering algorithm

    Returns
    -------
    float
        Purity score ranging from 0 to 1, where higher values indicate better clustering quality
    """
    # Initialize array for storing voted labels
    y_voted_labels = np.zeros(y_true.shape)
    
    # Remap ground truth labels to consecutive integers
    original_labels = np.unique(y_true)
    ordered_labels = np.arange(len(original_labels))
    for k in range(len(original_labels)):
        y_true[y_true == original_labels[k]] = ordered_labels[k]
    
    # Create histogram bins covering all possible labels
    bins = np.concatenate([ordered_labels, [ordered_labels[-1] + 1]])

    # Assign clusters to dominant class
    for cluster in np.unique(y_pred):
        # Calculate class distribution within current cluster
        class_distribution, _ = np.histogram(y_true[y_pred == cluster], bins=bins)
        # Identify dominant class with maximum count
        dominant_class = np.argmax(class_distribution)
        # Label entire cluster with dominant class
        y_voted_labels[y_pred == cluster] = dominant_class

    # Calculate purity as alignment accuracy
    return metrics.accuracy_score(y_true, y_voted_labels)

def evaluate(label, pred):
    """
    Compute multiple clustering evaluation metrics between true labels and predictions

    Parameters
    ----------
    label : array-like of shape (n_samples,)
        Ground truth class labels
    pred : array-like of shape (n_samples,)
        Predicted cluster assignments

    Returns
    -------
    tuple
        Metric scores in order:
        nmi   : Normalized Mutual Information
        ari   : Adjusted Rand Index
        acc   : Clustering Accuracy (requires label alignment)
        pur   : Clustering Purity
        f     : Fowlkes-Mallows Index
        precision : Macro-averaged Precision
        recall   : Macro-averaged Recall
    """
    # Calculate normalized mutual information
    nmi = metrics.normalized_mutual_info_score(label, pred)
    # Compute adjusted Rand index
    ari = metrics.adjusted_rand_score(label, pred)
    # Compute clustering accuracy with label alignment
    acc = cluster_acc(label, pred)
    # Calculate clustering purity
    pur = purity(label, pred)
    # Compute Fowlkes-Mallows score
    f = metrics.fowlkes_mallows_score(label, pred)
    # Calculate macro-averaged precision
    precision = metrics.precision_score(label, pred, average='macro')
    # Calculate macro-averaged recall
    recall = metrics.recall_score(label, pred, average='macro')

    return nmi, ari, acc, pur, f, precision, recall

def inference(args, loader, model, view, data_size):
    """
    Perform batch inference and collect feature representations

    Parameters
    ----------
    args : argparse.Namespace
        Configuration parameters with device information
    loader : DataLoader
        Data loader for input batches
    model : nn.Module
        Multi-view neural network model
    view : int
        Number of data views/modalities
    data_size : tuple
        Original data dimensions for reshaping

    Returns
    -------
    tuple
        labels_vector: Ground truth labels reshaped to data_size
        commonZ: Aggregated common representation matrix
    """
    model.eval()
    commonZ = []  # Stores fused common representations
    labels_vector = []  # Collects ground truth labels

    # Initialize feature containers
    ZS = [[] for _ in range(view)]  # View-specific features
    FS = [[] for _ in range(view * (view - 1) // 2)]  # Cross-view features

    for step, (xs, y, _) in enumerate(loader):
        # Move multi-view data to target device
        for v in range(view):
            xs[v] = xs[v].to(args.device)

        with torch.no_grad():
            # Forward pass through main model
            xrs, hs, zs = model(xs)
            
            # Feature fusion and attention processing
            commonz, fs, _, _ = model.fusion(xs)
            commonz = commonz.detach()
            commonz, s_z = attention_knn(commonz, commonz, commonz, args.knn)
            
            # Store common representation
            commonZ.extend(commonz.cpu().numpy())

            # Collect view-specific features
            for t in range(len(ZS)):
                ZS[t].extend(zs[t].cpu().numpy())

            # Collect cross-view interaction features 
            for t in range(len(fs)):
                FS[t].extend(fs[t].cpu().numpy())

        labels_vector.extend(y.numpy())

    # Reshape outputs to original data dimensions
    labels_vector = np.array(labels_vector).reshape(data_size)
    commonZ = np.array(commonZ)

    return labels_vector, commonZ