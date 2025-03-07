import os
import math
import torch
import random
import numpy as np
import torch.nn.functional as F

def setup_seed(seed):
    """
    Set global random seeds for reproducible experiments
    
    Parameters
    ----------
    seed : int
        Random seed value to ensure reproducibility
    
    Effects
    -------
    - PyTorch RNG states (CPU/CUDA)
    - NumPy RNG state
    - Python built-in random module
    - CuDNN deterministic algorithms
    
    Notes
    -----
    - Setting cudnn.deterministic=True may impact performance
    - Does NOT set Hash seed for Python (PYTHONHASHSEED)
    """
    # PyTorch CPU/CUDA seeds
    torch.manual_seed(seed)          # Sets both CPU and CUDA (current device)
    torch.cuda.manual_seed_all(seed) # For multi-GPU setups
    
    # NumPy random state
    np.random.seed(seed)
    
    # Python built-in random
    random.seed(seed)
    
    # CuDNN configurations
    torch.backends.cudnn.deterministic = True  # Deterministic algorithms
    torch.backends.cudnn.benchmark = False     # Disable auto-tuner for reproducibility

def pearson_correlation(vector1, vector2):
    """
    Compute Pearson correlation coefficient between two vectors

    Parameters
    ----------
    vector1 : torch.Tensor
        First input vector (1D tensor)
    vector2 : torch.Tensor
        Second input vector (1D tensor), same length as vector1

    Returns
    -------
    float
        Pearson correlation coefficient between [-1, 1]
    """
    # Calculate means of both vectors
    mean1 = torch.mean(vector1)
    mean2 = torch.mean(vector2)
    
    # Compute deviations from means
    diff1 = vector1 - mean1  # Centered vector1
    diff2 = vector2 - mean2  # Centered vector2
    
    # Calculate covariance term
    numerator = torch.sum(diff1 * diff2)  # Cross-product sum
    
    # Compute standard deviations product
    denominator = torch.sqrt(torch.sum(diff1 ** 2) * torch.sum(diff2 ** 2))  # Geometric mean of variances
    
    # Final correlation calculation
    correlation = numerator / denominator  # Normalized covariance
    
    return correlation.item()  # Convert scalar tensor to Python float


def compute_similarity_matrix(views):
    """
    Compute symmetric similarity matrix between multiple views using Pearson correlation

    Parameters
    ----------
    views : list of torch.Tensor
        List containing feature matrices for each view, each of shape [batch_size, feat_dim]

    Returns
    -------
    torch.Tensor of shape [num_views, num_views]
        Symmetric matrix where element (i,j) contains Pearson correlation between view_i and view_j
    """
    num_views = len(views)
    similarity_matrix = torch.zeros(num_views, num_views)  # Initialize matrix

    # Calculate pairwise upper triangular elements
    for i in range(num_views):
        for j in range(i+1, num_views):  # Skip diagonal and lower triangle
            # Reshape to vectors for correlation calculation
            view_i = views[i].view(1, -1).float()  # Flatten to [1, N*D]
            view_j = views[j].view(1, -1).float()
            
            # Compute bidirectional correlations (redundant but explicit)
            similarity_i = pearson_correlation(view_i, view_j)
            similarity_j = pearson_correlation(view_j, view_i)  # Should equal similarity_i
            
            # Populate symmetric positions
            similarity_matrix[i][j] = similarity_i
            similarity_matrix[j][i] = similarity_j  # Mirror value

    return similarity_matrix

def attention_knn(Q, K, V, top_k=10, mask=None, dropout=None):
    """
    Sparse attention mechanism with k-nearest neighbors selection

    Parameters
    ----------
    Q : torch.Tensor
        Query tensor of shape [batch_size, seq_len, dim]
    K : torch.Tensor
        Key tensor of shape [batch_size, seq_len, dim]
    V : torch.Tensor
        Value tensor of shape [batch_size, seq_len, dim]
    top_k : int, optional
        Number of nearest neighbors to retain, by default 10
    mask : torch.Tensor, optional
        Binary mask tensor of shape [seq_len, seq_len], by default None
    dropout : nn.Module, optional
        Dropout layer for attention weights, by default None

    Returns
    -------
    tuple
        - Output tensor of shape [batch_size, seq_len, dim]
        - Attention score tensor of shape [batch_size, seq_len, seq_len]
    """
    # Get dimension size for scaling
    dims = Q.size(-1)
    
    # Compute scaled dot-product attention scores
    scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(dims)
    
    # Apply optional attention mask
    if mask is not None:
        scores = scores.masked_fill(mask == 0, -1e9)
    
    # Normalize with softmax
    scores = F.softmax(scores, dim=-1)
    
    # Apply dropout regularization
    if dropout is not None:
        scores = dropout(scores)
    
    # Select top-k attention weights
    values, indices = torch.topk(scores, top_k, dim=-1)
    
    # Create sparse attention matrix
    sparse_attn = torch.zeros_like(scores)
    sparse_attn.scatter_(-1, indices, values)
    
    # Compute context vectors
    context = torch.matmul(sparse_attn, V)
    
    return context, scores

def save_model(state, dataset_name):
    """
    Save model state dictionary to specified path with dataset-based naming
    
    Parameters
    ----------
    state : dict
        Model state dictionary containing:
        - 'model_state_dict': Model parameters
        - 'optimizer_state_dict': Optimizer state (optional)
        - 'epoch': Training epoch (optional)
    dataset_name : str
        Identifier for model versioning, used in filename
    
    Raises
    ------
    PermissionError
        If lacking write permissions for model directory
    IOError
        If disk space insufficient or path invalid
    
    Notes
    -----
    - Automatically creates './models' directory if non-existent
    - Uses PyTorch's serialization format (.pth)
    - Recommended to include training metadata in state
    """
    # Create model directory if not exists
    if not os.path.exists('./models'):
        os.makedirs('./models')  # Recursive directory creation
    
    # Construct filesystem path
    save_path = os.path.join('./models', f'{dataset_name}.pth')
    
    # Serialize model state
    torch.save(state, save_path)  # Uses pickle protocol
    
    # User feedback
    print(f'Model checkpoint saved at: {save_path}')