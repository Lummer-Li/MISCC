import torch
import torch.nn as nn
import torch.nn.functional as F
from utils import attention_knn, compute_similarity_matrix


class Encoder(nn.Module):
    """
    Deep neural network for feature encoding with progressive dimension transformation
    
    Parameters
    ----------
    input_dim : int
        Dimension of input features
    feature_dim : int 
        Dimension of output latent features
    
    Forward Pass
    ------------
    x : torch.Tensor of shape [batch_size, input_dim]
        Input data matrix
    
    Returns
    -------
    torch.Tensor of shape [batch_size, feature_dim]
        Encoded low-dimensional representations
    """
    def __init__(self, input_dim, feature_dim):
        super(Encoder, self).__init__()
        self.encoder = nn.Sequential(
            # First compression layer
            nn.Linear(input_dim, 500),
            nn.ReLU(inplace=True),  # In-place operation saves memory
            
            # Feature refinement layer
            nn.Linear(500, 500),  # Maintain dimension for feature processing
            nn.ReLU(inplace=True),
            
            # Dimension expansion layer
            nn.Linear(500, 2000),  # Expand features for richer representations
            nn.ReLU(inplace=True),
            
            # Final compression to target dimension
            nn.Linear(2000, feature_dim),  # Bottleneck layer
        )

    def forward(self, x):
        """Performs forward propagation through the encoding layers"""
        return self.encoder(x)
    
class Decoder(nn.Module):
    """
    Symmetric deep decoder for input reconstruction from latent features
    
    Parameters
    ----------
    input_dim : int
        Dimension of original input space
    feature_dim : int
        Dimension of latent features from encoder
    
    Forward Pass
    ------------
    x : torch.Tensor of shape [batch_size, feature_dim]
        Latent representations from encoder
    
    Returns
    -------
    torch.Tensor of shape [batch_size, input_dim]
        Reconstructed input data
    """
    def __init__(self, input_dim, feature_dim):
        super(Decoder, self).__init__()
        self.decoder = nn.Sequential(
            # Initial expansion from latent space
            nn.Linear(feature_dim, 2000),
            nn.ReLU(),
            
            # Intermediate feature processing
            nn.Linear(2000, 500),  # First compression stage
            nn.ReLU(),
            
            # Feature refinement layer
            nn.Linear(500, 500),  # Maintain dimension for detail reconstruction
            nn.ReLU(),
            
            # Final reconstruction to original dimension
            nn.Linear(500, input_dim)  # Linear activation for real-valued output
        )

    def forward(self, x):
        """Performs progressive upsampling from latent features"""
        return self.decoder(x)

class MISCC(nn.Module):
    """
    Deep Multi-view Clustering with Intra-view Similarity and Cross-view Correlation Learning (MISCC)
    
    Parameters
    ----------
    args : argparse.Namespace
        Configuration parameters containing:
        - low_feature_dim: Dimension of encoder outputs
        - high_feature_dim: Dimension of fused features
        - knn: Number of nearest neighbors for attention
    view : int
        Number of data views/modalities
    dims : list
        Input dimensions for each view
    
    Attributes
    ----------
    encoders : nn.ModuleList
        View-specific encoder networks
    decoders : nn.ModuleList
        View-specific decoder networks
    """
    def __init__(self, args, view, dims):
        super(MISCC, self).__init__()
        self.encoders = []
        self.decoders = []
        self.view = view
        self.hidden_dim = 4096  # Internal projection dimension
        self.knn = args.knn  # k-NN parameter for attention

        # Initialize view-specific encoder/decoder pairs
        for v in range(self.view):
            self.encoders.append(Encoder(dims[v], args.low_feature_dim).to(args.device))
            self.decoders.append(Decoder(dims[v], args.low_feature_dim).to(args.device))
        self.encoders = nn.ModuleList(self.encoders)
        self.decoders = nn.ModuleList(self.decoders)

        # Feature projection heads
        self.projector_head = nn.Sequential(
            nn.Linear(args.low_feature_dim, self.hidden_dim),
            nn.BatchNorm1d(self.hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(self.hidden_dim, args.high_feature_dim)
        )
        
        # Cross-view fusion components
        f_view = view * (view - 1) // 2  # Number of view pairs
        self.predictor_head = nn.Sequential(
            nn.Linear(args.high_feature_dim * f_view, self.hidden_dim),
            nn.BatchNorm1d(self.hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(self.hidden_dim, args.high_feature_dim)
        )
        self.merge_head = nn.Sequential(
            nn.Linear(args.high_feature_dim * 2, self.hidden_dim),
            nn.BatchNorm1d(self.hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(self.hidden_dim, args.high_feature_dim)
        )

    def forward(self, xs):
        """
        Multi-view autoencoding forward pass
        
        Parameters
        ----------
        xs : list of torch.Tensor
            Input features for each view [view_count x (batch_size, feat_dim)]
            
        Returns
        -------
        tuple
            xrs: Reconstructed inputs for each view
            hs: Encoder hidden states
            zs: Projected features for contrastive learning
        """
        xrs, zs, hs = [], [], []
        for v in range(self.view):
            h = self.encoders[v](xs[v])  # View-specific encoding
            z = F.normalize(self.projector_head(h), dim=1)  # L2-normalized projection
            xr = self.decoders[v](h)  # Input reconstruction
            hs.append(h)
            zs.append(z)
            xrs.append(xr)
        return xrs, hs, zs
    
    def fusion(self, xs):
        """
        Cross-view feature fusion with attention
        
        Parameters
        ----------
        xs : list of torch.Tensor
            Input features for each view
            
        Returns
        -------
        tuple
            commonz: Fused consensus representation
            fs: Pairwise fused features
            zs: Projected view features
            ss: Attention weights from kNN
        """
        zs, ss = [], []
        # View-specific feature projection
        for v in range(self.view):
            h = self.encoders[v](xs[v])
            z = F.normalize(self.projector_head(h), dim=1)
            z, s = attention_knn(z, z, z, self.knn)  # Attention-weighted features
            zs.append(z)
            ss.append(s)

        # Cross-view fusion
        fs = []
        similarity_matrix = compute_similarity_matrix(zs)  # [view x view] affinity matrix
        
        # Process all unique view pairs
        for i in range(self.view):
            for j in range(i+1, self.view):
                # Weighted feature concatenation
                weighted_feature = similarity_matrix[i][j] * torch.cat([zs[i], zs[j]], dim=1)
                f = self.merge_head(weighted_feature)
                fs.append(f)
        
        # Consensus representation
        commonz = torch.cat(fs, dim=1)
        commonz = F.normalize(self.predictor_head(commonz), dim=1)
        
        return commonz, fs, zs, ss