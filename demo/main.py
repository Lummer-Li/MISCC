import torch
import argparse
from models import MISCC
import torch.nn.functional as F
from dataloader import load_data
from sklearn.cluster import KMeans
from metrics import inference, valid
from torch.utils.data import DataLoader
from utils import setup_seed, attention_knn
from losses import multi_samples_contrastive_loss

def pre_train(args, epoch):
    """
    Perform pretraining of multi-view autoencoder through reconstruction loss
    
    Parameters
    ----------
    args : argparse.Namespace
        Configuration parameters containing device information
    epoch : int
        Current training epoch number
    
    Returns
    -------
    None (prints training loss)
    """
    tot_loss = 0.  # Accumulates total epoch loss
    mse = torch.nn.MSELoss()  # Reconstruction loss criterion
    
    for batch_idx, (xs, _, _) in enumerate(data_loader):
        # Move multi-view data to target device (GPU/CPU)
        for v in range(view):
            xs[v] = xs[v].to(args.device)
        
        optimizer.zero_grad()  # Clear previous gradients
        
        # Forward pass through autoencoder
        xrs, _, _ = model(xs)  # xrs: reconstructed views
        
        # Calculate view-specific reconstruction losses
        loss_list = []
        for v in range(view):
            loss_list.append(mse(xs[v], xrs[v]))  # Compare input with reconstruction
        
        # Aggregate multi-view loss
        loss = sum(loss_list)  # Sum losses from all views
        loss.backward()  # Backpropagate gradients
        optimizer.step()  # Update model parameters
        
        tot_loss += loss.item()  # Accumulate batch loss
    
    # Print epoch statistics
    print('Epoch {}'.format(epoch), 
          'Loss:{:.6f}'.format(tot_loss / len(data_loader)))


def fine_tune(args, epoch):
    """
    Fine-tune model with combined reconstruction, contrastive, and distribution alignment losses
    
    Parameters
    ----------
    args : argparse.Namespace
        Configuration parameters containing:
        - device: Computation device
        - knn: Number of nearest neighbors for attention
        - vaild_epochs: Interval for clustering validation
    epoch : int
        Current training epoch

    Global Variables
    ----------------
    predicts_common : np.ndarray
        Stores cluster assignments for contrastive learning
    """
    model.train()
    tot_loss = 0.  # Accumulated training loss
    mse = torch.nn.MSELoss()  # Reconstruction loss criterion
    global predicts_common  # Cluster predictions for contrastive alignment

    for batch_idx, (xs, y, _) in enumerate(data_loader):
        # Transfer multi-view data to target device
        for v in range(view):
            xs[v] = xs[v].to(args.device)
        
        optimizer.zero_grad()  # Reset gradients
        
        # Forward pass through network
        xrs, hs, zs = model(xs)  # xrs: reconstructions, hs/zs: intermediate features
        commonz, fs, new_zs, ss = model.fusion(xs)  # Fused representations
        
        # Attention-enhanced representation
        n_z, s_z = attention_knn(commonz, commonz, commonz, args.knn)  # n_z: refined features
        
        # Periodic clustering for pseudo-labels
        if epoch % args.vaild_epochs == 0:
            kmeans = KMeans(n_clusters=class_num, n_init=10, random_state=0)
            predicts_common = kmeans.fit_predict(n_z.detach().cpu().numpy())

        # Multi-task loss calculation
        loss_list = []
        for i in range(view):
            # View reconstruction loss
            loss_list.append(mse(xs[i], xrs[i]) * 1)  # Reconstruction weight=1.0
            
            # Cross-modal contrastive alignment
            cca_loss = multi_samples_contrastive_loss(
                new_zs[i], n_z, 
                torch.from_numpy(predicts_common), 
                torch.from_numpy(predicts_common)
            )
            loss_list.append(cca_loss * 0.1)  # Contrastive weight=0.1
            
            # Symmetric KL divergence for distribution alignment
            kl_loss = (F.kl_div(torch.log(ss[i]), s_z, reduction='batchmean') + 
                      F.kl_div(torch.log(s_z), ss[i], reduction='batchmean')) / 2.0
            loss_list.append(kl_loss * 1)  # KL weight=1.0

        # Optimization step
        loss = sum(loss_list)  # Combine losses
        loss.backward()  # Backpropagate
        optimizer.step()  # Update parameters
        tot_loss += loss.item()  # Accumulate loss

    # Epoch statistics
    print('Epoch {}'.format(epoch), 
          'Loss:{:.6f}'.format(tot_loss/len(data_loader)))

if __name__ == '__main__':
    """
    Deep Multi-view Clustering with Intra-view Similarity and Cross-view Correlation Learning (MISCC)
    Main execution pipeline for multi-view clustering training
    
    Steps:
    1. Parameter configuration
    2. Dataset preparation
    3. Model initialization
    4. Pretraining & Fine-tuning
    5. Final evaluation
    """
    # Argument parsing for training configuration
    parser = argparse.ArgumentParser()
    # Experiment setup
    parser.add_argument('--seed', type=int, default=10,
                      help='Random seed for reproducibility')
    parser.add_argument('--dataset', type=str, default='BBCSport',
                      help='Dataset name from available options')
    parser.add_argument('--batch_size', type=int, default=256,
                      help='Number of samples per training batch')
    
    # Optimization parameters
    parser.add_argument('--lr', type=float, default=1e-4,
                      help='Initial learning rate')
    parser.add_argument('--momentum', type=float, default=0,
                      help='Momentum factor (not used in Adam)')
    parser.add_argument('--weight_decay', type=float, default=0,
                      help='Weight decay (L2 penalty)')
    
    # Model architecture
    parser.add_argument('--knn', type=int, default=10,
                      help='Number of nearest neighbors for attention')
    parser.add_argument('--low_feature_dim', type=int, default=1024,
                      help='Dimension of encoder hidden layer')
    parser.add_argument('--high_feature_dim', type=int, default=1024,
                      help='Dimension of final features')
    
    # Training schedule
    parser.add_argument('--rec_epochs', type=int, default=100,
                      help='Pretraining epochs for reconstruction')
    parser.add_argument('--fine_epochs', type=int, default=1000,
                      help='Fine-tuning epochs for joint training')
    parser.add_argument('--vaild_epochs', type=int, default=100,
                      help='Cluster validation interval during fine-tuning')
    
    # Advanced configurations
    parser.add_argument('--temperature', type=float, default=0.5,
                      help='Temperature parameter for contrastive loss')
    parser.add_argument('--device', type=str, default='cuda:0',
                      help='Computation device: cuda:id or cpu')
    parser.add_argument('--save_flag', type=bool, default=False,
                      help='Flag for model checkpoint saving')
    
    args = parser.parse_args()

    # Environment setup
    setup_seed(args.seed)  # Fix random seeds for reproducibility
    dataset, dims, view, data_size, class_num = load_data(args.dataset)  # dims: input dimensions per view
    
    # Create shuffled data loader with full batch utilization
    data_loader = DataLoader(
        dataset, 
        batch_size=args.batch_size, 
        shuffle=True, 
        drop_last=True  # Discard incomplete batches
    )

    # Print configuration header
    print("""================================================================================================================""")
    print(args)  # Display all parameters
    print("""================================================================================================================""")
    
    # Model initialization
    model = MISCC(  # Deep Multi-view Clustering with Intra-view Similarity and Cross-view Correlation Learning (MISCC)
        args=args,
        view=view, 
        dims=dims, 
    ).to(args.device)
    
    # Configure Adam optimizer
    optimizer = torch.optim.Adam(
        model.parameters(), 
        lr=args.lr, 
        weight_decay=args.weight_decay
    )

    # Training pipeline
    # Phase 1: Reconstruction pretraining
    for epoch in range(args.rec_epochs):
        pre_train(args, epoch)  # Note: Argument order corrected vs original code
        
    # Phase 2: Joint fine-tuning
    for epoch in range(args.fine_epochs):
        fine_tune(args, epoch)  # Includes contrastive learning and distribution alignment
    
    # Final evaluation
    nmi, ari, acc, pur, f, precision, recall = valid(
        args, model, dataset, view, data_size, class_num
    )