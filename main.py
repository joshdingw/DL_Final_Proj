from dataset import create_wall_dataloader
from evaluator import ProbingEvaluator
import torch
#from models import MockModel
from models import JEPAModel
import glob
from tqdm import tqdm
import torch.nn.functional as F
def get_device():
    """Check for GPU availability."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)
    return device


def load_data(device):
    data_path = "/scratch/DL24FA"

    probe_train_ds = create_wall_dataloader(
        data_path=f"{data_path}/probe_normal/train",
        probing=True,
        device=device,
        train=True,
    )

    probe_val_normal_ds = create_wall_dataloader(
        data_path=f"{data_path}/probe_normal/val",
        probing=True,
        device=device,
        train=False,
    )

    probe_val_wall_ds = create_wall_dataloader(
        data_path=f"{data_path}/probe_wall/val",
        probing=True,
        device=device,
        train=False,
    )

    probe_val_ds = {"normal": probe_val_normal_ds, "wall": probe_val_wall_ds}

    return probe_train_ds, probe_val_ds

def load_model():
    """Load or initialize the model."""
    # TODO: Replace MockModel with your trained model
    #model = MockModel()
    model = JEPAModel(device=device)
    model.to(device)
    try:
        model.load_state_dict(torch.load('jepa_model.pth'))
        print("Loaded saved JEPA model.")
    except FileNotFoundError:
        print("No saved model found, initializing a new model.")
    return model


def evaluate_model(device, model, probe_train_ds, probe_val_ds):
    evaluator = ProbingEvaluator(
        device=device,
        model=model,
        probe_train_ds=probe_train_ds,
        probe_val_ds=probe_val_ds,
        quick_debug=False,
    )

    prober = evaluator.train_pred_prober()

    avg_losses = evaluator.evaluate_all(prober=prober)

    for probe_attr, loss in avg_losses.items():
        print(f"{probe_attr} loss: {loss}")

def je_loss(predictions, targets):
    # Input shapes: [B, T, D]
    
    # Normalize embeddings to unit sphere (crucial!)
    predictions = F.normalize(predictions, dim=-1, p=2)  
    targets = F.normalize(targets, dim=-1, p=2)
    
    # Simple contrastive loss that encourages similar embeddings for same timesteps
    pos_cos_sim = (predictions * targets).sum(dim=-1)  # [B, T]
    
    # Scale up loss for better gradients
    loss = 2 * (1 - pos_cos_sim).mean()
    
    # Add regularization to prevent representation collapse
    pred_std = predictions.std(dim=1).mean()
    target_std = targets.std(dim=1).mean()
    
    std_reg = 0.1 * (torch.abs(1 - pred_std) + torch.abs(1 - target_std))
    
    return loss + std_reg

def train_model(device):
    train_loader = create_wall_dataloader(
        data_path="/scratch/DL24FA/train",
        probing=False, 
        device=device,
        train=True,
    )

    model = JEPAModel(device=device, momentum=0.996)
    model.to(device)

    # Initialize encoder randomly
    def init_weights(m):
        if isinstance(m, nn.Linear) or isinstance(m, nn.Conv2d):
            torch.nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                torch.nn.init.zeros_(m.bias)
    
    model.apply(init_weights)

    optimizer = torch.optim.AdamW([
        {'params': model.encoder.parameters(), 'lr': 3e-4},
        {'params': model.predictor.parameters(), 'lr': 1e-3}
    ], weight_decay=0.01)

    num_epochs = 10
    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        
        for batch_idx, batch in enumerate(tqdm(train_loader, desc=f"Training Epoch {epoch+1}")):
            states = batch.states
            actions = batch.actions

            # Zero gradients
            optimizer.zero_grad()

            # Forward pass
            with torch.cuda.amp.autocast():  # Mixed precision
                predictions = model(states, actions)
                with torch.no_grad():
                    targets = model.target_encoder(
                        states.view(-1, *states.shape[2:])
                    ).view(states.size(0), states.size(1), -1)
                    targets = targets.detach()

                # Compute loss
                loss = je_loss(predictions[:, 1:], targets[:, 1:])

            # Backward pass
            loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(model.encoder.parameters(), 1.0)
            torch.nn.utils.clip_grad_norm_(model.predictor.parameters(), 1.0)
            
            optimizer.step()
            
            # Update target network with momentum
            model.update_target_encoder()

            total_loss += loss.item()

            if batch_idx % 100 == 0:
                print(f"\nBatch {batch_idx}")
                print(f"Loss: {loss.item():.4f}")
                # Monitor embedding norms
                with torch.no_grad():
                    pred_norm = predictions.norm(dim=-1).mean().item()
                    target_norm = targets.norm(dim=-1).mean().item()
                    print(f"Pred norm: {pred_norm:.4f}, Target norm: {target_norm:.4f}")

        avg_loss = total_loss / len(train_loader)
        print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {avg_loss:.4f}")

    torch.save(model.state_dict(), 'jepa_model.pth')


if __name__ == "__main__":
    device = get_device()
    train_model(device)

    probe_train_ds, probe_val_ds = load_data(device)
    model = load_model()
    evaluate_model(device, model, probe_train_ds, probe_val_ds)
