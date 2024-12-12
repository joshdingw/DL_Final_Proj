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
    # Normalize embeddings to unit sphere
    predictions = F.normalize(predictions, dim=-1, p=2)
    targets = F.normalize(targets, dim=-1, p=2)
    
    # Reshape predictions and targets to 2D
    batch_size = predictions.size(0)
    seq_len = predictions.size(1)
    feat_dim = predictions.size(2)
    
    predictions = predictions.view(-1, feat_dim)  # [B*T, D]
    targets = targets.view(-1, feat_dim)  # [B*T, D]
    
    # Compute positive similarity
    similarity = (predictions * targets).sum(dim=-1)  # [B*T]
    
    # InfoNCE-style loss with temperature
    temperature = 0.5
    exp_sim = torch.exp(similarity / temperature)  # [B*T]
    
    # Compute all pairs of similarities
    all_sims = torch.matmul(predictions, targets.t())  # [B*T, B*T]
    exp_all_sims = torch.exp(all_sims / temperature)  # [B*T, B*T]
    
    # Remove diagonal (positive) similarities from denominator
    mask = torch.eye(exp_all_sims.size(0), device=exp_all_sims.device)
    exp_all_sims = exp_all_sims * (1 - mask)
    
    # Sum over negatives
    neg_exp_sim = exp_all_sims.sum(dim=1)  # [B*T]
    
    # Final loss
    loss = -torch.log(exp_sim / (exp_sim + neg_exp_sim + 1e-6)).mean()
    
    return loss

def train_model(device):
    train_loader = create_wall_dataloader(
        data_path="/scratch/DL24FA/train",
        probing=False,
        device=device,
        train=True,
    )

    model = JEPAModel(device=device, momentum=0.99)
    model.to(device)

    optimizer = torch.optim.AdamW(
        list(model.encoder.parameters()) + list(model.predictor.parameters()),
        lr=5e-4,
        weight_decay=0.05
    )
    
    warmup_steps = 1000
    total_steps = 10 * len(train_loader)
    step = 0
    
    num_epochs = 10
    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        
        for batch_idx, batch in enumerate(tqdm(train_loader, desc=f"Training Epoch {epoch+1}")):
            states = batch.states
            actions = batch.actions

            # Warmup learning rate
            if step < warmup_steps:
                lr = optimizer.param_groups[0]['lr'] * step / warmup_steps
                for param_group in optimizer.param_groups:
                    param_group['lr'] = lr
            
            optimizer.zero_grad()

            predictions = model(states, actions)
            with torch.no_grad():
                targets = model.target_encoder(
                    states.view(-1, *states.shape[2:])
                ).view(states.size(0), states.size(1), -1)
                targets = targets.detach()

            # Only predict future states
            loss = je_loss(predictions[:, 1:], targets[:, 1:])

            loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            
            optimizer.step()
            
            # EMA update for target encoder
            model.update_target_encoder()

            total_loss += loss.item()
            step += 1

            if batch_idx % 100 == 0:
                print(f"Batch {batch_idx}, Loss: {loss.item():.4f}")
                print(f"Learning rate: {optimizer.param_groups[0]['lr']:.6f}")

        avg_loss = total_loss / len(train_loader)
        print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {avg_loss:.4f}")

    torch.save(model.state_dict(), 'jepa_model.pth')

    prober = evaluator.train_pred_prober()
    avg_losses = evaluator.evaluate_all(prober=prober)
    for probe_attr, loss in avg_losses.items():
        print(f"{probe_attr} loss: {loss}")


if __name__ == "__main__":
    device = get_device()
    train_model(device)

    probe_train_ds, probe_val_ds = load_data(device)
    model = load_model()
    evaluate_model(device, model, probe_train_ds, probe_val_ds)
