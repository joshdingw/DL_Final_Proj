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
    
    # Stronger loss scaling and better regularization
    cos_sim = (predictions * targets).sum(dim=-1)
    loss = (1 - cos_sim).mean()  # Range [0, 2]
    
    # Add diversity loss to prevent collapse
    batch_cos_sim = torch.matmul(predictions, predictions.transpose(-2, -1))
    identity = torch.eye(batch_cos_sim.shape[-1], device=predictions.device)
    diversity_loss = (batch_cos_sim - identity).pow(2).mean()
    
    return loss + 0.1 * diversity_loss

def train_model(device):
    train_loader = create_wall_dataloader(
        data_path="/scratch/DL24FA/train",
        probing=False,
        device=device,
        train=True,
    )

    model = JEPAModel(device=device, momentum=0.996)  # Higher initial momentum
    model.to(device)

    optimizer = torch.optim.AdamW(  # Switch to AdamW
        list(model.encoder.parameters()) + list(model.predictor.parameters()),
        lr=1e-3,
        weight_decay=0.01  # Stronger weight decay
    )
    
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=10 * len(train_loader)
    )

    num_epochs = 10
    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        
        for batch_idx, batch in enumerate(tqdm(train_loader, desc=f"Training Epoch {epoch+1}")):
            states = batch.states
            actions = batch.actions

            optimizer.zero_grad()

            # Get predictions and targets
            predictions = model(states, actions)
            with torch.no_grad():
                targets = model.target_encoder(
                    states.view(-1, *states.shape[2:])
                ).view(states.size(0), states.size(1), -1)
                targets = targets.detach()

            # Compute loss only on future predictions
            loss = je_loss(predictions[:, 1:], targets[:, 1:])
            
            # Skip problematic batches
            if not torch.isfinite(loss) or loss.item() < 0.1:
                print(f"Skipping batch with loss {loss.item()}")
                continue

            loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            
            optimizer.step()
            scheduler.step()
            
            # Update target network
            model.update_target_encoder()

            total_loss += loss.item()

            if batch_idx % 100 == 0:
                print(f"Batch {batch_idx}, Loss: {loss.item():.4f}")
                print(f"Learning rate: {scheduler.get_last_lr()[0]:.6f}")

        avg_loss = total_loss / len(train_loader)
        print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {avg_loss:.4f}")

    torch.save(model.state_dict(), 'jepa_model.pth')

# def evaluate_current_model(model, device):
#     probe_train_ds, probe_val_ds = load_data(device)
#     evaluator = ProbingEvaluator(
#         device=device,
#         model=model,
#         probe_train_ds=probe_train_ds,
#         probe_val_ds=probe_val_ds,
#         quick_debug=False,
#     )

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
