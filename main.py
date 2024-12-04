from dataset import create_wall_dataloader
from evaluator import ProbingEvaluator
import torch
#from models import MockModel
from models import JEPAModel
import glob
from tqdm import tqdm

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
    # Normalize the representations
    predictions = F.normalize(predictions, dim=-1)
    targets = F.normalize(targets, dim=-1)

    # Compute MSE loss
    loss = F.mse_loss(predictions, targets)
    return loss


def train_model(device):
    # Load training data
    train_loader = create_wall_dataloader(
        data_path="/scratch/DL24FA/train",
        probing=False,
        device=device,
        train=True,
    )

    model = JEPAModel(device=device)
    model.to(device)

    optimizer = torch.optim.Adam(
        list(model.encoder.parameters()) + list(model.predictor.parameters()),
        lr=1e-3
    )

    num_epochs = 10
    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        for batch_idx, batch in enumerate(tqdm(train_loader, desc=f"Training Epoch {epoch+1}")):
            states = batch.states  # [B, T, C, H, W]
            actions = batch.actions  # [B, T-1, action_dim]

            # Debugging statements
            if batch_idx == 0 and epoch == 0:
                print(f"States shape: {states.shape}, Actions shape: {actions.shape}")
                print(f"States min: {states.min()}, max: {states.max()}")
                print(f"Actions min: {actions.min()}, max: {actions.max()}")

            optimizer.zero_grad()
            predictions, targets = model(states, actions)

            # Additional debugging
            print(f"Predictions shape: {predictions.shape}, Targets shape: {targets.shape}")
            print(f"Predictions min: {predictions.min()}, max: {predictions.max()}")
            print(f"Targets min: {targets.min()}, max: {targets.max()}")

            loss = je_loss(predictions, targets)
            loss.backward()

            # Check gradient norms
            for name, param in model.named_parameters():
                if param.grad is not None:
                    print(f"{name}: grad norm = {param.grad.norm()}")
                else:
                    print(f"{name} has no gradient")

            optimizer.step()

            # Update target encoder
            model.update_target_encoder()

            total_loss += loss.item()

            # Print loss every 100 batches
            if batch_idx % 100 == 0:
                print(f"Batch {batch_idx}, Loss: {loss.item():.8f}")

        avg_loss = total_loss / len(train_loader)
        print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {avg_loss:.8f}")

        # Optionally evaluate the model
        if (epoch + 1) % 2 == 0:
            evaluate_current_model(model, device)

    # Save the trained model
    torch.save(model.state_dict(), 'jepa_model.pth')

def evaluate_current_model(model, device):
    # Load evaluation datasets
    probe_train_ds, probe_val_ds = load_data(device)
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

    # Save the trained model
    torch.save(model.state_dict(), 'jepa_model.pth')

if __name__ == "__main__":
    device = get_device()
    train_model(device)
    
    probe_train_ds, probe_val_ds = load_data(device)
    model = load_model()
    evaluate_model(device, model, probe_train_ds, probe_val_ds)
