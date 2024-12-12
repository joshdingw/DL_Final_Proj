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
    
    # BYOL loss: negative cosine similarity
    # Scale up the loss to prevent extremely small values
    loss = (2 - 2 * (predictions * targets).sum(dim=-1)).mean()
    
    # Add L2 regularization to prevent collapse
    l2_loss = 0.01 * (predictions.pow(2).sum(dim=-1).mean() + targets.pow(2).sum(dim=-1).mean())
    
    return loss + l2_loss

def train_model(device):
    train_loader = create_wall_dataloader(
        data_path="/scratch/DL24FA/train",
        probing=False,
        device=device,
        train=True,
    )

    model = JEPAModel(device=device, momentum=0.99)  # Start with slightly lower momentum
    model.to(device)

    # Separate optimizers for encoder and predictor with different learning rates
    encoder_optimizer = torch.optim.Adam(model.encoder.parameters(), lr=3e-4)
    predictor_optimizer = torch.optim.Adam(model.predictor.parameters(), lr=1e-3)

    num_epochs = 20  # Increase epochs
    
    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        for batch_idx, batch in enumerate(tqdm(train_loader, desc=f"Training Epoch {epoch+1}")):
            states = batch.states
            actions = batch.actions

            encoder_optimizer.zero_grad()
            predictor_optimizer.zero_grad()

            # Forward pass
            predictions = model(states, actions)
            
            # Get target representations
            with torch.no_grad():
                targets = model.target_encoder(
                    states.view(-1, *states.shape[2:])
                ).view(states.size(0), states.size(1), -1)

            # Compute loss only on predicted futures
            loss = je_loss(predictions[:, 1:], targets[:, 1:])

            # Stability check with reasonable threshold
            if loss.item() < 0.1:  
                print(f"Warning: Low loss value detected: {loss.item()}")
                # But continue training - don't skip batch
            
            loss.backward()

            # Gradient clipping per optimizer
            torch.nn.utils.clip_grad_norm_(model.encoder.parameters(), 1.0)
            torch.nn.utils.clip_grad_norm_(model.predictor.parameters(), 1.0)
            
            encoder_optimizer.step()
            predictor_optimizer.step()

            # Exponential moving average update of target encoder
            model.update_target_encoder()

            total_loss += loss.item()

            if batch_idx % 100 == 0:
                print(f"Batch {batch_idx}, Loss: {loss.item():.4f}")
                print(f"Average pred norm: {predictions.norm(dim=-1).mean():.4f}")
                print(f"Average target norm: {targets.norm(dim=-1).mean():.4f}")

        avg_loss = total_loss / len(train_loader)
        print(f"Epoch [{epoch+1}/{num_epochs}], Average Loss: {avg_loss:.4f}")

        # Gradually increase momentum as training progresses
        if epoch > 0 and epoch % 5 == 0:
            model.momentum = min(0.999, model.momentum + 0.001)

    torch.save(model.state_dict(), 'jepa_model.pth')

def evaluate_current_model(model, device):
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


if __name__ == "__main__":
    device = get_device()
    train_model(device)

    probe_train_ds, probe_val_ds = load_data(device)
    model = load_model()
    evaluate_model(device, model, probe_train_ds, probe_val_ds)
