import pytest
import torch
import torch.nn as nn
from unittest.mock import MagicMock, patch

from src.classifier.models.factory import get_model
from src.classifier.train import train_model
from src.classifier.evaluate import Evaluator

class MockModel(nn.Module):
    def __init__(self, num_classes=2):
        super().__init__()
        self.fc = nn.Linear(10, num_classes)

    def forward(self, x):
        batch_size = x.size(0)
        
        # Create a dummy input for the linear layer to connect the graph
        dummy_feat = torch.randn(batch_size, 10, device=x.device, requires_grad=True)
        
        pooled = x.mean(dim=[2, 3]) # (B, 3)
        if pooled.size(1) < 10:
            padded = torch.cat([pooled, torch.zeros(batch_size, 10-3, device=x.device)], dim=1)
        else:
            padded = pooled[:, :10]
            
        return self.fc(padded)

@pytest.fixture
def mock_loader():
    """Returns a dummy DataLoader."""
    data = []
    for _ in range(10): # 10 samples
        img = torch.randn(3, 224, 224) # Standard ImageNet size
        label = torch.randint(0, 2, (1,)).item()
        data.append((img, label))
    
    loader = torch.utils.data.DataLoader(data, batch_size=2)
    return loader

@pytest.fixture
def train_env(tmp_path):
    """Setup directories for training outputs."""
    save_dir = tmp_path / "models"
    save_dir.mkdir()
    return save_dir

def test_model_factory():
    """Verify get_model logic (Mocked)."""
    with patch("src.classifier.models.factory.VGGClassifier", MagicMock()) as mock_vgg:
        _ = get_model("vgg", num_classes=10)
        mock_vgg.assert_called_once()
        
    with patch("src.classifier.models.factory.ResNet152Classifier", MagicMock()) as mock_res:
        _ = get_model("resnet", freeze_backbone=False)
        # Verify kwargs passing
        _, kwargs = mock_res.call_args
        assert kwargs["freeze_backbone"] is False

def test_training_loop(train_env, mock_loader):
    """
    Verify train_model execution:
    - Runs without error
    - Saves checkpoints
    - Returns accuracy
    """
    model = MockModel(num_classes=2)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
    criterion = nn.CrossEntropyLoss()
    
    save_path = train_env / "best_model.pt"
    check_path = train_env / "ckpt_last.pt"
    
    acc = train_model(
        model=model,
        train_loader=mock_loader,
        valid_loader=mock_loader,
        criterion=criterion,
        optimizer=optimizer,
        device=torch.device("cpu"),
        epochs=1,
        save_path=str(save_path),
        check_path=str(check_path),
        num_classes=2
    )
    
    assert isinstance(acc, float)
    assert save_path.exists(), "Best model not saved"
    assert check_path.exists(), "Checkpoint not saved"

def test_evaluator_pipeline(train_env, mock_loader):
    """
    Verify Evaluator execution:
    - Loads model
    - Computes metrics
    - Saves artifacts (JSON, PNG)
    """
    # 1. Create a dummy trained model file
    model_name = "resnet" 
    model_path = train_env / f"{model_name}.pt"
    
    dummy_weights = MockModel(num_classes=2).state_dict()
    torch.save(dummy_weights, model_path)
    
    metric_dir = train_env / "metrics"
    
    # 2. Initialize Evaluator
    with patch.object(Evaluator, "_load_data", return_value=mock_loader):
        with patch("src.classifier.evaluate.get_model", return_value=MockModel(num_classes=2)):
            
            evaluator = Evaluator(
                model=model_name,   
                input_dir="dummy/path", 
                save_dir=train_env,
                metric_dir=metric_dir,
                num_classes=2
            )
            
            acc, f1 = evaluator.run()
            
            assert 0.0 <= acc <= 1.0
            
            # 3. Check Artifacts
            result_dir = metric_dir / model_name
            assert (result_dir / "metrics.json").exists()
            assert (result_dir / "confusion_matrix.png").exists()