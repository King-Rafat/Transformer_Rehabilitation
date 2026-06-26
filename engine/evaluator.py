"""Evaluation: reports MAD, RMSE, MSE, MAPE on the test set."""
import numpy as np
import torch

from utils import performance_metrics


@torch.no_grad()
def evaluate(net, test_loader, data_loader=None, device=None):
    """Evaluate a trained model.

    Args:
        net: a :class:`TemporalModel`.
        test_loader: test DataLoader.
        data_loader: KIMORE scaler object (for inverse-transforming scores), or None.
        device: torch device string.

    Returns:
        dict with keys ``mad``, ``rmse``, ``mse``, ``mape``.
    """
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    net = net.to(device)
    net.eval()

    preds, trues = [], []
    for x, y in test_loader:
        x, y = x.to(device), y.to(device)
        x = torch.unsqueeze(x, 1)
        b, d, t, n, c = x.size()
        x = x.view(-1, t, n, c)
        pred, _ = net(x, return_features=True)
        preds.extend(pred.flatten().cpu().numpy())
        trues.extend(y.flatten().cpu().numpy())

    preds = np.array(preds).reshape(-1, 1)
    trues = np.array(trues).reshape(-1, 1)

    if data_loader is not None and hasattr(data_loader, "sc2"):
        preds = data_loader.sc2.inverse_transform(preds)
        trues = data_loader.sc2.inverse_transform(trues)

    mad, rmse, mse, mape = performance_metrics(trues, preds)
    return {"mad": mad, "rmse": rmse, "mse": mse, "mape": mape}
