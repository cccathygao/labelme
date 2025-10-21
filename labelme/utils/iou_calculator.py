import numpy as np
import numpy.typing as npt


def calculate_iou(mask1: npt.NDArray[np.bool_], mask2: npt.NDArray[np.bool_]) -> float:
    """
    Calculate Intersection over Union (IoU) between two binary masks.
    
    Args:
        mask1: First binary mask
        mask2: Second binary mask
        
    Returns:
        IoU score between 0 and 1
    """
    if mask1.shape != mask2.shape:
        raise ValueError(f"Mask shapes must match: {mask1.shape} vs {mask2.shape}")
    
    intersection = np.logical_and(mask1, mask2).sum()
    union = np.logical_or(mask1, mask2).sum()
    
    if union == 0:
        return 0.0
    
    return float(intersection / union)
