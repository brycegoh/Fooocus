from transformers import OneFormerProcessor, OneFormerForUniversalSegmentation
from modules.util import get_ade_class_to_idx_map
import numpy as np
from typing import List

class SegmentationMaskGenerator:
  def __init__(self):
      self.semantic_processor, self.semantic_model = self._init_segformer()

  def _init_segformer(self):
    processor = OneFormerProcessor.from_pretrained("shi-labs/oneformer_ade20k_dinat_large")
    model = OneFormerForUniversalSegmentation.from_pretrained("shi-labs/oneformer_ade20k_dinat_large")
    return processor, model
  
  def run_segmentation(self, image):
    self.semantic_model.to('cuda:0')
    inputs = self.semantic_processor(images=image, task_inputs=["semantic"], return_tensors="pt").to('cuda:0')
    outputs = self.semantic_model(**inputs)
    predicted_semantic_map = self.semantic_processor.post_process_semantic_segmentation(outputs, target_sizes=[image.size[::-1]])
    predicted_semantic_map=predicted_semantic_map[0]
    self.semantic_model.to('cpu')
    inputs.to('cpu')
    del inputs, outputs
    return predicted_semantic_map.to('cpu')
  
  def get_mask(self, img: np.ndarray, classes_to_avoid: List[str] = ['walls']):
    class_to_idx_map = get_ade_class_to_idx_map()
    segmentation_map = self.run_segmentation(img)
    idx_to_avoid = []
    for c in classes_to_avoid:
      if c not in class_to_idx_map:
        raise ValueError(f"Invalid class name: {c}")
      idx_to_avoid.append(class_to_idx_map[c])
    
    is_avoided = np.isin(segmentation_map, idx_to_avoid)

    mask = np.zeros(segmentation_map.shape + (3,), dtype=np.uint8)
    mask[is_avoided] = [0, 0, 0]  # Black
    mask[~is_avoided] = [255, 255, 255]  # White

    return mask