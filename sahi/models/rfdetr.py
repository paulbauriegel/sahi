# OBSS SAHI Tool
# Code written by Fatih C Akyon, 2020.

import logging
from typing import Any, Dict, List, Optional

import numpy as np
from rfdetr import RFDETRBase, RFDETRLarge
from rfdetr.util.coco_classes import COCO_CLASSES

from sahi.models.base import DetectionModel
from sahi.prediction import ObjectPrediction
from sahi.utils.compatibility import fix_full_shape_list, fix_shift_amount_list
from sahi.utils.torch import empty_cuda_cache, has_torch, select_device

logger = logging.getLogger(__name__)


class RFDetrDetectionModel(DetectionModel):
    def __init__(
        self,
        model_path: Optional[str] = None,
        model: Optional[Any] = None,
        config_path: Optional[str] = None,
        device: Optional[str] = None,
        mask_threshold: float = 0.5,
        confidence_threshold: float = 0.3,
        category_mapping: Optional[Dict] = None,
        category_remapping: Optional[Dict] = None,
        load_at_init: bool = True,
        image_size: Optional[int] = None,
    ):
        """
        Init object detection/instance segmentation model.
        Args:
            model_path: str
                Path for the instance segmentation model weight
            config_path: str
                Path for the mmdetection instance segmentation model config file
            device: Torch device, "cpu", "mps", "cuda", "cuda:0", "cuda:1", etc.
            mask_threshold: float
                Value to threshold mask pixels, should be between 0 and 1
            confidence_threshold: float
                All predictions with score < confidence_threshold will be discarded
            category_mapping: dict: str to str
                Mapping from category id (str) to category name (str) e.g. {"1": "pedestrian"}
            category_remapping: dict: str to int
                Remap category ids based on category names, after performing inference e.g. {"car": 3}
            load_at_init: bool
                If True, automatically loads the model at initialization
            image_size: int
                Inference input size.
        """
        self.model_path = model_path
        self.config_path = config_path
        self.model = None
        self.mask_threshold = mask_threshold
        self.confidence_threshold = confidence_threshold
        self.category_mapping = category_mapping
        self.category_remapping = category_remapping
        self.image_size = image_size
        self._original_predictions = None
        self._object_prediction_list_per_image = None
        self.set_device()

        # automatically load model if load_at_init is True
        if load_at_init:
            if model:
                self.set_model(model)
            else:
                self.load_model()

        if self.category_mapping is None:
            self.category_mapping = COCO_CLASSES

    def check_dependencies(self) -> None:
        """
        This function can be implemented to ensure model dependencies are installed.
        """
        pass

    def load_model(self):
        """
        This function should be implemented in a way that detection model
        should be initialized and set to self.model.
        (self.model_path, self.config_path, and self.device should be utilized)
        """
        self.model = RFDETRLarge(resolution=728) # self.model_path

    def set_model(self, model: Any, **kwargs):
        """
        This function should be implemented to instantiate a DetectionModel out of an already loaded model
        Args:
            model: Any
                Loaded model
        """
        self.model = model

    def perform_inference(self, image: np.ndarray):
        """
        This function should be implemented in a way that prediction should be
        performed using self.model and the prediction result should be set to self._original_predictions.
        Args:
            image: np.ndarray
                A numpy array that contains the image to be predicted.
        """
        self._original_predictions = self.model.predict(image, threshold=self.confidence_threshold)

    def _create_object_prediction_list_from_original_predictions(
        self,
        shift_amount_list: Optional[List[List[int]]] = [[0, 0]],
        full_shape_list: Optional[List[List[int]]] = None,
    ):
        """
        Converts raw model predictions to SAHI ObjectPrediction format.
        Handles coordinate shifting for sliced inference and category remapping.
        """
        object_prediction_list = []
        shift_amount_list = fix_shift_amount_list(shift_amount_list)
        full_shape_list = fix_full_shape_list(full_shape_list)
        prediction = self._original_predictions
        bbox = prediction.xyxy  # [xmin, ymin, xmax, ymax]
        score = prediction.confidence
        category_ids = prediction.class_id
        detections_nbr = bbox.shape[0]
        for image_ind in range(detections_nbr):
          shift_amount = shift_amount_list[0]
          full_shape = None if full_shape_list is None else full_shape_list[0]
          category_id=int(category_ids[image_ind])

          # Create ObjectPrediction instance
          object_prediction = ObjectPrediction(
                    bbox=bbox[image_ind],
                    score=score[image_ind],
                    category_id=category_id,
                    category_name=self.category_mapping[category_id],
                    shift_amount=shift_amount,
                    full_shape=full_shape,
                )
          object_prediction_list.append(object_prediction)

        self._object_prediction_list_per_image = [object_prediction_list]
