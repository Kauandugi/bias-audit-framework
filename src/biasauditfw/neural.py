"""Optional neural integrations used by the Colab execution."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Protocol

import pandas as pd
from PIL import Image

from .validation import match_boxes


FAIRFACE_FILE_ID = "11y0Wi3YQf21a_VcspUV4FwqzhMcfaVAB"
FAIRFACE_RACES = (
    "White",
    "Black",
    "Latino_Hispanic",
    "East Asian",
    "Southeast Asian",
    "Indian",
    "Middle Eastern",
)
FAIRFACE_GENDERS = ("Male", "Female")


class CropOracle(Protocol):
    """Callable contract shared by FairFace and deterministic test doubles."""

    def __call__(self, crop: Image.Image) -> dict[str, Any]:
        ...


def download_fairface_weights(destination: Path) -> Path:
    """Download the published seven-category FairFace checkpoint."""

    import gdown

    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    result = gdown.download(
        id=FAIRFACE_FILE_ID,
        output=str(destination),
        quiet=False,
    )
    if result is None or not destination.exists():
        raise RuntimeError("FairFace weights could not be downloaded.")
    return destination


class FairFaceOracle:
    """Secondary face-attribute reference; it is not demographic truth."""

    def __init__(self, weights_path: Path, device: Any) -> None:
        import torch
        import torch.nn as nn
        import torchvision

        weights_path = Path(weights_path)
        if not weights_path.exists():
            raise FileNotFoundError(f"FairFace weights not found: {weights_path}")
        self._torch = torch
        self.device = device
        self.model = torchvision.models.resnet34(weights=None)
        self.model.fc = nn.Linear(self.model.fc.in_features, 18)
        state = torch.load(weights_path, map_location=device, weights_only=True)
        self.model.load_state_dict(state)
        self.model.to(device).eval()
        self.transform = torchvision.transforms.Compose(
            [
                torchvision.transforms.Resize((224, 224)),
                torchvision.transforms.ToTensor(),
                torchvision.transforms.Normalize(
                    [0.485, 0.456, 0.406],
                    [0.229, 0.224, 0.225],
                ),
            ]
        )

    def __call__(self, crop: Image.Image) -> dict[str, Any]:
        tensor = self.transform(crop.convert("RGB")).unsqueeze(0).to(self.device)
        with self._torch.inference_mode():
            output = self.model(tensor)[0]
            race_probabilities = self._torch.softmax(output[:7], dim=0)
            gender_probabilities = self._torch.softmax(output[7:9], dim=0)
        race_index = int(race_probabilities.argmax())
        gender_index = int(gender_probabilities.argmax())
        return {
            "fairface_raca": FAIRFACE_RACES[race_index],
            "fairface_conf_raca": float(race_probabilities[race_index].cpu()),
            "fairface_genero": FAIRFACE_GENDERS[gender_index],
            "fairface_conf_genero": float(gender_probabilities[gender_index].cpu()),
        }


def run_fairface_on_ground_truth(
    oracle: CropOracle,
    truth: pd.DataFrame,
    images: pd.DataFrame,
    input_root: Path,
) -> pd.DataFrame:
    """Run the pseudo-oracle on Ground Truth crops, not detector crops."""

    paths = images.set_index("imagem_id")["caminho_relativo"].to_dict()
    rows: list[dict[str, Any]] = []
    for row in truth.itertuples(index=False):
        image_path = Path(input_root) / paths[row.imagem_id]
        with Image.open(image_path) as source:
            image = source.convert("RGB")
            x, y = max(0.0, row.bbox_x), max(0.0, row.bbox_y)
            right = min(float(image.width), x + max(0.0, row.bbox_w))
            bottom = min(float(image.height), y + max(0.0, row.bbox_h))
            if right <= x or bottom <= y:
                raise ValueError(f"Invalid Ground Truth crop: {row.gt_face_id}")
            crop = image.crop((x, y, right, bottom))
            rows.append(
                {
                    "imagem_id": row.imagem_id,
                    "gt_face_id": row.gt_face_id,
                    **oracle(crop),
                }
            )
    return pd.DataFrame(rows)


def align_deepface_to_ground_truth(
    faces: pd.DataFrame,
    truth: pd.DataFrame,
    threshold: float = 0.5,
) -> pd.DataFrame:
    """Align detected faces to Ground Truth with one-to-one IoU matching."""

    rows: list[dict[str, Any]] = []
    image_ids = sorted(set(faces["imagem_id"]) | set(truth["imagem_id"]))
    for image_id in image_ids:
        predicted = faces[faces["imagem_id"] == image_id].reset_index(drop=True)
        target = truth[truth["imagem_id"] == image_id].reset_index(drop=True)
        for predicted_index, target_index, iou in match_boxes(
            predicted, target, threshold
        ):
            predicted_row = predicted.iloc[predicted_index]
            target_row = target.iloc[target_index]
            rows.append(
                {
                    "imagem_id": image_id,
                    "face_id": predicted_row["face_id"],
                    "gt_face_id": target_row["gt_face_id"],
                    "match_iou": iou,
                    "deepface_raca": predicted_row["raca"],
                    "deepface_genero": predicted_row["genero"],
                }
            )
    return pd.DataFrame(rows)

