from pathlib import Path
import torch
import numpy as np
import pandas as pd
import imgaug
from imgaug.augmentables.bbs import BoundingBox


class CardiacDataset(torch.utils.data.Dataset):

    def __init__(self, path_to_labels_csv, patients, root_path, augs):
        
        self.labels = pd.read_csv(path_to_labels_csv)
        
        self.patients = np.load(patients)
        self.root_path = Path(root_path)
        self.augment = augs
        
    def  __len__(self):
        """
        Returns the length of the dataset
        """
        return len(self.patients)
        
    def __getitem__(self, idx):
        """
        Returns an image paired with bbox around the heart
        """
        patient = self.patients[idx]
        data = self.labels[self.labels["name"]==patient]
        patientId = data["name"].item()

        bbox = []
        x_min = data["x0"].item()
        bbox.append(x_min)
        y_min = data["y0"].item()
        bbox.append(y_min)
        x_max = x_min + data["w"].item()
        bbox.append(x_max)
        y_max = y_min + data["h"].item()
        bbox.append(y_max)

        file_path = self.root_path/str(patientId)
        img = np.load(f"{file_path}.npy").astype(np.float32)

        if self.augment:
            
            bb = BoundingBox(x1=bbox[0], y1=bbox[1], x2=bbox[2], y2=bbox[3])
            
            random_seed = torch.randint(0, 1000000, (1,))[0].item()
            imgaug.seed(random_seed)

            img, aug_bbox  = self.augment(image=img, bounding_boxes=bb)
            bbox = aug_bbox[0][0], aug_bbox[0][1], aug_bbox[1][0], aug_bbox[1][1]

        img = (img - 0.49466) / 0.25284

        img = torch.tensor(img).unsqueeze(0)
        bbox = torch.tensor(bbox)
            
        return img, bbox
