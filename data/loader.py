import trimesh
from typing import List
import kagglehub
from trimesh import Geometry


class ModelNetLoader:
    def __init__(self, dataset_path: str) -> None:
        self.dataset_path = dataset_path

    def load_dataset(self) -> Geometry | list[Geometry]:
        mesh = trimesh.load(self.dataset_path)
        return mesh

    def download_dataset(self) -> str:
        self.dataset_path = kagglehub.dataset_download("balraj98/modelnet40-princeton-3d-object-dataset")
        return self.dataset_path


