from abc import ABC, abstractmethod
import numpy as np

class Rotation(ABC):
    @abstractmethod
    def rotate(self, mesh: object, *params) -> object:
        """
        Применяет поворот к 3D-объекту (mesh) с использованием переданных параметров.

        Параметры:
            mesh (object): 3D-объект, к которому применяется трансформация.
            *params: Дополнительные параметры для поворота, зависящие от типа поворота.

        Возвращаемое значение:
            object: Трансформированный 3D-объект.
        """
        pass

    @abstractmethod
    def to_bryan(self) -> tuple[float, float, float]:
        """
        Преобразует текущий тип поворота в углы Бриана (yaw, pitch, roll).

        Возвращаемое значение:
            tuple: Кортеж из трех углов (float) — roll, pitch, yaw (в радианах).
        """
        pass

    @abstractmethod
    def to_quaternion(self) -> np.ndarray:
        """
        Преобразует текущий тип поворота в кватернион.

        Возвращаемое значение:
            np.ndarray: Кватернион, представляющий поворот.
        """
        pass

    @abstractmethod
    def to_vector_angle(self) -> tuple[np.ndarray, float]:
        """
        Преобразует текущий тип поворота в вектор и угол.

        Возвращаемое значение:
            tuple: Кортеж из вектора (np.ndarray) и угла (float) поворота.
        """
        pass

    @abstractmethod
    def to_basis(self) -> np.ndarray:
        """
        Преобразует текущий тип поворота в матрицу поворота.

        Возвращаемое значение:
            np.ndarray: Матрица 3x3, представляющая поворот.
        """
        pass
