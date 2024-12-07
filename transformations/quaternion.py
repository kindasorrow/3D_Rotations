import numpy as np
import trimesh
from abstract.a_rotation import Rotation
from abc import ABC

class QuaternionRotation(Rotation, ABC):
    @staticmethod
    def rotate(mesh, quaternion: list[float]) -> 'trimesh.Trimesh':
        """
        Применяет вращение к 3D-модели с использованием кватерниона.

        :param mesh: Трёхмерная модель, к которой будет применено вращение (тип данных: trimesh.Trimesh).
        :param quaternion: Кватернион (x, y, z, w), который описывает вращение (тип данных: list[float] длиной 4).

        :return: Возвращает новую модель после применения вращения (тип данных: trimesh.Trimesh).
        """
        x, y, z, w = quaternion

        # Матрица вращения из кватерниона
        R = np.array([
            [1 - 2 * (y ** 2 + z ** 2), 2 * (x * y - w * z), 2 * (x * z + w * y)],
            [2 * (x * y + w * z), 1 - 2 * (x ** 2 + z ** 2), 2 * (y * z - w * x)],
            [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x ** 2 + y ** 2)]
        ])

        # Применяем матрицу вращения к модели
        return mesh.apply_transform(np.vstack((np.hstack((R, np.zeros((3, 1)))), [0, 0, 0, 1])))

    @staticmethod
    def to_vector_angle(quaternion: list[float]) -> tuple[np.ndarray, float]:
        """
        Преобразует кватернион в вектор угла и угол.

        :param quaternion: Кватернион (x, y, z, w) для преобразования (тип данных: list[float] длиной 4).

        :return: Кортеж, содержащий вектор (x, y, z) и угол вращения в радианах (тип данных: tuple[np.ndarray, float]).
        """
        x, y, z, w = quaternion
        angle = 2 * np.arccos(w)

        sin_theta_half = np.sqrt(1 - w ** 2)

        # Обработка случая, когда угол близок к нулю
        if sin_theta_half < 1e-6:
            vector = np.array([1, 0, 0])
        else:
            vector = np.array([x, y, z]) / sin_theta_half

        return vector, angle

    @staticmethod
    def to_bryan(quaternion: list[float]) -> tuple[float, float, float]:
        """
        Преобразует кватернион в углы Эйлера по методу Брайана (yaw, pitch, roll).

        :param quaternion: Кватернион (x, y, z, w) для преобразования (тип данных: list[float] длиной 4).

        :return: Кортеж углов Брайана (yaw, pitch, roll) в радианах (тип данных: tuple[float, float, float]).
        """
        x, y, z, w = quaternion

        yaw = np.arctan2(2 * (w * z + x * y), 1 - 2 * (y ** 2 + z ** 2))
        pitch = np.arcsin(2 * (w * y - z * x))
        roll = np.arctan2(2 * (w * x + y * z), 1 - 2 * (x ** 2 + y ** 2))

        return yaw, pitch, roll

    @staticmethod
    def to_basis(quaternion: list[float]) -> np.ndarray:
        """
        Преобразует кватернион в матрицу вращения (базис).

        :param quaternion: Кватернион (x, y, z, w) для преобразования (тип данных: list[float] длиной 4).

        :return: Матрица вращения 3x3 (тип данных: np.ndarray).
        """
        x, y, z, w = quaternion

        # Формируем матрицу вращения из кватерниона
        R = np.array([
            [1 - 2 * (y ** 2 + z ** 2), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x ** 2 + z ** 2), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x ** 2 + y ** 2)]
        ])

        return R
