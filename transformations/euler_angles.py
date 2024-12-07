from abc import ABC
import numpy as np
import trimesh
from abstract.a_rotation import Rotation


class BryanRotation(Rotation, ABC):
    """Класс для работы с поворотами по углам Бриана (yaw, pitch, roll)."""

    @staticmethod
    def rotate(mesh: 'trimesh.Trimesh', yaw: float, pitch: float, roll: float) -> 'trimesh.Trimesh':
        """
        Возвращает модель после применения поворота, заданного углами Бриана (yaw, pitch, roll).

        :param mesh: Трёхмерная модель, к которой будет применено вращение (тип данных: trimesh.Trimesh).
        :param yaw: Угол поворота вокруг оси Z (тип данных: float, радианы).
        :param pitch: Угол поворота вокруг оси Y (тип данных: float, радианы).
        :param roll: Угол поворота вокруг оси X (тип данных: float, радианы).

        :return: Модифицированная модель после применения поворота (тип данных: trimesh.Trimesh).
        """
        # Создание матрицы поворота по углам Бриана (yaw, pitch, roll)
        R_x = np.array([
            [1, 0, 0],
            [0, np.cos(roll), -np.sin(roll)],
            [0, np.sin(roll), np.cos(roll)]
        ])
        R_y = np.array([
            [np.cos(pitch), 0, np.sin(pitch)],
            [0, 1, 0],
            [-np.sin(pitch), 0, np.cos(pitch)]
        ])
        R_z = np.array([
            [np.cos(yaw), -np.sin(yaw), 0],
            [np.sin(yaw), np.cos(yaw), 0],
            [0, 0, 1]
        ])

        # Получение итоговой матрицы поворота
        R = R_z @ R_y @ R_x

        # Применение матрицы поворота к модели
        return mesh.apply_transform(np.vstack((np.hstack((R, np.zeros((3, 1)))), [0, 0, 0, 1])))

    @staticmethod
    def to_vector_angle(yaw: float, pitch: float, roll: float) -> tuple[np.ndarray, float]:
        """
        Преобразует углы Бриана (yaw, pitch, roll) в вектор и угол поворота.

        :param yaw: Угол поворота вокруг оси Z (тип данных: float, радианы).
        :param pitch: Угол поворота вокруг оси Y (тип данных: float, радианы).
        :param roll: Угол поворота вокруг оси X (тип данных: float, радианы).

        :return: Кортеж, содержащий вектор оси вращения и угол вращения в радианах
                 (тип данных: tuple[np.ndarray, float]).
        """
        # Получение матрицы поворота
        R = BryanRotation.to_basis(roll, pitch, yaw)

        # Вычисление угла вращения
        angle = np.arccos((np.trace(R) - 1) / 2)

        # Если угол вращения близок к нулю, то ось вращения можно выбрать произвольной
        if np.sin(angle) < 1e-6:
            vector = np.array([1, 0, 0])
        else:
            # Нахождение компонентов вектора оси вращения
            x = (R[2, 1] - R[1, 2]) / (2 * np.sin(angle))
            y = (R[0, 2] - R[2, 0]) / (2 * np.sin(angle))
            z = (R[1, 0] - R[0, 1]) / (2 * np.sin(angle))
            vector = np.array([x, y, z])

        # Нормализация вектора оси вращения
        vector = vector / np.linalg.norm(vector)

        return vector, angle

    @staticmethod
    def to_basis(roll: float, pitch: float, yaw: float) -> np.ndarray:
        """
        Преобразует углы Бриана (yaw, pitch, roll) в матрицу поворота.

        :param roll: Угол поворота вокруг оси X (тип данных: float, радианы).
        :param pitch: Угол поворота вокруг оси Y (тип данных: float, радианы).
        :param yaw: Угол поворота вокруг оси Z (тип данных: float, радианы).

        :return: Матрица поворота 3x3 (тип данных: np.ndarray).
        """
        # Создание матриц для каждого из углов
        R_x = np.array([
            [1, 0, 0],
            [0, np.cos(roll), -np.sin(roll)],
            [0, np.sin(roll), np.cos(roll)]
        ])
        R_y = np.array([
            [np.cos(pitch), 0, np.sin(pitch)],
            [0, 1, 0],
            [-np.sin(pitch), 0, np.cos(pitch)]
        ])
        R_z = np.array([
            [np.cos(yaw), -np.sin(yaw), 0],
            [np.sin(yaw), np.cos(yaw), 0],
            [0, 0, 1]
        ])

        # Возвращаем итоговую матрицу поворота
        return R_z @ R_y @ R_x

    @staticmethod
    def to_quaternion(roll: float, pitch: float, yaw: float) -> np.ndarray:
        """
        Преобразует углы Бриана (yaw, pitch, roll) в кватернион.

        :param roll: Угол поворота вокруг оси X (тип данных: float, радианы).
        :param pitch: Угол поворота вокруг оси Y (тип данных: float, радианы).
        :param yaw: Угол поворота вокруг оси Z (тип данных: float, радианы).

        :return: Кватернион, представляющий вращение (тип данных: np.ndarray с 4 элементами).
        """
        # Вычисление половинных углов
        half_roll = roll / 2
        half_pitch = pitch / 2
        half_yaw = yaw / 2

        # Вычисление косинусов и синусов половинных углов
        cos_roll = np.cos(half_roll)
        sin_roll = np.sin(half_roll)
        cos_pitch = np.cos(half_pitch)
        sin_pitch = np.sin(half_pitch)
        cos_yaw = np.cos(half_yaw)
        sin_yaw = np.sin(half_yaw)

        # Вычисление компонентов кватерниона
        w = cos_yaw * cos_pitch * cos_roll + sin_yaw * sin_pitch * sin_roll
        x = cos_yaw * cos_pitch * sin_roll - sin_yaw * sin_pitch * cos_roll
        y = cos_yaw * sin_pitch * cos_roll + sin_yaw * cos_pitch * sin_roll
        z = sin_yaw * cos_pitch * cos_roll - cos_yaw * sin_pitch * sin_roll

        # Возвращаем кватернион
        return np.array([x, y, z, w])
