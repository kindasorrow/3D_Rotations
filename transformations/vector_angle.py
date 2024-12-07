import numpy as np
import trimesh
from abc import ABC
from abstract.a_rotation import Rotation

class VectorAngleRotation(Rotation, ABC):
    @staticmethod
    def rotate(mesh, vector: np.ndarray, angle: float) -> 'trimesh.Trimesh':
        """
        Применяет вращение к 3D-модели с использованием вектора и угла.

        :param mesh: Трёхмерная модель, к которой будет применено вращение (тип данных: trimesh.Trimesh).
        :param vector: Вектор оси вращения (тип данных: np.ndarray с 3 элементами).
        :param angle: Угол вращения в радианах (тип данных: float).

        :return: Возвращает новую модель после применения вращения (тип данных: trimesh.Trimesh).
        """
        # Нормализация вектора
        vector = vector / np.linalg.norm(vector)

        # Матрица Клейна для оси вращения
        K = np.array([
            [0, -vector[2], vector[1]],
            [vector[2], 0, -vector[0]],
            [-vector[1], vector[0], 0]
        ])

        # Вычисление матрицы вращения R
        R = np.eye(3) + np.sin(angle) * K + (1 - np.cos(angle)) * np.dot(K, K)

        # Применение матрицы вращения к модели
        return mesh.apply_transform(np.vstack((np.hstack((R, np.zeros((3, 1)))), [0, 0, 0, 1])))

    @staticmethod
    def to_bryan(vector: np.ndarray, angle: float) -> tuple[float, float, float]:
        """
        Преобразует вектор и угол в углы Эйлера по методу Брайана (yaw, pitch, roll).

        :param vector: Вектор оси вращения (тип данных: np.ndarray с 3 элементами).
        :param angle: Угол вращения в радианах (тип данных: float).

        :return: Кортеж углов Брайана (yaw, pitch, roll) в радианах (тип данных: tuple[float, float, float]).
        """
        # Нормализация вектора
        vector = vector / np.linalg.norm(vector)
        x, y, z = vector

        c = np.cos(angle)
        s = np.sin(angle)
        v = 1 - c

        # Формирование матрицы вращения
        R = np.array([
            [x * x * v + c, x * y * v - z * s, x * z * v + y * s],
            [y * x * v + z * s, y * y * v + c, y * z * v - x * s],
            [z * x * v - y * s, z * y * v + x * s, z * z * v + c]
        ])

        # Расчёт углов Эйлера
        pitch = -np.arcsin(R[2, 0])
        yaw = np.arctan2(R[1, 0], R[0, 0])
        roll = np.arctan2(R[2, 1], R[2, 2])

        return yaw, pitch, roll

    @staticmethod
    def to_basis(vector: np.ndarray, angle: float) -> np.ndarray:
        """
        Преобразует вектор и угол в матрицу вращения в виде базиса.

        :param vector: Вектор оси вращения (тип данных: np.ndarray с 3 элементами).
        :param angle: Угол вращения в радианах (тип данных: float).

        :return: Матрица вращения 3x3 (тип данных: np.ndarray).
        """
        # Нормализация вектора
        vector = vector / np.linalg.norm(vector)
        x, y, z = vector

        # Выбор начального вектора
        if abs(x) < 1e-6 and abs(z) < 1e-6:
            v = np.array([1, 0, 0])
        else:
            v = np.array([0, 1, 0])

        # Вычисление ортогональных векторов
        u2 = np.cross(v, vector)
        u2 = u2 / np.linalg.norm(u2)
        u3 = np.cross(vector, u2)

        # Формирование базиса
        basis = np.column_stack((vector, u2, u3))

        # Матрица вращения в локальном пространстве
        c = np.cos(angle)
        s = np.sin(angle)
        R_local = np.array([
            [1, 0, 0],
            [0, c, -s],
            [0, s, c]
        ])

        # Возвращаем результат
        return basis @ R_local @ np.linalg.inv(basis)

    @staticmethod
    def to_quaternion(vector: np.ndarray, angle: float) -> np.ndarray:
        """
        Преобразует вектор и угол в кватернион.

        :param vector: Вектор оси вращения (тип данных: np.ndarray с 3 элементами).
        :param angle: Угол вращения в радианах (тип данных: float).

        :return: Кватернион, представляющий вращение (тип данных: np.ndarray с 4 элементами).
        """
        # Нормализация вектора
        vector = vector / np.linalg.norm(vector)

        # Вычисление кватерниона
        w = np.cos(angle / 2)
        x, y, z = np.sin(angle / 2) * vector

        return np.array([x, y, z, w])
