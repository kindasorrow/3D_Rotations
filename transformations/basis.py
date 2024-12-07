import numpy as np


class BasisRotation:
    @staticmethod
    def rotate(mesh: object, basis: np.ndarray) -> object:
        """
        Применяет поворот к 3D-объекту (mesh) с использованием матрицы поворота (basis).

        Параметры:
            mesh (object): 3D-объект, к которому применяется трансформация.
            basis (np.ndarray): Матрица 3x3, представляющая поворот (матрица поворота).

        Возвращаемое значение:
            object: Трансформированный 3D-объект.
        """
        return mesh.apply_transform(np.vstack((np.hstack((basis, np.zeros((3, 1)))), [0, 0, 0, 1])))

    @staticmethod
    def to_vector_angle(basis: np.ndarray) -> tuple[np.ndarray, float]:
        """
        Преобразует матрицу поворота в вектор и угол, который соответствует этому повороту.

        Параметры:
            basis (np.ndarray): Матрица 3x3, представляющая поворот.

        Возвращаемое значение:
            tuple: Кортеж из вектора (np.ndarray), представляющего ось вращения,
                   и угла (float), соответствующего этому повороту.
        """
        trace = np.trace(basis)
        angle = np.arccos((trace - 1) / 2)

        if np.sin(angle) < 1e-6:
            return angle, np.array([1, 0, 0])

        vx = basis[2, 1] - basis[1, 2]
        vy = basis[0, 2] - basis[2, 0]
        vz = basis[1, 0] - basis[0, 1]

        vector = np.array([vx, vy, vz])
        vector = vector / (2 * np.sin(angle))

        return vector, angle

    @staticmethod
    def to_basis(vector: np.ndarray, angle: float) -> np.ndarray:
        """
        Преобразует вектор и угол в матрицу поворота.

        Параметры:
            vector (np.ndarray): Вектор оси вращения (должен быть нормализован).
            angle (float): Угол поворота в радианах.

        Возвращаемое значение:
            np.ndarray: Матрица 3x3, представляющая поворот.
        """
        vector = vector / np.linalg.norm(vector)
        x, y, z = vector

        if abs(x) < 1e-6 and abs(z) < 1e-6:
            v = np.array([1, 0, 0])
        else:
            v = np.array([0, 1, 0])

        u2 = np.cross(v, vector)
        u2 = u2 / np.linalg.norm(u2)

        u3 = np.cross(vector, u2)

        basis = np.column_stack((vector, u2, u3))

        c = np.cos(angle)
        s = np.sin(angle)
        R_local = np.array([
            [1, 0, 0],
            [0, c, -s],
            [0, s, c]
        ])

        return basis @ R_local @ np.linalg.inv(basis)

    @staticmethod
    def to_bryan(basis: np.ndarray) -> tuple[float, float, float]:
        """
        Преобразует матрицу поворота в углы Брайана (yaw, pitch, roll).

        Параметры:
            basis (np.ndarray): Матрица 3x3, представляющая поворот.

        Возвращаемое значение:
            tuple: Кортеж из трех углов (float) — roll, pitch, yaw (в радианах).
        """
        if abs(basis[2, 0]) != 1:  # basis[2, 0] is -basis31
            # Compute pitch
            pitch = np.arcsin(-basis[2, 0])

            # Compute yaw and roll
            yaw = np.arctan2(basis[1, 0], basis[0, 0])  # basis21, basisR11
            roll = np.arctan2(basis[2, 1], basis[2, 2])  # basis32, basis33
        else:
            # Handle gimbal lock (pitch = ±90°)
            yaw = 0
            if basis[2, 0] == -1:  # basis31 = -1
                pitch = np.pi / 2
                roll = np.arctan2(-basis[0, 1], basis[0, 2])  # -basis12, basis13
            else:  # R31 = 1
                pitch = -np.pi / 2
                roll = np.arctan2(basis[0, 1], basis[0, 2])  # basis12, basis13

        return roll, pitch, yaw

    @staticmethod
    def to_quaternion(R: np.ndarray) -> np.ndarray:
        """
        Преобразует матрицу поворота в кватернион.

        Параметры:
            R (np.ndarray): Матрица 3x3, представляющая поворот.

        Возвращаемое значение:
            np.ndarray: Кватернион, представляющий поворот.
        """
        trace = np.trace(R)

        if trace > 0:
            w = np.sqrt(1 + trace) / 2
            x = (R[2, 1] - R[1, 2]) / (4 * w)
            y = (R[0, 2] - R[2, 0]) / (4 * w)
            z = (R[1, 0] - R[0, 1]) / (4 * w)

        elif R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
            x = np.sqrt(1 + R[0, 0] - R[1, 1] - R[2, 2]) / 2
            w = (R[2, 1] - R[1, 2]) / (4 * x)
            y = (R[0, 1] + R[1, 0]) / (4 * x)
            z = (R[0, 2] + R[2, 0]) / (4 * x)

        elif R[1, 1] > R[2, 2]:
            y = np.sqrt(1 + R[1, 1] - R[0, 0] - R[2, 2]) / 2
            w = (R[0, 2] - R[2, 0]) / (4 * y)
            x = (R[0, 1] + R[1, 0]) / (4 * y)
            z = (R[1, 2] + R[2, 1]) / (4 * y)

        else:
            z = np.sqrt(1 + R[2, 2] - R[0, 0] - R[1, 1]) / 2
            w = (R[1, 0] - R[0, 1]) / (4 * z)
            x = (R[0, 2] + R[2, 0]) / (4 * z)
            y = (R[1, 2] + R[2, 1]) / (4 * z)

        quaternion = np.array([x, y, z, w])
        quaternion /= np.linalg.norm(quaternion)

        return quaternion
