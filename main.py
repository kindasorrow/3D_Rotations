import numpy as np
import trimesh
from data.loader import ModelNetLoader
from transformations.vector_angle import VectorAngleRotation
from transformations.euler_angles import BryanRotation
from transformations.basis import BasisRotation
from transformations.quaternion import QuaternionRotation
from visualisation.plot import ObjectVisualizer

# Загрузка данных из ModelNet40
loader = ModelNetLoader('C:\\Users\\kinda\\.cache\\kagglehub\\datasets\\balraj98\\modelnet40-princeton-3d-object-dataset\\versions\\1')
print("Path to dataset files:", loader.dataset_path)

# Загрузка 3D-модели (пианино) из набора данных
mesh = trimesh.load(loader.dataset_path + '/ModelNet40/piano/test/piano_0331.off')
print(mesh)
mesh.show()  # Визуализация исходной 3D-модели

# --- Задание поворота через вектор и угол

vector = np.array([0, 1, 1])  # Ось вращения
angle = np.pi / 4  # Угол вращения в радианах

# Вращение модели вокруг оси с заданным углом
mesh_vector_rotated = mesh.copy()
VectorAngleRotation.rotate(mesh_vector_rotated, vector, angle)

# Конвертация вектора и угла в базис
basis = VectorAngleRotation.to_basis(vector, angle)
mesh_basis_rotated = mesh.copy()
BasisRotation.rotate(mesh_basis_rotated, basis)

# Конвертация вектора и угла в кватернион
quaternion = VectorAngleRotation.to_quaternion(vector, angle)
mesh_quaternion_rotated = mesh.copy()
QuaternionRotation.rotate(mesh_quaternion_rotated, quaternion)

# Конвертация вектора и угла в углы Брайана (Yaw, Pitch, Roll)
yaw, pitch, roll = VectorAngleRotation.to_bryan(vector, angle)
mesh_bryan_rotated = mesh.copy()
BryanRotation.rotate(mesh_bryan_rotated, yaw, pitch, roll)

# Визуализация всех вариантов вращения
mesh_objects = [mesh, mesh_vector_rotated, mesh_bryan_rotated, mesh_basis_rotated, mesh_quaternion_rotated]
titles = ["Original", "Vector & Angle", "Bryan", "Basis", "Quaternion"]
ObjectVisualizer.plot_objects_row(mesh_objects, titles)

# --- Задание поворота с помощью углов Брайана

yaw, pitch, roll = np.radians(50), np.radians(25), np.radians(30)  # Углы Брайана
mesh_bryan_rotated = mesh.copy()
BryanRotation.rotate(mesh_bryan_rotated, yaw, pitch, roll)

# Конвертация углов Брайана в вектор и угол
vector, angle = BryanRotation.to_vector_angle(yaw, pitch, roll)
mesh_vector_rotated = mesh.copy()
VectorAngleRotation.rotate(mesh_vector_rotated, vector, angle)

# Конвертация углов Брайана в базис
basis = BryanRotation.to_basis(roll, pitch, yaw)
mesh_basis_rotated = mesh.copy()
BasisRotation.rotate(mesh_basis_rotated, basis)

# Конвертация углов Брайана в кватернион
quaternion = BryanRotation.to_quaternion(roll, pitch, yaw)
mesh_quaternion_rotated = mesh.copy()
QuaternionRotation.rotate(mesh_quaternion_rotated, quaternion)

# Визуализация всех вариантов вращения
mesh_objects = [mesh, mesh_vector_rotated, mesh_bryan_rotated, mesh_basis_rotated, mesh_quaternion_rotated]
titles = ["Original", "Vector & Angle", "Bryan", "Basis", "Quaternion"]
ObjectVisualizer.plot_objects_row(mesh_objects, titles)

# --- Задание поворота с помощью базиса

basis = np.array([
    [0.866, -0.7, 0],  # Строки матрицы задают направление базисных векторов
    [0.7, 0.866, 0],
    [0, 0, 1]
])
mesh_basis_rotated = mesh.copy()
BasisRotation.rotate(mesh_basis_rotated, basis)

# Конвертация базиса в вектор и угол
vector, angle = BasisRotation.to_vector_angle(basis)
mesh_vector_rotated = mesh.copy()
VectorAngleRotation.rotate(mesh_vector_rotated, vector, angle)

# Конвертация базиса в углы Брайана
roll, pitch, yaw = BasisRotation.to_bryan(basis)
mesh_bryan_rotated = mesh.copy()
BryanRotation.rotate(mesh_bryan_rotated, yaw, pitch, roll)

# Конвертация базиса в кватернион
quaternion = BasisRotation.to_quaternion(basis)
mesh_quaternion_rotated = mesh.copy()
QuaternionRotation.rotate(mesh_quaternion_rotated, quaternion)

# Визуализация всех вариантов вращения
mesh_objects = [mesh, mesh_vector_rotated, mesh_bryan_rotated, mesh_basis_rotated, mesh_quaternion_rotated]
titles = ["Original", "Vector & Angle", "Bryan", "Basis", "Quaternion"]
ObjectVisualizer.plot_objects_row(mesh_objects, titles)

# --- Задание поворота с помощью кватерниона

quaternion = [0, 0, 0.707, 0.707]  # Кватернион вращения

# Вращение модели с использованием кватерниона
mesh_quaternion_rotated = mesh.copy()
QuaternionRotation.rotate(mesh_quaternion_rotated, quaternion)

# Конвертация кватерниона в вектор и угол
vector, angle = QuaternionRotation.to_vector_angle(quaternion)
mesh_vector_rotated = mesh.copy()
VectorAngleRotation.rotate(mesh_vector_rotated, vector, angle)

# Конвертация кватерниона в углы Брайана
yaw, pitch, roll = QuaternionRotation.to_bryan(quaternion)
mesh_bryan_rotated = mesh.copy()
BryanRotation.rotate(mesh_bryan_rotated, yaw, pitch, roll)

# Конвертация кватерниона в базис
basis = QuaternionRotation.to_basis(quaternion)
mesh_basis_rotated = mesh.copy()
BasisRotation.rotate(mesh_basis_rotated, basis)

# Визуализация всех вариантов вращения
mesh_objects = [mesh, mesh_vector_rotated, mesh_bryan_rotated, mesh_basis_rotated, mesh_quaternion_rotated]
titles = ["Original", "Vector & Angle", "Bryan", "Basis", "Quaternion"]
#ObjectVisualizer.plot_objects_row(mesh_objects, titles)
