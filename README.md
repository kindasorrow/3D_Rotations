# GraphicsExam

Проект "3D Rotations" представляет собой систему для работы с различными методами поворота 3D-объектов с использованием различных типов представлений: углов Бриана, векторных углов, матриц поворота и кватернионов. Проект включает абстрактные и конкретные классы для выполнения операций преобразования объектов и их визуализации.

## Описание

Проект реализует четыре основных типа поворота:

- **BasisRotation**: Работает с матрицами поворота, представленными в виде базисных векторов.
- **BryanRotation**: Использует углы Бриана (yaw, pitch, roll) для выполнения поворотов.
- **VectorAngleRotation**: Осуществляет повороты с использованием векторного угла.
- **QuaternionRotation**: Применяет кватернионы для поворота объектов.

Каждый класс реализует метод `rotate()`, который выполняет соответствующее преобразование 3D-объекта.

## Установка

1. Клонируйте репозиторий:

    ```bash
    git clone https://github.com/kindasorrow/3D_rotations.git
    ```

2. Перейдите в каталог проекта:

    ```bash
    cd 3D_rotations
    ```

3. Установите необходимые зависимости (если у вас их нет):

    ```bash
    pip install -r requirements.txt
    ```
   
4. Установите с помощью `ModelNetLoader.download_dataset()` данные из ModelNet40 

5. Укажите путь до датасета в `main.py` 

## Использование

Пример использования классов поворота:

```python
from transformations.basis_rotation import BasisRotation
from transformations.vector_angle import VectorAngleRotation
from transformations.quaternion_rotation import QuaternionRotation
from abstract.a_rotation import Rotation

# Создайте объект типа Mesh (представление 3D-объекта) и выполните поворот
mesh = ...  # Ваш 3D-объект

# Поворот с использованием BasisRotation
BasisRotation.rotate(mesh, basis_matrix)

# Поворот с использованием VectorAngleRotation
VectorAngleRotation.rotate(mesh, yaw, pitch, roll)

# Поворот с использованием QuaternionRotation
QuaternionRotation.rotate(mesh, quaternion)

