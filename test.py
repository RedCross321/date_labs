import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Создаем массивы значений x1 и x2 
x1 = np.arange(1, 101)
x2 = np.arange(1, 101)

# Создаем список для хранения всех комбинаций
data = []

# Генерируем все комбинации x1 и x2 с шумом для y
for i in x1:
    for j in x2:
        # Вычисляем y = 3*x1 + 8*x2
        y_exact = 3*i + 8*j
        
        # Генерируем случайный шум в интервале [0.01; 0.1]
        noise = np.random.uniform(0.01, 0.1)
        
        # Случайно выбираем знак шума (+/-)
        if np.random.random() > 0.5:
            noise = -noise
            
        # Добавляем шум к точному значению
        y_noisy = y_exact + noise
        
        data.append([i, j, y_noisy])

# Создаем DataFrame
df = pd.DataFrame(data, columns=['x1', 'x2', 'y'])

# Выводим первые 10 строк для проверки
print(df.head(10))

# Сохраняем в CSV-файл
df.to_csv('data_with_noise.csv', index=False)

print(f"Таблица с шумом создана с {len(df)} строками и сохранена в файл data_with_noise.csv")

# Визуализируем на 3D графике (используем подмножество данных для наглядности)
# Берем каждую 10-ю точку по каждой оси для разреженной визуализации
x1_sparse = x1[::10]
x2_sparse = x2[::10]

# Создаем сетку для точной поверхности
X1_mesh, X2_mesh = np.meshgrid(x1_sparse, x2_sparse)
Y_exact = 3*X1_mesh + 8*X2_mesh

# Создаем фигуру
fig = plt.figure(figsize=(12, 10))
ax = fig.add_subplot(111, projection='3d')

# Строим точную поверхность (полупрозрачную)
surface = ax.plot_surface(X1_mesh, X2_mesh, Y_exact, 
                        color='blue', alpha=0.3, 
                        label='Точная функция')

# Выбираем точки с шумом для тех же значений x1 и x2
points = []
for i in x1_sparse:
    for j in x2_sparse:
        # Ищем соответствующую строку в данных с шумом
        mask = (df['x1'] == i) & (df['x2'] == j)
        if not df[mask].empty:
            row = df[mask].iloc[0]
            points.append([i, j, row['y']])

points = np.array(points)

# Строим точки с шумом
ax.scatter(points[:, 0], points[:, 1], points[:, 2], 
          color='red', s=30, alpha=0.7,
          label='Точки с шумом')

# Настраиваем оси и заголовок
ax.set_xlabel('x1')
ax.set_ylabel('x2')
ax.set_zlabel('y = 3*x1 + 8*x2 + шум')
ax.set_title('3D график зависимости y = 3*x1 + 8*x2 с шумом в интервале [0.01; 0.1]', fontsize=14)

plt.tight_layout()
plt.show()