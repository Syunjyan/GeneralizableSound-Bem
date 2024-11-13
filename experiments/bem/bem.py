import sys

sys.path.append("./")

import bempp.api
import numpy as np

# 创建网格
grid = bempp.api.shapes.sphere(h=0.1)

# 定义函数空间
space = bempp.api.function_space(grid, "P", 1)

# 定义边界条件
@bempp.api.real_callable
def dirichlet_data(x, n, domain_index, result):
    result[0] = np.sin(np.pi * x[0])

# 创建 Dirichlet 边界条件
dirichlet_fun = bempp.api.GridFunction(space, fun=dirichlet_data)

# 定义 Helmholtz 方程的单层和双层算子
slp = bempp.api.operators.boundary.helmholtz.single_layer(space, space, space, 1)
dlp = bempp.api.operators.boundary.helmholtz.double_layer(space, space, space, 1)

# 定义右侧向量
rhs = dlp * dirichlet_fun

# 求解方程
from bempp.api.linalg import gmres
sol, info = gmres(slp, rhs)

# 输出结果
print("Solution:", sol)
