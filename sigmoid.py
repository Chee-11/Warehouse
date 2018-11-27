from sympy import *
import numpy as np

# 连续值
x = symbols("x")  # 符号x，自变量
y = 1/(1+exp(-x))
dif_1 = diff(y,x) #求导
print(dif_1)  #打印导数
# plot(dif_1)
dif_2 = diff(dif_1,x)
print(dif_2)
# plot(dif_2)
dif_3 = diff(dif_2,x)
print(dif_3)
# plot(dif_3)
dif_4 = diff(dif_3,x)
print(dif_4)
plot(dif_4)

# 离散值
