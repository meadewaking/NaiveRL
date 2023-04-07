import matplotlib.pyplot as plt
import math


class OBJ(object):
    def __init__(self):
        self.v = 30
        self.x = 0
        self.y = 0
        self.angle = 45


obj = OBJ()
T = 20  # 模拟时长
t = 0.005  # 模拟精度
g = 9.8  # 重力加速度
k = 0.005  # 空气阻力系数
m = 0.046
x, y = [], []
for i in range(int(T / t)):  # 无空气阻力
    vx = obj.v * math.cos(obj.angle)
    vy = obj.v * math.sin(obj.angle)
    vy -= g * t
    obj.angle = math.atan(vy / vx)
    obj.v = math.sqrt(math.pow(vx, 2) + math.pow(vy, 2))
    obj.x += vx * t
    obj.y += vy * t
    x.append(obj.x)
    y.append(obj.y)
    if obj.y < 0:
        break
plt.plot(x, y, 'b')
x, y = [], []
obj = OBJ()
obj.v = 1000
for i in range(int(T / t)):  # 有空气阻力
    vx = obj.v * math.cos(obj.angle)
    vy = obj.v * math.sin(obj.angle)
    vy -= g * t
    if vy < 0:
        vy = -(abs(vy) - k * math.pow(vy, 2) / m * t)
    else:
        vy -= k * math.pow(vy, 2) / m * t
    if vx < 0:
        vx = -(abs(vx) - k * math.pow(vx, 2) / m * t)
    else:
        vx -= k * math.pow(vx, 2) / m * t
    obj.angle = math.atan(vy / vx)
    obj.v = math.sqrt(math.pow(vx, 2) + math.pow(vy, 2))
    obj.x += vx * t
    obj.y += vy * t
    x.append(obj.x)
    y.append(obj.y)
    if obj.y < 0:
        break

plt.plot(x, y, 'r')
plt.show()
