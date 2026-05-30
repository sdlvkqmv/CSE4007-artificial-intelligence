import matplotlib.pyplot as plt

t = [0, 1, 2, 3, 4, 5, 6]
y = [1, 4, 5, 8, 9, 5, 3]

plt.figure(figsize=(10,6))
plt.plot(t, y, color='green', linewidth=6, marker='s', markersize=15, markerfacecolor='red')
plt.show()
