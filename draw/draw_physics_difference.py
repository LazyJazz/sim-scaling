import matplotlib.pyplot as plt
import json

x = [i * 0.1 for i in range(11)]
y_sim4000 = [0.99, 0.993, 0.99, 0.988, 0.991, 0.983, 0.97, 0.951, 0.931, 0.92, 0.845]
y_sim10000 = [0.995, 0.993, 0.995, 0.995, 0.988, 0.989, 0.967, 0.956, 0.939, 0.914, 0.849]
y_sim40000 = [0.992, 0.993, 0.988, 0.995, 0.99, 0.989, 0.972, 0.962, 0.927, 0.861, 0.805]
y_sim100000 = [0.973, 0.98, 0.973, 0.977, 0.974, 0.966, 0.972, 0.953, 0.941, 0.833, 0.776]

plt.plot(x, y_sim4000, marker='o', linestyle=':', color=(1, 0, 0), alpha=1.0, label='sim4000')
plt.plot(x, y_sim10000, marker='o', linestyle='-.', color=(1, 0, 0), alpha=1.0, label='sim10000')
plt.plot(x, y_sim40000, marker='o', linestyle='--', color=(1, 0, 0), alpha=1.0, label='sim40000')
plt.plot(x, y_sim100000, marker='o', linestyle='-', color=(1, 0, 0), alpha=1.0, label='sim100000')

y_cotrain_400 = [0.805, 0.802, 0.836, 0.853, 0.864, 0.874, 0.865, 0.869, 0.841, 0.861, 0.774]
y_cotrain_1000 = [0.957, 0.968, 0.966, 0.975, 0.955, 0.96, 0.957, 0.962, 0.901, 0.856, 0.75]
y_cotrain_4000 = [0.985, 0.984, 0.988, 0.991, 0.994, 0.991, 0.976, 0.973, 0.967, 0.932, 0.876]
y_cotrain_10000 = [0.996, 1.0, 0.999, 0.996, 0.997, 0.989, 0.981, 0.968, 0.963, 0.907,  0.859]
plt.plot(x, y_cotrain_400, marker='s', linestyle=' ', color=(0, 0, 1), alpha=1.0, label='cotrain_400_100')
plt.plot(x, y_cotrain_1000, marker='s', linestyle=':', color=(0, 0, 1), alpha=1.0, label='cotrain_1000_100')
plt.plot(x, y_cotrain_4000, marker='s', linestyle='-.', color=(0, 0, 1), alpha=1.0, label='cotrain_4000_100')
plt.plot(x, y_cotrain_10000, marker='s', linestyle='--', color=(0, 0, 1), alpha=1.0, label='cotrain_10000_100')
plt.xlabel('Physics difference')
plt.ylabel('Success rate')
plt.xlim(0, 1)
# plt.ylim(0, 1)
plt.grid(True)
plt.legend()
plt.title('Success rate v.s. physics difference')
# set size
plt.gcf().set_size_inches(8, 6)
plt.savefig('physics_difference_success_rate.png')