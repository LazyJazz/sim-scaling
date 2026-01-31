import matplotlib.pyplot as plt

x = [400, 1000, 4000, 10000, 40000, 100000]
y = [0.79, 0.938, 0.987, 0.995, 0.992, 0.99]
y_damp = [0.797, 0.736, 0.889, 0.87, 0.886, 0.883]

x_multi = [4000, 10000, 40000]
y_multi_damp = [
    [0.827, 0.877, 0.829, 0.749, 0.911, 0.831, 0.851, 0.832],
    [0.879, 0.865, 0.86, 0.884, 0.9,  0.884, 0.815, 0.878],
    [0.859, 0.887, 0.789, 0.89, 0.909, 0.878, 0.92, 0.894]
]
y_multi_mean_damp = [sum(v)/len(v) for v in y_multi_damp]
y_multi_max_damp = [max(v) for v in y_multi_damp]
y_multi_min_damp = [min(v) for v in y_multi_damp]

y_multi = [
    [0.992, 0.992, 0.993, 0.992, 0.994, 0.994, 0.995, 0.991],
    [0.993, 0.997, 0.997, 0.998, 0.996, 0.995, 0.991, 0.995]
]
y_multi_mean = [sum(v)/len(v) for v in y_multi]
y_multi_max = [max(v) for v in y_multi]
y_multi_min = [min(v) for v in y_multi]

# overwrite y_multi_mean to y with corresponding x values
for i in range(len(x_multi)):
    for j in range(len(x)):
        if x_multi[i] == x[j]:
            y_damp[j] = y_multi_mean_damp[i]
            # y[j] = y_multi_mean[i]

# plt.plot(x, y, marker='o', color='blue', label='eval on sim')
# plt.plot(x, y_damp, marker='o', color='blue', linestyle='--', label='eval on targ')
# plt.fill_between(x_multi, y_multi_min_damp, y_multi_max_damp, color='blue', alpha=0.2, label='eval on trag, min-max of 8 trains')


# mark all the dots of y_multi_damp
# for i in range(len(x_multi)):
#     plt.scatter([x_multi[i]] * len(y_multi_damp[i]), y_multi_damp[i], color='blue', alpha=0.5)

# show y value on each point
# for i, j in zip(x, y):
#     plt.text(i, j + 0.02, f'{j:.3f}', ha='center')
# for i, j in zip(x, y_damp):
#     plt.text(i, j - 0.04, f'{j:.3f}', ha='center')


xd = [1000, 4000, 10000, 40000, 100000]
yd = [0.8,0.959, 0.97, 0.985, 0.976]
yd_damp = [0.959, 0.988, 0.998, 0.995, 0.988]

plt.plot(xd, yd, marker='s', color='orange', label='eval on sim, trained on targ')
plt.plot(xd, yd_damp, marker='s', color='orange', linestyle='--', label='eval on targ, trained on targ')

x_pure = [4000, 10000, 40000, 100000]
y_pure = [0.991, 0.994, 0.991, 0.972]
y_pure_damp = [0.819, 0.828, 0.801, 0.759]

# plt.plot(x_pure, y_pure, marker='^', color='cyan', label='eval on sim, trained on sim')
# plt.plot(x_pure, y_pure_damp, marker='^', color='cyan', linestyle='--', label='eval on targ, trained on sim')

xd_cotrain = [4000, 10000, 40000, 100000]
yd_cotrain = [0.937, 0.97, 0.976, 0.973]
yd_cotrain_damp = [0.993, 0.995, 0.994, 0.986]
plt.plot(xd_cotrain, yd_cotrain, marker='x', color='green', label='eval on sim, reversed')
plt.plot(xd_cotrain, yd_cotrain_damp, marker='x', color='green', linestyle='--', label='eval on targ, reversed')

plt.xlim(300, 120000)
plt.xscale('log')
# plt.ylim(0, 1)
plt.xlabel('#sim trajectories')
plt.ylabel('success rate')

plt.title('Success rate comparison of cotrained DP models (#target trajectory = 100)')
plt.grid(True, which="both", ls="--")
plt.legend()
# set figure size
plt.gcf().set_size_inches(10, 6)

plt.savefig('success_rate_comparison_plot.png')
# plt.show()