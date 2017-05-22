import numpy as np
import autodiff as ad

x = ad.Variable(name='x')
y = ad.Variable(name='y')
W = ad.Variable(name='W')
b = ad.Variable(name='b')
z = ad.matmul_op(x, W)
output = z + ad.broadcastto_op(b, z)

num_point = 1000

cost = ad.reduce_sum_op((y - output) * (y - output)) / (2.0 * num_point)
grad_cost_w, grad_b = ad.gradients(cost, [W, b])

x_data = np.linspace(0, 10, num_point).reshape((num_point, 1))
y_data = 2.0 * x_data + np.random.uniform(-0.2, 0.2, (num_point, 1)) + 5.0 * np.ones((num_point, 1))

w_val = np.zeros((1, 1))
b_val = np.zeros(1)
executor = ad.Executor([cost, grad_cost_w, grad_b])
# train
n_epoch = 2000
lr = 0.01

print("training...")

for i in range(n_epoch):
    # evaluate the graph
    cost_val, grad_cost_w_val, grad_b_val = executor.run(feed_dict={x: x_data, W: w_val, y: y_data, b: b_val})
    print("cost: ", cost_val[0])
    w_val = w_val - lr * grad_cost_w_val
    b_val = b_val - lr * grad_b_val

print('\n')
print('W: ', w_val[0, 0])
print('b: ', b_val[0])
