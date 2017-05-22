import autodiff as ad
import numpy as np

np.random.seed(0)
N = 100  # number of points per class
D = 2  # dimensionality
K = 3  # number of classes
X_data = np.zeros((N * K, D))
y_data = np.zeros(N * K, dtype='uint8')
for j in range(K):
    ix = range(N * j, N * (j + 1))
    r = np.linspace(0.0, 1, N)
    t = np.linspace(j * 4, (j + 1) * 4, N) + np.random.randn(N) * 0.2  # theta
    X_data[ix] = np.c_[r * np.sin(t), r * np.cos(t)]
    y_data[ix] = j

print(y_data.shape)
y_one_hot = np.zeros((N * K, K))
y_one_hot[range(N * K), y_data] = 1

######################################
x = ad.Variable(name="x")
y = ad.Variable(name='y')
w = ad.Variable(name="w")
b = ad.Variable(name="b")

z = ad.matmul_op(x, w)
softmax = z + ad.broadcastto_op(b, z)

loss = ad.softmax_crossentropy_op(softmax, y)
grad_w, grad_b = ad.gradients(loss, [w, b])
executor = ad.Executor([loss, grad_w, grad_b])

w_val = np.zeros((D, K))
b_val = np.zeros(K)

n_epoch = 1000
lr = 0.01

for i in range(n_epoch):
    loss_val, grad_w_val, grad_b_val = executor.run(feed_dict={x: X_data, w: w_val, y: y_one_hot, b: b_val})
    if i % 10 == 0:
        print(loss_val[0])
    w_val = w_val - lr * grad_w_val
    b_val = b_val - lr * grad_b_val

z = ad.matmul_op(x, w)
softmax = z + ad.broadcastto_op(b, z)
softmax = ad.softmax_op(softmax)
executor = ad.Executor([softmax, x, w, b])
a, b, c, d = executor.run(feed_dict={x: X_data, w: w_val, b: b_val})
print(np.argmax(a, axis=1))
print(y_data)
correct = np.sum(np.equal(y_data, np.argmax(a, axis=1)))
print(correct / (N * K))