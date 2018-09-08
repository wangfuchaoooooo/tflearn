"""linear regression example"""

import tflearn
import matplotlib.pyplot as plt
import numpy as np

# Regression Data
X = [3.3, 4.4, 5.5, 6.71, 6.93, 4.168, 9.779, 6.182, 7.59, 2.167, 7.042, 10.791, 5.313, 7.997, 5.654, 9.27, 3.1]
Y = [1.7, 2.76, 2.09, 3.19, 1.694, 1.573, 3.366, 2.596, 2.53, 1.221, 2.827, 3.465, 1.65, 2.904, 2.42, 2.94, 1.3]
plt.scatter(X, Y, label='orig_Data')


# linear regression graph
input_ = tflearn.input_data(shape=[None])
linear = tflearn.single_unit(input_)
regression = tflearn.regression(linear, optimizer='sgd', loss='mean_square', metric='R2', learning_rate=0.01)
model = tflearn.DNN(regression)
model.fit(X, Y, n_epoch=1000, show_metric=False, snapshot_epoch=True)


print("\nRegression result:")
print("Y = " + str(model.get_weights(linear.W)) + "*X + " + str(model.get_weights(linear.b)))

new_x = np.arange(0, 20)
new_y = model.get_weights(linear.W) * new_x + model.get_weights(linear.b)
plt.plot(new_x, new_y, label='line')
plt.legend()

print("\nTest prediction for x = 3.2, 3.3, 3.4:")
print(model.predict([3.2, 3.3, 3.4]))

plt.scatter([3.2, 3.3, 3.4], model.predict([3.2, 3.3, 3.4]), label='predict')
plt.legend()
plt.show()
