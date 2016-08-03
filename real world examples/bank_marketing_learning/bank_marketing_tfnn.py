import tfnn
import pandas as pd

bank_data = pd.read_csv('bank-full.csv', sep=';')

data = tfnn.Data(bank_data.iloc[:, :-1], bank_data.iloc[:, -1])
data.encode_cat_y(inplace=True)
data.encode_cat_x(inplace=True)
network = tfnn.ClfNetwork(data.xs.shape[1], data.ys.shape[1],)
data = network.normalizer.minmax_fit(data, -1, 1)
train_data, test_data = data.train_test_split()
network.add_hidden_layer(50, activator=tfnn.nn.relu)
network.add_output_layer(activator=None)
network.set_optimizer(tfnn.train.GradientDescentOptimizer(0.0001))
evaluator = tfnn.Evaluator(network)

for i in range(1000):
    b_xs, b_ys = train_data.next_batch(100, loop=True)
    network.run_step(b_xs, b_ys)
    if i % 50 == 0:
        print(evaluator.compute_accuracy(test_data.xs, test_data.ys))
# print(test_data.ys.iloc[:,0].value_counts())
print(network.predict(test_data.xs.iloc[20:30, :]))
print(test_data.ys.iloc[20:30, :])
