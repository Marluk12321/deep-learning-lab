import os
import numpy as np
from rnn import RNN
from dataset import Dataset



root = "data"
dataset_name = "selected_conversations.txt"



def one_hot_from2d(x, dataset):
    xoh = []
    for xi in x:
        xoh.append(dataset.one_hot_encode(xi))
    return np.array(xoh)

def run_language_model(dataset, max_epochs, hidden_size=100, sequence_length=30,
                       learning_rate=1e-1, sample_every=100):
    vocab_size = len(dataset.sorted_chars)
    # initialize the recurrent network
    rnn = RNN(vocab_size, hidden_size, sequence_length, learning_rate)

    current_epoch = 0
    batch = 1

    h0 = np.zeros((dataset.batch_size, rnn.hidden_size))

    average_loss = 0.0

    while current_epoch < max_epochs:
        e, x, y = dataset.next_minibatch()

        if e:
            current_epoch += 1
            h0 = np.zeros((dataset.batch_size, rnn.hidden_size))

        # One-hot transform the x and y batches
        x_oh = one_hot_from2d(x, dataset)
        y_oh = one_hot_from2d(y, dataset)

        # Run the recurrent network on the current batch
        # Since we are using windows of a short length of characters,
        # the step function should return the hidden state at the end
        # of the unroll. You should then use that hidden state as the
        # input for the next minibatch. In this way, we artificially
        # preserve context between batches.
        loss, h0 = rnn.step(h0, x_oh, y_oh)
        average_loss += loss / sample_every

        if batch % sample_every == 0:
            print("current_epoch:", current_epoch, "loss:", average_loss)
            average_loss = 0.0
            seed = "HAN:\nIs that good or bad?\n\n"
            n_sample = len(seed) * 3
            print(rnn.sample(seed, n_sample, dataset))
            print()

        batch += 1

def main():
    dataset_path = os.path.join(root, dataset_name)
    batch_size = 15
    sequence_length = 30
    dataset = Dataset(batch_size, sequence_length, dataset_path)
    print("Num batches:", dataset.num_batches, "\n")
    max_epochs = 100
    sample_every = int(dataset.num_batches / 10)
    run_language_model(dataset, max_epochs, sequence_length=sequence_length,
                       sample_every=sample_every)

if __name__ == "__main__":
    main()
