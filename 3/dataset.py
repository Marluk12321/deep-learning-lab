import os
from operator import itemgetter
import numpy as np



class Dataset:
    def __init__(self, batch_size, sequence_length, dataset_path=None):
        self.batch_size = batch_size
        self.sequence_length = sequence_length

        if dataset_path is not None:
            self.preprocess(dataset_path)
            self.create_minibatches()

    def preprocess(self, input_file):
        with open(input_file, "r") as f:
            data = f.read()

        # count and sort most frequent characters
        char_frequencies = {}
        for c in data:
            if c not in char_frequencies:
                char_frequencies[c] = 0
            char_frequencies[c] += 1

        sorted_char_frequencies = \
            sorted(char_frequencies.items(), key=itemgetter(1), reverse=True)
        self.sorted_chars = list(map(lambda kv: kv[0], sorted_char_frequencies))

        # self.sorted chars contains just the characters ordered descending by frequency
        self.char2id = dict(zip(self.sorted_chars, range(len(self.sorted_chars))))
        # reverse the mapping
        self.id2char = {k:v for v, k in self.char2id.items()}
        # convert the data to ids
        self.x = self.encode(data)

    def encode(self, sequence):
        # returns the sequence encoded as integers
        return np.array(list(map(self.char2id.get, sequence)))

    def decode(self, encoded_sequence):
        # returns the sequence decoded as letters
        return list(map(self.id2char.get, encoded_sequence))

    def create_minibatches(self):
        # calculate the number of batches
        self.num_batches = int(len(self.x) / (self.batch_size * self.sequence_length))

        #######################################
        #       Convert data to batches       #
        #######################################
        self.current_batch = 0

    def next_minibatch(self):
        if self.current_batch is None:
            raise RuntimeError("next_minibatch called before create_minibatches")

        real_batch_size = self.batch_size * self.sequence_length
        batch_start = self.current_batch * real_batch_size # self.batch_indices[self.current_batch]
        batch_end = batch_start + real_batch_size
        batch_shape = (self.batch_size, self.sequence_length)

        batch_x = self.x[batch_start : batch_end].reshape(batch_shape)
        batch_y = self.x[batch_start+1 : batch_end+1].reshape(batch_shape)

        # handling batch pointer & reset
        # new_epoch is a boolean indicating if the batch pointer was reset
        # in this function call
        self.current_batch = (self.current_batch + 1) % self.num_batches
        new_epoch = self.current_batch == 0

        return new_epoch, batch_x, batch_y

    def one_hot_encode(self, encoded_sequence):
        oh_encoded = np.zeros((len(encoded_sequence), len(self.sorted_chars)))
        for i, index in enumerate(encoded_sequence):
            oh_encoded[i, index] = 1
        return oh_encoded




def main():
    root = "data"
    dataset_name = "selected_conversations.txt"
    dataset_path = os.path.join(root, dataset_name)
    dataset = Dataset(1, 2, dataset_path)

    minibatch = dataset.next_minibatch()
    print(minibatch)

if __name__ == "__main__":
    main()
