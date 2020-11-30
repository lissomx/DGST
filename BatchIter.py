import math
import operator
# from torch.nn.utils.rnn import pad_sequence
import torch


class BatchIter():
    def __init__(self, dataset, batch_size, batch_first=False, shuffle=True, quantity = 1, force_length = None):
        self.quantity = quantity
        self.dataset = dataset
        self.batch_size = batch_size
        self.id = 0
        self.batch_first = batch_first
        self.shuffle = shuffle
        self.force_length = force_length
        if shuffle:
            self.dataset.shuffle()


    def __len__(self):
        return math.ceil(self.quantity*len(self.dataset)/self.batch_size)

    def __getitem__(self, index):
        if self.id >= len(self.dataset):
            self.id = 0
            if self.shuffle:
                self.dataset.shuffle()
            raise StopIteration()
        # x = self.dataset[0:3]
        data0, data1 = self.dataset[self.id:min(self.id+self.batch_size,len(self.dataset))]
        sens0 = self.pad_sequence(data0, padding_value=0, batch_first=self.batch_first, force_length=self.force_length)
        sens1 = self.pad_sequence(data1, padding_value=0, batch_first=self.batch_first, force_length=self.force_length)
        self.id += self.batch_size
        return (sens0, sens1)

    def totext(self, sen):
        return self.dataset.totext(sen)
    
    def pad_sequence(self, sequences, batch_first=False, padding_value=0, force_length=None):
        # assuming trailing dimensions and type of all the Tensors
        # in sequences are same and fetching those from sequences[0]
        max_size = sequences[0].size()
        trailing_dims = max_size[1:]
        if force_length is None:
            max_len = max([s.size(0) for s in sequences])
        else:
            max_len = force_length
        if batch_first:
            out_dims = (len(sequences), max_len) + trailing_dims
        else:
            out_dims = (max_len, len(sequences)) + trailing_dims

        out_tensor = torch.empty(*out_dims, dtype=torch.long).fill_(padding_value)
        for i, tensor in enumerate(sequences):
            length = min(max_len, tensor.size(0))
            # use index notation to prevent duplicate references to the tensor
            if batch_first:
                out_tensor[i, :length, ...] = tensor[:length]
            else:
                out_tensor[:length, i, ...] = tensor[:length]
        return out_tensor