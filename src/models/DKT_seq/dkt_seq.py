import torch.nn as nn

class DeepKnowledgeTracing(nn.Module):
    def __init__(self, rnn_type, input_size, hidden_size, num_skills, nlayers,
                 dropout=0.6, tie_weights=False, bidirectional=False):
        super(DeepKnowledgeTracing, self).__init__()

        self.rnn_type = rnn_type
        self.nhid = hidden_size
        self.nlayers = nlayers
        self.bidirectional = bidirectional
        self.num_directions = 2 if self.bidirectional else 1

        # Pick RNN type with optional bidirectionality
        if rnn_type == 'LSTM':
            self.rnn = nn.LSTM(
                input_size, hidden_size, nlayers,
                batch_first=True, dropout=dropout,
                bidirectional=self.bidirectional
            )
        elif rnn_type == 'GRU':
            self.rnn = nn.GRU(
                input_size, hidden_size, nlayers,
                batch_first=True, dropout=dropout,
                bidirectional=self.bidirectional
            )
        elif rnn_type == 'RNN_TANH':
            self.rnn = nn.RNN(
                input_size, hidden_size, nlayers,
                nonlinearity='tanh', dropout=dropout,
                batch_first=True, bidirectional=self.bidirectional
            )
        else:
            self.rnn = nn.RNN(
                input_size, hidden_size, nlayers,
                nonlinearity='relu', dropout=dropout,
                batch_first=True, bidirectional=self.bidirectional
            )

        # Fully connected decoder layer, adjusted for bidirectional output
        self.decoder = nn.Linear(hidden_size * self.num_directions, num_skills)

        self.init_weights()

    def init_weights(self):
        initrange = 0.05
        self.decoder.bias.data.zero_()
        self.decoder.weight.data.uniform_(-initrange, initrange)

    def forward(self, input, hidden):
        output, hidden = self.rnn(input, hidden)
        # Flatten output: (batch * seq_len, hidden * directions)
        decoded = self.decoder(output.contiguous().view(-1, self.nhid * self.num_directions))
        return decoded, hidden

    def init_hidden(self, bsz):
        weight = next(self.parameters())
        hidden_shape = (self.nlayers * self.num_directions, bsz, self.nhid)
        if self.rnn_type == 'LSTM':
            return (
                weight.new_zeros(hidden_shape),
                weight.new_zeros(hidden_shape)
            )
        else:
            return weight.new_zeros(hidden_shape)

