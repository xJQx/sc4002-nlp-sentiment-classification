import torch
import torch.nn as nn

class RNN(nn.Module):
    def __init__(self, embedding_matrix, hidden_dim, output_dim, num_layers, sentence_representation_type: str):
        super(RNN, self).__init__()
        
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.num_layers = num_layers
        
        if sentence_representation_type not in ["last", "max", "average"]:
            raise Exception("Invalid `sentence_representation_type`")
        self.sentence_representation_type = sentence_representation_type

        # embedding layer
        vocab_size, embedding_dim = embedding_matrix.shape
        self.embedding = nn.Embedding.from_pretrained(torch.FloatTensor(embedding_matrix), freeze=True)
        
        # rnn layer
        self.rnn = nn.RNN(embedding_dim, hidden_dim, num_layers, batch_first=True)

        # non-linear layer
        self.relu = nn.ReLU()
        
        # fc layer
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        x = self.embedding(x)
        
        output, hidden = self.rnn(x)
        
        # use last hidden state as sentence representation
        if self.sentence_representation_type == "last":
            sentence_representation = hidden[-1]
        elif self.sentence_representation_type == "max":
            # TODO
            pass
        elif self.sentence_representation_type == "average":
            # TODO
            pass

        # non-linear layer
        sentence_representation = self.relu(sentence_representation)
        
        # output layer
        logits = self.fc(sentence_representation)
        
        return logits
        
        
        
        
        
        
        


        
        
        
        
        
