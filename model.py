import torch
import torch.nn as nn
import torchvision.models as models


class EncoderCNN(nn.Module):
    def __init__(self, embed_size):
        super(EncoderCNN, self).__init__()
        resnet = models.resnet50(pretrained=True)
        for param in resnet.parameters():
            param.requires_grad_(False)
        
        modules = list(resnet.children())[:-1]
        self.resnet = nn.Sequential(*modules)
        self.bn1 = nn.BatchNorm2d(2048, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.embed = nn.Linear(resnet.fc.in_features, embed_size)

    def forward(self, images):
        features = self.resnet(images)
        features = features.view(features.size(0), -1)
        features = self.embed(features)
        return features
    

class DecoderRNN(nn.Module):
    def __init__(self, embed_size, hidden_size, vocab_size, num_layers=1):
        #embed_size -> 224
        #hidden_size -> 512
        #vocab_size -> 7072
        
        #1 embed
        #2 lstm
        #3 Linear
        
        super().__init__()
        
        #1
        self.embedding = nn.Embedding(vocab_size, embed_size)
        
        #attention - interest on play with it later
        #https://medium.com/intel-student-ambassadors/implementing-attention-models-in-pytorch-f947034b3e66
        #self.attn = nn.Linear(hidden_size + vocab_size, 1)
        
        #2 LSTM
        self.lstm = nn.LSTM(embed_size, hidden_size, num_layers, batch_first=True)
        
        #Dropout at this LSTM would not apply
        #https://discuss.pytorch.org/t/dropout-in-lstm/7784
        #"dropout – If non-zero, introduces a dropout layer on the outputs of each RNN layer except the last layer"
        #"If there is only one layer, dropout is not applied, as indicated in the docs (only layer = last layer)."
        
        #3 Linear
        self.lin = nn.Linear(hidden_size, vocab_size)
        
        #Init Weights
        #https://arxiv.org/pdf/1411.4555.pdf
        #not initialized - at this paper, if I have not misunderstood, they said that in CNN the weight initialization was great for avoidind dropout, but for embedding it appeared not to make much difference. I tried wothout, and it appeared to work even when I tested with the image of my dog. The result was not 100% right, but is was very good.
    
    def forward(self, features, captions):
        
        #outputs should be a PyTorch tensor with size [batch_size, captions.shape[1], vocab_size]
        
        #Discarting <end>
        #captions = captions[:, :-1]
        #1 Passing in embedding layer
        captions_emb = self.embedding(captions[:, :-1])
        
        #Concatenate
        #RuntimeError: Tensors must have same number of dimensions: got 3 and 2
        #https://github.com/pecia6/Image_Captioning/blob/master/model.py
        features_unq = features.unsqueeze(1)
        inputs = torch.cat((features_unq, captions_emb), 1)
        
        #2 lstm
        outputs, _hidden = self.lstm(inputs)
        
        #3 Linear
        out = self.lin(outputs)
        
        return out
    
    def sample(self, inputs, states=None, max_len=20):
        " accepts pre-processed image tensor (inputs) and returns predicted sentence (list of tensor ids of length max_len) "
        
        dict_out = []
        
        for i in range(max_len): 
            
            #LSTM
            lstm_out, states = self.lstm(inputs, states)            
            
            #Linear
            pred_out = self.lin(lstm_out)
            
            #Max - get best result
            pred_out = pred_out.squeeze(1)
            #print(pred_out.max(1))
            max_pred = pred_out.max(1)[1]
            
            #Store
            dict_out.append(max_pred.item()) 
            
            #Next word
            inputs = self.embedding(max_pred).unsqueeze(1)
            
        return dict_out