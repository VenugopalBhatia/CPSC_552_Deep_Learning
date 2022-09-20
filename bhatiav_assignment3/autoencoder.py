import torch
from torchvision import datasets
from torchvision import transforms
import matplotlib.pyplot as plt



tensor_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
        ])

dataset = datasets.MNIST(root = "./data",
                         train = True,
                         download = True,
                         transform = tensor_transform)
  

loader = torch.utils.data.DataLoader(dataset = dataset,
                                     batch_size = 128,
                                     shuffle = True)



class AE(torch.nn.Module):
    def __init__(self):
        super().__init__()
          
        
        self.encoder = torch.nn.Sequential(
            torch.nn.Linear(28 * 28, 390),
            torch.nn.ReLU(),
            torch.nn.Linear(390, 100),
            torch.nn.ReLU(),
            torch.nn.Linear(100, 20),
            torch.nn.ReLU(),
            torch.nn.Linear(20, 2)
        )
          
        
        self.decoder = torch.nn.Sequential(
            torch.nn.Linear(2, 20),
            torch.nn.ReLU(),
            torch.nn.Linear(20, 100),
            torch.nn.ReLU(),
            torch.nn.Linear(100, 390),
            torch.nn.ReLU(),
            torch.nn.Linear(390, 28 * 28),
            torch.nn.Sigmoid()
        )
  
    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded


model = AE()
  

loss_MSE = torch.nn.MSELoss()
loss_CE = torch.nn.CrossEntropyLoss()
  

optimizer = torch.optim.Adam(model.parameters(),
                             lr = 0.01,
                             weight_decay = 1e-6)


def train():                           
    epochs = 100
    outputs = []
    loss_vals = []
    for epoch in range(epochs):
        for (images, n) in loader:
            
            optimizer.zero_grad()
            images = images.reshape(-1, 28*28)
                
            # Output of Autoencoder
            reconstructed_output = model(images)
                
            # Calculating the loss function
            loss = loss_MSE(reconstructed_output, images)
            # loss = loss_CE(reconstructed_output,images)
                
            
            loss.backward()
            optimizer.step()
                
            loss_vals.append(loss)
            outputs.append((epoch, images, reconstructed_output,n))
        
        if(epoch%10 == 0):
            print("loss: {}\t at epoch: {}".format(loss,epoch))
    
    return loss_vals,outputs
    
def returnDataset():
    return dataset

def getLatentEmbeddings(x):
    x = x.reshape(-1,28*28)
    latentEmbeddings = model.encoder(x)
    return latentEmbeddings
