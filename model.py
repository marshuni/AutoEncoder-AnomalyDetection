from torch import nn

class Autoencoder_Teacher(nn.Module):
    def __init__(self,z_dim):
        super(Autoencoder_Teacher, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(64 * 64, 1024),
            nn.ReLU(True),
            nn.Linear(1024, 128),
            nn.ReLU(True),
            nn.Linear(128, z_dim))

        self.decoder = nn.Sequential(
            nn.Linear(z_dim, 128),
            nn.ReLU(True),
            nn.Linear(128, 1024),
            nn.ReLU(True),
            nn.Linear(1024, 64 * 64),
            nn.Tanh()
        )

    def forward(self, x):
        z = self.encoder(x)
        xhat = self.decoder(z)
        return xhat
    
class Autoencoder_Student(nn.Module):
    def __init__(self,z_dim):
        super(Autoencoder_Student, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(64 * 64, 768),
            nn.ReLU(True),
            nn.Linear(768, z_dim))

        self.decoder = nn.Sequential(
            nn.Linear(z_dim, 768),
            nn.ReLU(True),
            nn.Linear(768, 64 * 64),
            nn.Tanh()
        )

    def forward(self, x):
        z = self.encoder(x)
        xhat = self.decoder(z)
        return xhat