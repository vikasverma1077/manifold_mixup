from networks.model_resnet import Generator, Discriminator

def get_network(z_dim):
    gen = Generator(z_dim)
    disc = Discriminator(sigmoid=False)
    return gen, disc
