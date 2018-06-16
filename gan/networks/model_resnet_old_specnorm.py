from networks.model_resnet_old import Generator, Discriminator

def get_network(z_dim):
    gen = Generator(z_dim)
    disc = Discriminator(spec_norm=True, sigmoid=True)
    return gen, disc
