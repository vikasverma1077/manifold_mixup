from networks.model_resnet import Generator, Discriminator

def get_network(z_dim):
    gen = Generator(z_dim)
    disc = Discriminator(spec_norm=True, sigmoid=False)
    return gen, disc

if __name__ == '__main__':
    dd, gg = get_network(z_dim=62)
    print(dd)
    print(gg)
