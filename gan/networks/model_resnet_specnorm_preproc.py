from networks.model_resnet_preproc import (Generator,
                                           Discriminator)

def get_network(z_dim):
    gen = Generator(z_dim)
    disc = Discriminator(spec_norm=True,
                         sigmoid=True)
    return gen, disc

if __name__ == '__main__':
    a1, a2 = get_network(62)
    print(a1)
    print(a2)
