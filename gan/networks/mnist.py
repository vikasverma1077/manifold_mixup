from christorch.gan.architectures import gen
from networks import base # importlib works from root dir

def get_network(z_dim):
    gen_fn = gen.generator(input_width=28,
                           input_height=28,
                           output_dim=1,
                           z_dim=z_dim)
    disc_fn = base.discriminator(input_width=28,
                            input_height=28,
                            input_dim=1,
                            output_dim=1,
                            out_nonlinearity='sigmoid')
    return gen_fn, disc_fn
