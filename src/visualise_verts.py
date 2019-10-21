import sys
import os

import random
import torch
import numpy as np

def load_model(ae_path, seed=1, device='cpu'):
    from autoencoder import Autoencoder

    # load autoencoder model
    base_path = os.path.join(ae_path, 'log', 'mnist')
    data_path = os.path.join(ae_path, 'data')

    dataset_name = 'mnist'
    net_name = 'mnist_LeNet'
    rep_dim = 8
    optimizer_name = 'adam'
    lr = 0.0001
    n_epochs = 150
    lr_milestone = 50
    batch_size = 128
    weight_decay = 0.5e-3
    n_jobs_dataloader = 0

    # set paths
    xp_path = os.path.join(base_path, 'seed_' + str(seed))
    load_model = os.path.join(xp_path, 'model.tar')

    print('Loading model from {}s.'.format(load_model))

    # set seed
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True

    # initialize Autoencoder model and set network
    ae_model = Autoencoder()
    ae_model.set_network(net_name, rep_dim=rep_dim)

    # load autoencoder model (network weights)
    ae_model.load_model(model_path=load_model, device=device)

    return ae_model

def decode(model, verts, device='cpu'):

    vert_batch = torch.from_numpy(verts).float().to(device)
    model.net = model.net.to(device)

    # decode
    rec = model.net.decoder(vert_batch)

    return rec

if __name__ == "__main__":
    assert len(sys.argv) == 2

    DEVICE = 'cpu'
    SEED = 1

    script_file = os.path.abspath(sys.argv[0])
    vertices_file = os.path.abspath(sys.argv[1])

    AE_PATH = os.path.join(os.path.dirname(script_file), os.pardir, "AE-PyTorch")

    sys.path.append(os.path.join(AE_PATH, "src"))

    # load vertices
    verts = np.loadtxt(vertices_file, delimiter=",")

    if verts.shape[0] > 1000:
        random.seed(1234)
        idcs = random.randint(0, verts.shape[0]-1, 1000)
        verts = verts[idcs]

    #print(verts[[1,3]])

    ae_model = load_model(AE_PATH, seed=SEED, device=DEVICE)
    rec = decode(ae_model, verts, device=DEVICE)

    from utils.visualization.plot_images_grid import plot_images_grid

    fn = os.path.basename(vertices_file)[0:-3]

    export_img = os.path.join(os.path.dirname(vertices_file), fn)

    reconstructed_verts = rec.detach().numpy()[:, 0, :, :]
    print(reconstructed_verts.shape)

    np.savetxt(vertices_file+"_rec", reconstructed_verts.reshape(reconstructed_verts.shape[0], 28*28), delimiter=",")
    print("Reconstructed points written to {}".format(vertices_file+"_rec"))

    plot_images_grid(rec.detach(),
                     export_img=export_img,
                     nrow=50, padding=2)

    print("Reconstructed images written to {}.png".format(export_img))
