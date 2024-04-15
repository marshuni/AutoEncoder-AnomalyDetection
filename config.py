z_dim = 64

batch_size = 16
num_epochs = 10
learning_rate = 4.0e-4

cuda = True

CKPT_PATH = "./checkpoint/model.pt"           # 修改

capsule_label = {
    'good':0,
    'crack':1,
    'squeeze':2,
    'poke':3,
    'faulty_imprint':4,
    'scratch':5,
    'bad':6
}