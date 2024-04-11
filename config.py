z_dim = 64
cuda = True


batch_size = 16

num_epochs_student = 10
learning_rate_student = 1.0e-4
CKPT_PATH = "./checkpoint/model_capsule.pt"

num_epochs_teacher = 10
learning_rate_teacher = 1.0e-4
CKPT_PATH_TEACHER = "./checkpoint/model_teacher_capsule.pt"

capsule_label = {
    'good':0,
    'crack':1,
    'squeeze':2,
    'poke':3,
    'faulty_imprint':4,
    'scratch':5,
    'bad':6
}