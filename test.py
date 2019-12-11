import torch

from flags import use_cuda
from models import EfficientDet

if __name__ == '__main__':
    if use_cuda:
        inputs = torch.randn(5, 3, 512, 512).cuda()
    else:
        inputs = torch.randn(5, 3, 512, 512)

    model = EfficientDet(num_classes=2, is_training=False)
    if use_cuda:
        model = model.cuda()
    output = model(inputs)
    for p in output:
        print(p.size())
