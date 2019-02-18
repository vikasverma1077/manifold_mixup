"""The models subpackage contains definitions for the following model
architectures:
-  `ResNeXt` for CIFAR10 CIFAR100
You can construct a model with random weights by calling its constructor:
.. code:: python
    import models
    resnext29_16_64 = models.ResNeXt29_16_64(num_classes)
    resnext29_8_64 = models.ResNeXt29_8_64(num_classes)
    resnet20 = models.ResNet20(num_classes)
    resnet32 = models.ResNet32(num_classes)


.. ResNext: https://arxiv.org/abs/1611.05431
"""

from .resnext import resnext29_8_64, resnext29_16_64
#from .resnet import resnet20, resnet32, resnet44, resnet56, resnet110
from .resnet import resnet18, resnet34, resnet50, resnet101, resnet152
from .preresnet import preactresnet18, preactresnet34, preactresnet50, preactresnet101, preactresnet152
#from .preact_resnet_temp import preactresnet18, preactresnet34, preactresnet50, preactresnet101, preactresnet152
from .caffe_cifar import caffe_cifar
from .densenet import densenet100_12,densenet100_24
from .wide_resnet import wrn28_10, wrn28_2

#from .imagenet_resnet import resnet18, resnet34, resnet50, resnet101, resnet152
