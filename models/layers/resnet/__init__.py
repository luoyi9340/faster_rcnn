import utils.conf as conf
conf.append_sys_path("models/layers/resnet")


from models.layers.resnet.models import ResNet34
from models.layers.resnet.models import ResNet50