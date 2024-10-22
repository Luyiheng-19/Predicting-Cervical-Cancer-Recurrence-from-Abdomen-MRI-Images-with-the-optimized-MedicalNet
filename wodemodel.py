import resnet
from collections import OrderedDict
from paddle import nn

import paddle
class _CustomDataParallel(paddle.DataParallel):
    """Custom data parallel class."""

    def __init__(self, model, **kwargs):
        """Instanciate the custom data parallel class."""
        super(_CustomDataParallel, self).__init__(model, **kwargs)

    def __getattr__(self, name):
        """Rewrite the attribute link."""
        try:
            return super(_CustomDataParallel, self).__getattr__(name)
        except AttributeError:
            return getattr(self.module, name)

class MedicalNetClassifier(nn.Layer):
    """Class for a modified ResNet with a classifier as the output."""

    def __init__(self, model_depth:int, checkpoint:dict) -> None:
        """Instanciate a modified ResNet with specified parameters."""
        self.model_depth = model_depth
        self.checkpoint = checkpoint
        self.dim = 512
        super(MedicalNetClassifier, self).__init__()
        if self.model_depth == 10:
            self.model = resnet.resnet10()
        elif self.model_depth == 18:
            self.model = resnet.resnet18()
        elif self.model_depth == 34:
            self.model = resnet.resnet34()
        elif self.model_depth == 50:
            self.dim = 2048
            self.model = resnet.resnet50()
        elif self.model_depth == 101:
            self.dim = 2048
            self.model = resnet.resnet101()
        elif self.model_depth == 152:
            self.dim = 2048
            self.model = resnet.resnet152()
        elif self.model_depth == 200:
            self.dim = 2048
            self.model = resnet.resnet200()
        if self.checkpoint:
            self.model.set_state_dict(self.checkpoint)
        for param in self.model.parameters():
            param.stop_gradient = True
        #Add a classification output
        self.add_sublayer("out_layer", self._output_block(0.2))

    def _output_block(self,dropout):
        """Return an output block to be added at the end of a model."""
        output_block = nn.Sequential(
            ("ad_avg_pool3d", nn.AdaptiveAvgPool3D(1)),
            ("flatten", nn.Flatten()),
            ("out_linear1", nn.Linear(self.dim, int(self.dim/2))),
            ("out_bn1", nn.BatchNorm1D(int(self.dim/2))),
            ("out_relu1", nn.ReLU()),
            ("dropout1",nn.Dropout(dropout)),
            ("out_linear2", nn.Linear( int(self.dim/2), int(self.dim/2/2))),
            ("out_bn2", nn.BatchNorm1D( int(self.dim/2/2))),
            ("out_relu2", nn.ReLU()),
            ("dropout2",nn.Dropout(dropout)),
            ("out_linear3", nn.Linear(int(self.dim/2/2) , 2)),
        )
        return output_block

    def forward(self, x):
        """Forward computations including the new block."""
        out = self.model.forward(x)
        out = self.out_layer(out)
        return out

def generate_model(model_depth:int,  checkpoint=None) -> MedicalNetClassifier:
    """Build an dreturn the selected model."""
    model = MedicalNetClassifier(model_depth, checkpoint)

    return model

if __name__ == "__main__":
    import os
    MODELS_DIR = "/home/aistudio/resnet_101"
    load_layer_state_dict = paddle.load(os.path.join(MODELS_DIR, "%s.pdiparams" % 'resnet_18'))
    model = generate_model(18,checkpoint=load_layer_state_dict)
    params_info = paddle.summary(model, (1,64,128, 128))
    print(params_info)
