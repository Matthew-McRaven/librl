import librl.nn.core
import librl.nn.core.cnn
import librl.nn.classifier

def build_class_net(kind, dims, labels):
    class_kernel = None
    if kind == "mlp": 
        class_kernel = librl.nn.core.MLPKernel(dims, (200, 100))
    elif kind == "cnn":
        conv_list = [
        librl.nn.core.cnn.conv_def(4, 4, 1, 0, 1, False),
        librl.nn.core.cnn.conv_def(4, 4, 1, 0, 1, False),
        librl.nn.core.cnn.pool_def(1, 1, 0, 1, True, 'max'),
        ]
        class_kernel = librl.nn.core.ConvolutionalKernel(conv_list, dims[1:], dims[0])
    else: raise NotImplementedError("I don't understand the network type.")
    return librl.nn.classifier.Classifier(class_kernel, labels)