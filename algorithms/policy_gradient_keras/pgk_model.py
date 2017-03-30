from library.base_model import BaseModel
from library.utils import Utils
from library.layers import Layers
from library.loss import Loss
from library.optimizers import Optimizers


class AgentModel(BaseModel):
    def build_model(self):
        self.state
        self.layers
        self.output
        self.loss
        self.compute_gradients

    @define_scope
    def state(self):
        return Utils.placeholder(shape=self.cfg.state_size)

    @define_scope
    def layers(self):
        layers = []
        last_layer = self.state
        for shape in self.cfg.hidden_layers:
            last_layer = Layers.fully_connected(
                shape=shape,
                activation='elu',
                init='glorot_uniform',
                input=last_layer
            )
            layers.append(last_layer)
        return layers

    @define_scope
    def output(self):
        last_layer = self.state if len(self.layers) == 0 else self.layers[-1]
        return Layers.fully_connected(
            shape=self.cfg.action_size,
            activation='softmax',
            init='glorot_uniform',
            input=last_layer
        )

    @define_scope
    def loss(self):
        return Loss.simple_loss(self.output)

    @define_scope
    def compute_gradients(self):
        return Gradients.compute_gradients(self.loss)


class ParameterServerModel(BaseModel):
    def build_model(self):
        self.weights
        self.apply_gradients

    @define_scope
    def gradients(self):
        return [
            Utils.placeholder(shape=shape) for shape in self.cfg.hidden_layers
        ]

    @define_scope
    def weights(self):
        return [
            Utils.weights(shape=shape) for shape in self.cfg.hidden_layers
        ]

    @define_scope
    def apply_gradients(self):
        adam = Optimizers.adam(self.cfg.learning_rate)
        return adam.apply_gradients(self.weights, self.gradients)
