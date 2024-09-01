import onnx
from onnx_tf.backend import prepare
from tensorflow.keras import layers, models


onnx_model = onnx.load("ppe1.onnx")
tf_rep = prepare(onnx_model)
tf_rep.export_graph("ppe1.pb")
