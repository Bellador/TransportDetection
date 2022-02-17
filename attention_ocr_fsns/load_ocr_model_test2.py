import tensorflow as tf
from tensorflow import keras
from tensorflow.python.framework.convert_to_constants import convert_variables_to_constants_v2
import numpy as np

'''
From: https://github.com/leimao/Frozen-Graph-TensorFlow/blob/8767d1ee8aa5e5b340a52dd0a8208896687c3131/TensorFlow_v2/example_1.py
'''
import tensorflow as tf
from tensorflow import keras
import numpy as np
import scipy
from scipy import misc


def get_fashion_mnist_data():

    fashion_mnist = keras.datasets.fashion_mnist
    (train_images, train_labels), (test_images,
                                   test_labels) = fashion_mnist.load_data()
    class_names = [
        "T-shirt/top", "Trouser", "Pullover", "Dress", "Coat", "Sandal",
        "Shirt", "Sneaker", "Bag", "Ankle boot"
    ]
    train_images = train_images.astype(np.float32) / 255.0
    test_images = test_images.astype(np.float32) / 255.0

    return (train_images, train_labels), (test_images, test_labels)


def wrap_frozen_graph(graph_def, inputs, outputs, print_graph=False):
    def _imports_graph_def():
        tf.compat.v1.import_graph_def(graph_def, name="")

    wrapped_import = tf.compat.v1.wrap_function(_imports_graph_def, [])
    import_graph = wrapped_import.graph

    if print_graph == True:
        print("-" * 50)
        print("Frozen model layers: ")
        layers = [op.name for op in import_graph.get_operations()]
        for layer in layers:
            print(layer)
        print("-" * 50)

    return wrapped_import.prune(
        tf.nest.map_structure(import_graph.as_graph_element, inputs),
        tf.nest.map_structure(import_graph.as_graph_element, outputs))


def main():
    # Load frozen graph using TensorFlow 1.x functions
    with tf.io.gfile.GFile("./20210408_attention_ocr_export/saved_model.pb", "rb") as f:
        graph_def = tf.compat.v1.GraphDef()
        graph_def.ParseFromString(f.read())
    # Wrap frozen graph to ConcreteFunctions
    frozen_func = wrap_frozen_graph(graph_def=graph_def,
                                    inputs=["x:0"],
                                    outputs=["Identity:0"],
                                    print_graph=True)
    print("-" * 50)
    print("Frozen model inputs: ")
    print(frozen_func.inputs)
    print("Frozen model outputs: ")
    print(frozen_func.outputs)

    # Get predictions for test images
    test_image = scipy.misc.imread("./python/testdata/fsns_train_00.png")
    test_image = test_image.astype(np.float32) / 255.0
    frozen_graph_predictions = frozen_func(x=tf.constant(test_image))[0]
    # Print the prediction for the first image
    print("-" * 50)
    print("Example TensorFlow frozen graph prediction reference:")
    print(frozen_graph_predictions[0].numpy())
    # The two predictions should be almost the same.
    assert np.allclose(a=frozen_graph_predictions[0].numpy(), b=1, rtol=1e-05, atol=1e-08, equal_nan=False)

if __name__ == "__main__":
    main()