'''
how to load .pb tensorflow model: https://leimao.github.io/blog/Save-Load-Inference-From-TF-Frozen-Graph/
https://stackoverflow.com/questions/45697823/single-image-inference-in-tensorflow-python
https://gist.github.com/vishal-keshav/a0a1c0b526a9fd3f0bf6356cac88a23d

'''

import scipy
from scipy import misc
import tensorflow as tf # Default graph is initialized when the library is imported
from tensorflow.python.platform import gfile

with tf.Graph().as_default() as graph: # Set default graph as graph
    with tf.Session() as sess:
        # Load the graph in graph_def
        print("Load graph")
        # We load the protobuf file from the disk and parse it to retrive the unserialized graph_drf
        with gfile.FastGFile("./20210408_attention_ocr_export/saved_model.pb",'rb') as f:
            print("Load Image...")
            # Read the image & get statstics
            image = scipy.misc.imread("./python/testdata/fsns_train_00.png")
            image = image.astype(float)
            Input_image_shape = image.shape
            height, width, channels = Input_image_shape
            print("Plot image...")
            #scipy.misc.imshow(image)

            # Set FCN graph to the default graph
            graph_def = tf.GraphDef()
            graph_def.ParseFromString(f.read())
            sess.graph.as_default()
            # Import a graph_def into the current default Graph (In this case, the weights are (typically) embedded in the graph)
            tf.import_graph_def(
            graph_def,
            input_map=None,
            return_elements=None,
            name="",
            op_dict=None,
            producer_op_list=None
            )
            # Print the name of operations in the session
            for op in graph.get_operations():
                    print("Operation Name :", op.name)         # Operation name
                    print("Tensor Stats :", str(op.values()))     # Tensor name
            # INFERENCE Here
            l_input = graph.get_tensor_by_name('Inputs/fifo_queue_Dequeue:0') # Input Tensor
            l_output = graph.get_tensor_by_name('upscore32/conv2d_transpose:0') # Output Tensor
            print("Shape of input : ", tf.shape(l_input))
            #initialize_all_variables
            tf.global_variables_initializer()
            # Run Kitty model on single image
            Session_out = sess.run(l_output, feed_dict={l_input: image})

