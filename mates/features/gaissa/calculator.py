import os, csv
import tensorflow as tf
import tensorflow_hub as hub
# from tensorflow.keras.models import load_model
from tf_keras.models import load_model
from tensorflow.python.framework.convert_to_constants import convert_variables_to_constants_v2_as_graph

def calculateH5(model_path, output_path):
    model = load_model(model_path, custom_objects={'KerasLayer': hub.KerasLayer})

    #### MODEL SIZE (# parameters)
    # Get number of parameters
    num_params = model.count_params()

    #### MODEL FILE SIZE
    model_size_bytes = os.path.getsize(model_path)
    model_size_mb = model_size_bytes / (1024 * 1024)

    #### FLOPS
    # convert tf.keras model into frozen graph to count FLOPS about operations used at inference
    # FLOPS depends on batch size
    inputs = [
        tf.TensorSpec([1] + inp.shape[1:], inp.dtype) for inp in model.inputs
    ]
    real_model = tf.function(model).get_concrete_function(inputs)
    frozen_func, _ = convert_variables_to_constants_v2_as_graph(real_model)

    # Calculate FLOPS with tf.profiler
    run_meta = tf.compat.v1.RunMetadata()
    opts = tf.compat.v1.profiler.ProfileOptionBuilder.float_operation()
    flops = tf.compat.v1.profiler.profile(
        graph=frozen_func.graph, run_meta=run_meta, cmd="scope", options=opts
    )
    
    
    # Writing to CSV
    with open(output_path, 'w', newline='') as file:
        writer = csv.writer(file)

        # Write the header
        writer.writerow(['model_size', 'file_size', 'flops'])

        # Write the values
        writer.writerow([num_params, model_size_mb, flops.total_float_ops])
