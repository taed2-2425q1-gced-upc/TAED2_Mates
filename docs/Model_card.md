# Model card for mobilenet_v2

## Model Details

### Model description

- **Developed by**: Google, TensorFlow
- **Shared by**: Kaggle
- **Model type**: SSD-based object detection model
- **Language(s) (NLP)**:
- **License**: Apache 2.0
- **Finetuned from model (opt)**:


### Model Sources

The model is [publicly available](https://github.com/tensorflow/models/blob/master/research/object_detection/samples/configs/faster_rcnn_inception_resnet_v2_atrous_oid.config) as part of [TensorFlow Object Detection API](https://github.com/tensorflow/models/tree/master/research/object_detection). The MobileNet V2 feature extractor was trained on ImageNet and fine-tuned with SSD head on [Open Images V4 dataset](https://storage.googleapis.com/openimages/web/index.html), containing 600 classes.

- **Repository**: https://github.com/tensorflow/tfhub.dev/tree/master/assets/docs/google
- **Paper (opt)**:
- **Demo (opt)**:


## Uses

### Direct use

This is a [SavedModel in TensorFlow 2 format](https://www.tensorflow.org/hub/tf2_saved_model). Using it requires TensorFlow 2 (or 1.15) and TensorFlow Hub 0.5.0 or newer.

This model can be used with the ```hub.KerasLayer``` as follows. It *cannot* be used with the ```hub.Module``` API for TensorFlow 1.

```
import tensorflow_hub as hub
m = tf.keras.Sequential([
    hub.KerasLayer("https://www.kaggle.com/models/google/mobilenet-v2/TensorFlow2/130-224-feature-vector/2",
                   trainable=False),  # Can be True, see below.
    tf.keras.layers.Dense(num_classes, activation='softmax')
])
m.build([None, 224, 224, 3])  # Batch input shape.

```
#### Inputs

The input ```images``` are expected to have color values in the range [0,1], following the [common image](https://www.tensorflow.org/hub/common_signatures/images#input) input conventions. For this model, the size of the input images is fixed to ```height``` x ```width``` = 224 x 224 pixels.

#### Outputs

The output is a batch of feature vectors. For each input image, the feature vector has size ```num_features``` = 1664.

### Downstream Use (opt)

### Out-of-Scope Use

## Bias, Risks and Limitations

### Recommendations

## How to Get Started with the Model

Use the code below to get started with the model.

```
# Apply image detector on a single image.
detector = hub.Module("https://kaggle.com/models/google/mobilenet-v2/frameworks/TensorFlow1/variations/openimages-v4-ssd-mobilenet-v2/versions/1")
detector_output = detector(image_tensor, as_dict=True)
class_names = detector_output["detection_class_names"]

```

## Training Details

### Training Data

The MobileNet V2 feature extractor was trained on [Open Images V4](https://storage.googleapis.com/openimages/web/index.html).

The checkpoint exported into this model was ```mobilenet_v2_1.3_224/mobilenet_v2_1.3_224.ckpt``` downloaded from [MobileNet V2 pre-trained models](https://github.com/tensorflow/models/blob/master/research/slim/nets/mobilenet/README.md). Its weights were originally obtained by training on the ILSVRC-2012-CLS dataset for image classification ("Imagenet").

### Training Procedure

#### Preprocessing (opt)

#### Training Hyperparameters

Four different batch configurations have been built as well as various optimization functions. The resulting combinations are:

- ```Batch 32```
- ```Batch 64```
- ```Batch 32``` with ```Adam``` optimizer
- ```Batch 32``` with ```Rmsprop``` optimizer

#### Speeds, Sizes, Times (opt)

## Evaluation

### Testing Data, Factors & Metrics

#### Testing Data

#### Factors

#### Metrics

The metrics considered when evaluating the model can be divided into two main categories. The first one refers to model performance while the second is used to environmental. These last will be addressed in the section below.

The performance metrics considered are *Train accuracy*, *Train validation accuracy*, *Train validation loss* and *Train loss*.

### Results

The Batch 32 Rmsprop configuration appears to strike an optimal balance between accuracy and
emissions, making it a strong candidate for deployment. In contrast, Batch 64 shows potential for
slightly better RAM usage but does not offer significant improvements in accuracy, suggesting that
adjustments to batch size need to be made cautiously based on specific use-case requirements.
The consistent CPU power consumption across batches indicates that further exploration of GPU
utilization could enhance performance, particularly if large datasets or complex models are consid-
ered in future experiments. Given the low emissions and energy metrics, these configurations are
environmentally sustainable, allowing for their use in scenarios where energy efficiency is paramount.

#### Summary

The table below, summarizes all the performance metrics considered when evaluating the model.

| Metric | Batch 32 |Batch 64 | Batch 32 Adam | Batch 32 Rmsprop |
| ------ | ----- | ----- | ----- | ----- |
|Train accuracy | 0.28498 | 0.26010| 0.26010 | 0.28861 |
|Train validation accuracy | 0.39909 | 0.39517| 0.39517 | 0.41669 |
|Train loss | 3.16361 | 3.33960| 3.33960 | 3.12644 |
|Train validation loss | 3.16361 | 3.33960| 3.33960 | 3.12644 |

## Model Examination (opt)

## Environmental Impact

The next table shows the results of the environmental metrics avaluated.

| Metric | Batch 32 |Batch 64 | Batch 32 Adam | Batch 32 Rmsprop |
| ------ | ----- | ----- | ----- | ----- |
| Emissions | 0.00023 | 0.00011 | 0.00011 | 0.00025 |
| Emissions rate | 6.98e-07 | 6.99e-07 | 6.99e-07 | 6.98e-07 |
| CPU power | 14.0000 | 14.0000 | 14.0000 | 14.0000 |
| GPU power | 0.00000 | 0.00000 | 0.00000 | 0.00000 |
| RAM power | 0.46661 | 0.49706 | 0.49706 | 0.45724 |
| CPU energy | 0.00127 | 0.00064 | 0.00064 | 0.00137 |
| GPU energy | 0.00000 | 0.00002 | 0.00002 | 0.00004 |
| RAM energy | 0.00131 | 0.00066 | 0.00066 | 0.00142 |
| Energy consumed | 2.36840 | 2.35004 | 2.35004 | 2.30601 |


Carbon emissions can be estimated using the [Machine Learning Impact calculator](https://mlco2.github.io/impact/#compute) presented in [Lacoste et al. (2019)](https://arxiv.org/abs/1910.09700).

- **Hardware Type**:
- **Hours used**:
- **Cloud Provider**:
- **Compute Region**:
- **Carbon Emitted**: Different experiments have been run and the results regarding the emissions can be seen in the following plot.
![Carbon emissions](emissions_image.png)

We can see that the experiment with the highest emission is the one using the Adam optimizer.

## Technical Specifications (opt)

### Model Architecture and Objective

### Compute Infrastructure

#### Hardware

#### Software

## Citation

Kaggle link of the model: https://www.kaggle.com/models/google/mobilenet-v2

Model card template: https://github.com/huggingface/huggingface_hub/blob/main/src/huggingface_hub/templates/modelcard_template.md

### BibTeX

### APA

## Glossary (opt)

## More information (opt)

## Model Card Authors (opt)

## Model Card Contact