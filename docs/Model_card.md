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


### Out-of-Scope Use

A different use of this model could be using it for tasks like object detection or image segmentation.

## Bias, Risks and Limitations

Image classification models are powerful tools but come with challenges. These models can develop biases if the data used to train them is not representative of the general population, or if the algorithms themselves are inherently biased. 

Furthermore, models have limitations such as difficulty generalizing to new data, lack of interpretability, and vulnerability to adversarial attacks. This means that models may not perform well on data that is different from what they were trained on, it can be difficult to understand why a model made a particular decision, and models can be tricked into misclassifying images.

### Recommendations

To address these issues, it is crucial to use high-quality training data, employ bias reduction techniques, and conduct thorough evaluations of model fairness. Additionally, fostering transparency and collaboration among different disciplines is essential.

## How to Get Started with the Model

Use the code below to get started with the model.

```
# Apply image detector on a single image.
detector = hub.Module("https://kaggle.com/models/google/mobilenet-v2/frameworks/TensorFlow1/variations/openimages-v4-ssd-mobilenet-v2/versions/1")
detector_output = detector(image_tensor, as_dict=True)
class_names = detector_output["detection_class_names"]

```

## Training Details

The MobileNet V2 feature extractor was trained on [Open Images V4](https://storage.googleapis.com/openimages/web/index.html).

The checkpoint exported into this model was ```mobilenet_v2_1.3_224/mobilenet_v2_1.3_224.ckpt``` downloaded from [MobileNet V2 pre-trained models](https://github.com/tensorflow/models/blob/master/research/slim/nets/mobilenet/README.md). Its weights were originally obtained by training on the ILSVRC-2012-CLS dataset for image classification ("Imagenet").


### Training Data

More specifically, this model card has been completed by extracting results from running the model with the data from [Stanford Dogs Dataset](http://vision.stanford.edu/aditya86/ImageNetDogs/).

The training set from this data contains the 49.67% of the total data, which are approximately 10223 images.

### Training Procedure


#### Training Hyperparameters

Different batch configurations have been built as well as various optimization functions. 
The two batches are ones using a 32 batch and a 64 batch and the optimizers considered are *Rmsprop*, *Adam*, *AdamW* and *SGD*.

The chosen combination of batch and optimizer is **Batch 32 using AdamW**, as it is the one with best accuracy.


## Evaluation

### Testing Data, Factors & Metrics

#### Testing Data

The data set corresponding to test contains 50.33% of the total images. This means that approximately 10358 images have been used for this part.

#### Factors

The factors taken into account for the model evaluation are *Accuracy* and *Loss*.

#### Metrics

The metrics considered when evaluating the model can be divided into two main categories. The first one refers to model performance while the second is used to environmental. These last will be addressed in the section below.

The performance metrics considered are *Train accuracy*, *Train validation accuracy*, *Train validation loss* and *Train loss*.

### Results

Firstly, between the Batch 32 and Batch 64 configurations, we see that, overall, Batch 32 configurations show better results. Because of this, we will do a further comparison just between them.

The Batch 32 AdamW configuration appears to strike an optimal balance between accuracy and
emissions, making it a strong candidate for deployment. Even though it does not have the best numbers when it comes to RAM power, the difference is so low that can be almost ignored. Given the low emissions and energy metrics, this configuration is environmentally sustainable, allowing for their use in scenarios where energy efficiency is paramount.

#### Summary

The table below, summarizes all the performance metrics considered when evaluating the model. As said above, considering only the different configurations of Batch 32.

| Metric | Rmsprop | AdamW | SGD | Adam |
| ------ | ----- | ----- | ----- | ----- |
|Duration (min) | 9.7 | 9.7| 10.2 | 13.7 |
|Train validation accuracy | 0.8194 | 0.8230| 0.7982 | 0.8236 |
|Train accuracy | 0.9713| 0.9943| 0.8485 | 0.9993 |
|Train validation loss | 0.5966 | 0.5694| 0.8626 | 0.5682 |
|Train loss | 0.1097 | 0.0732| 0.7642 | 0.0414 |


## Environmental Impact

The next table shows the results of the environmental metrics avaluated. Again, the focus is on Batch 32.

| Metric            | Rmsprop  | AdamW    | SGD     | Adam    |
|-------------------|----------|----------|---------|---------|
| Emissions (kg)     | 0.000391 | 0.000390 | 0.000412| 0.000560|
| Emissions Rate     | 7.07 × 10⁻⁷ | 7.07 × 10⁻⁷ | 7.07 × 10⁻⁷ | 7.07 × 10⁻⁷ |
| CPU Power (W)      | 14       | 14       | 14      | 14      |
| RAM Power (W)      | 0.6395   | 0.6519   | 0.6487  | 0.6556  |
| Energy Consumed (kWh) | 0.00225 | 0.00224 | 0.00237 | 0.00322 |



Carbon emissions can be estimated using the [Machine Learning Impact calculator](https://mlco2.github.io/impact/#compute) presented in [Lacoste et al. (2019)](https://arxiv.org/abs/1910.09700).

- **Hardware Type**:

CPU count = 8

CPU Model: 11th Gen Intel(R) Core(TM) i5-1155G7 @ 2.50GHz

- **Hours used**: Looking at the table shown in the evaluation summary, we estimate that the Batch 32 used 43.3 minutes. Adding the 38.2 minutes used by the Batch 64 we get that the total used hours are **less than 1.5h**
- **Cloud Provider**: NaN
- **Compute Region**: NaN
- **Carbon Emitted**: 0.390g


## Citation

Kaggle link of the model: https://www.kaggle.com/models/google/mobilenet-v2

Model card template: https://github.com/huggingface/huggingface_hub/blob/main/src/huggingface_hub/templates/modelcard_template.md


## Model Card Contact

Noa Mediavilla - noa.mediavilla@estudiantat.upc.edu

Matilde Simões - matilde.simoes@estudiantat.upc.edu

Andrea Tomás - andrea.tomas@estudiantat.upc.edu

Maria Tubella - maria.tubella@estudiantat.upc.edu

Juan Pablo Zaldívar - juan.pablo.zaldivar@estudiantat.upc.edu