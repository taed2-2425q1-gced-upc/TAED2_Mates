# Dataset Card for Stanford Dogs Dataset

## Dataset Description

- **Homepage**: http://vision.stanford.edu/aditya86/ImageNetDogs/
- **Repository**:
- **Paper**:

Primary: 

 Aditya Khosla, Nityananda Jayadevaprakash, Bangpeng Yao and Li Fei-Fei. Novel dataset for Fine-Grained Image Categorization. First Workshop on Fine-Grained Visual Categorization (FGVC), IEEE Conference on Computer Vision and Pattern Recognition (CVPR), 2011.  [[pdf]](https://people.csail.mit.edu/khosla/papers/fgvc2011.pdf) [[poster]](http://vision.stanford.edu/documents/KhoslaJayadevaprakashYaoFeiFei_FGVC2011.pdf) [[BibTex]](http://vision.stanford.edu/bibTex/KhoslaJayadevaprakashYaoFeiFei_FGVC2011.bib)

Secondary:

 J. Deng, W. Dong, R. Socher, L.-J. Li, K. Li and L. Fei-Fei, ImageNet: A Large-Scale Hierarchical Image Database. IEEE Computer Vision and Pattern Recognition (CVPR), 2009. 



- **Leaderboard**:
- **Point of contact**: aditya86@cs.stanford.edu, bangpeng@cs.stanford.edu

### Dataset Summary

The Stanford Dogs dataset contains images of 120 breeds of dogs from around the world. This dataset has been built using images and annotation from ImageNet for the task of fine-grained image categorization.

### Supported Tasks and Leaderboards

### Languages
American English

## Dataset Structure

### Data Instances

There are a total of 20,580 images; ~150 images per class. Moreover, there is an annotation for each instance (image).

The list of breeds is as follows:

```
affenpinscher
afghan_hound
african_hunting_dog
airedale
american_staffordshire_terrier
appenzeller
australian_terrier
basenji
basset
beagle
bedlington_terrier
bernese_mountain_dog
black-and-tan_coonhound
blenheim_spaniel
bloodhound
bluetick
border_collie
border_terrier
borzoi
boston_bull
bouvier_des_flandres
boxer
brabancon_griffon
briard
brittany_spaniel
bull_mastiff
cairn
cardigan
chesapeake_bay_retriever
chihuahua
chow
clumber
cocker_spaniel
collie
curly-coated_retriever
dandie_dinmont
dhole
dingo
doberman
english_foxhound
english_setter
english_springer
entlebucher
eskimo_dog
flat-coated_retriever
french_bulldog
german_shepherd
german_short-haired_pointer
giant_schnauzer
golden_retriever
gordon_setter
great_dane
great_pyrenees
greater_swiss_mountain_dog
groenendael
ibizan_hound
irish_setter
irish_terrier
irish_water_spaniel
irish_wolfhound
italian_greyhound
japanese_spaniel
keeshond
kelpie
kerry_blue_terrier
komondor
kuvasz
labrador_retriever
lakeland_terrier
leonberg
lhasa
malamute
malinois
maltese_dog
mexican_hairless
miniature_pinscher
miniature_poodle
miniature_schnauzer
newfoundland
norfolk_terrier
norwegian_elkhound
norwich_terrier
old_english_sheepdog
otterhound
papillon
pekinese
pembroke
pomeranian
pug
redbone
rhodesian_ridgeback
rottweiler
saint_bernard
saluki
samoyed
schipperke
scotch_terrier
scottish_deerhound
sealyham_terrier
shetland_sheepdog
shih-tzu
siberian_husky
silky_terrier
soft-coated_wheaten_terrier
staffordshire_bullterrier
standard_poodle
standard_schnauzer
sussex_spaniel
tibetan_mastiff
tibetan_terrier
toy_poodle
toy_terrier
vizsla
walker_hound
weimaraner
welsh_springer_spaniel
west_highland_white_terrier
whippet
wire-haired_fox_terrier
yorkshire_terrier

```

This is an example of an annotation of an image of class chihuahua:

```
<annotation>
	<folder>02085620</folder>
	<filename>n02085620_7</filename>
	<source>
		<database>ImageNet database</database>
	</source>
	<size>
		<width>250</width>
		<height>188</height>
		<depth>3</depth>
	</size>
	<segment>0</segment>
	<object>
		<name>Chihuahua</name>
		<pose>Unspecified</pose>
		<truncated>0</truncated>
		<difficult>0</difficult>
		<bndbox>
			<xmin>71</xmin>
			<ymin>1</ymin>
			<xmax>192</xmax>
			<ymax>180</ymax>
		</bndbox>
	</object>
</annotation>
```

On the other hand, this is an example of an image of this same class:

![Chihuahua](example_image.jpg)

Each image has a filename that is its unique ```id```. 

### Data Fields

Images and annotations for each of the classes.

### Data Splits

- ```train``` - the training set, where we are provided the breed for these dogs.
- ```test``` - the test set, we must predict the probability of each breed for each image.
- ```annotations``` - the breeds for the images in the train set.
- ```images``` - images of the dogs used in both sets.

## Dataset Creation

### Curation Rationale

### Source Data

#### Initial Data Collection and Normalization

#### Who are the source language producers?

### Annotations

#### Annotation process

#### Who are the annotators?

### Personal and Sensitive Information

## Considerations for Using the Data

### Social Impact of Dataset

### Discussion of Biases

### Other Known Limitations

## Additional Information

### Dataset Curators

- Aditya Khosla (aditya@pathai.com, 
khosla@csail.mit.edu) 
- Nityananda Jayadevaprakash     
- Bangpeng Yao     
- Li Fei-Fei (feifeili@cs.stanford.edu)

### Licensing Information

[Attribution 4.0 International (CC BY 4.0)](https://creativecommons.org/licenses/by/4.0/)

### Citation Information

Dataset card template: https://github.com/huggingface/datasets/blob/main/templates/README_guide.md

### Contributors