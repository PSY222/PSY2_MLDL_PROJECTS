# Competitions Overview
Lymph node metastasis takes a critical role in the spread of cancer. Treatment and progress of the cancer heavily depends on lymph node metastasis. Thus, Yonsei medical school hosted the competition to build an AI model which decides lymph node metastasis based on the breast cancer pathology slide.


![image](https://user-images.githubusercontent.com/86555104/209470159-4b44a21d-c4a7-4bf3-8fc1-cf424e537bf3.png)
- example image of the pathology slide_irrelevant from the training image

# Data Overview
train data(structured data) :(1000,23)
train images: 1000
test data(structured data) : (250,22)
test images: 250

# Approach
I planned for the approach in two-stage. The first stage is making a prediction with only using non-image data (structured data). Then, I separately built CNN image classification model only with the image data. 

![image](https://user-images.githubusercontent.com/86555104/209469670-4a9f6c38-5a1c-4ea7-9d8c-f455268818ba.png)

The biggest issue in the structured dataset was severe a null ratio of the data. 15 columns out of 23 features had null values ranging from 0.1% up to 90%. Columns with 80-90% were dropped, but I had to be cautious with the data with 20-50% of null ratio data which can lose the information. After checking overlapped null values over the several columns with strong correlation, I was able to discover the reasonable substitute value for the NaN. (For example, 82 nan values of 'size' has common 78 'nan' values with HG, HG_score_1, HG_score_2,HG_score3)

I initially trained various kinds of model to find the base models suitable for the ensemble. LogisticRegression, CatBoostClassifier, SVC(Support Vector Classifier), RandomForestClassifierm ExtraTreeClassifier showed fair performance. These models were retrained with the hyperparameters resulted from GridSearchCV. After testing both soft voting and hard voting method, hard voting showed slightly better performance.
<hr>

Dealing with pathology slide was new to me, that I looked through many pathology, biology image Kaggle competition codes and research papers. "Tailoring automated data augmentation to H&E-stained histopathology" "A generalized deep learning framework for whole-slide image segmentation and analysis" were particularly helpful to find out what kind of methodologies researchers use for the pathology slide data augmentation and the structure of CNN model for the prediction. "Tailoring automated data augmentation to H&E-stained histopathology" was very resourceful providing tensorflow codes in github. I really appreciate Laura Fink's work from Kaggle which helped me organize the data generator for the model.

To perform the data augmentation of the image, I conducted some experiments prior to building a ImagePreprocessor class. "Tailoring automated data augmentation to H&E-stained histopathology" provided recommended magnitude range of data augmentation on github which helped me adjust the range of transforms.


![image](https://user-images.githubusercontent.com/86555104/209470513-8597a08d-1c30-4621-8f6a-2d1bb489d39a.png)

To preserve the height width ratio of the image and minimize the data loss, I calculated max_h, max_w, min_h, min_w of the image size. My first attempt of the resizing was (1300,2800) = (min_h, min_w), however, due to computational limit, I had to shrink the image down to (260,520).

My CNN model mainly influenced by CNN model of "Tailoring automated data augmentation to H&E-stained histopathology". The research paper used multiple convolution layers while doubling the filters in every two steps . <br>
<br>
<img src ="https://user-images.githubusercontent.com/86555104/209471572-2c24f880-62f7-481e-8543-9c2546acf802.png" width=300 height =300>

    def build_model(self):
        self.model = Sequential()
        self.model.add(Conv2D(32, kernel_size = 3, activation='relu', input_shape = self.input_shape))
        #self.model.add(Conv2D(32, kernel_size = 3, strides=(2,2),activation='relu'))
        #self.model.add(Conv2D(64, kernel_size = 3, activation='relu'))
        #self.model.add(Conv2D(64, kernel_size = 3, strides=(2,2),activation='relu'))
        #self.model.add(MaxPool2D(pool_size=(2,2),padding='valid'))
        #self.model.add(Conv2D(128, kernel_size = 3, activation='relu'))
        self.model.add(Conv2D(128, kernel_size = 3, activation='relu'))
        #self.model.add(Conv2D(256, kernel_size = 3, activation='relu'))
        #self.model.add(MaxPool2D(pool_size=(2,2),padding='valid'))
        #self.model.add(Conv2D(256, kernel_size = 3, activation='relu'))
        #self.model.add(Conv2D(512, kernel_size = 14, activation='relu'))
        #self.model.add(Dropout(0.6))
        self.model.add(GlobalAveragePooling2D())
        self.model.add(Dense(512,activation="relu"))
        self.model.add(Dense(1, activation='sigmoid'))

However, the limitation of this trial was that image size was too big that I had to add additional pooling layer and freeze most of the Convolution layers during the training.

Nevertheless, this competition was meaningful for me to learn how to deal with image data and approach the problems based on the findings from the prior research. In the next computer vision competition, I am looking forward to improve my model by trying new models together.


# Reference Sources 
These are links of sources that helped me understand and handle the image better
https://www.kaggle.com/code/allunia/protein-atlas-exploration-and-baseline
https://github.com/DIAGNijmegen/pathology-he-auto-augment
https://www.kaggle.com/iafoss/panda-concat-tile-pooling-starter-0-79-lb
https://www.kaggle.com/haqishen/train-efficientnet-b0-w-36-tiles-256-lb0-8714b-4e22-b6af-0ad0e6d8c74c.png)
