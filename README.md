# Final Project: Sports Medicine through Video Classification 

![Tennis](https://github.com/ffidai/Final_Project/blob/master/MD/tennis.jpg)

## Background

Sports medicine is a branch of medicine that deals with physical fitness and the treatment and prevention of injuries related to sports and exercise. Although most sports teams have employed team physicians for many years, it is only since the late 20th century that sports medicine has emerged as a distinct field of health care.

Ideally, sports video classification will identify common errors during practice and correct them, supplementing a physical coach.

* Shoulder Impingement
    * overhead shots and serving
* Calf tears
    * planting the foot down 
* Tennis Elbow
    * repetitive movements including backhand strokes

Good technique and racquet grip is important to reduce the risk of the injury developing. In all of the above, it’s important to note that correct technique and coaching for tennis and racquet sports is very important! All the above necessitates granular and processed data on every muscle movement of player to be recorded and analyzed.

## Goal

The goal of our project is to expand on the image classification model and utilize video classification to differentiate between various sports movements. Ideally, the model can be used to help athletes, physical therapists, and doctors better understand how sports movements affects the body. An unintended use for this model can be to predict players' behavior and movements which can be used to help teams/sports insititutions select team rosters and viewers/fans place players in fanatasy leagues and legal bets. 

## The Process & Code Highlights
As a start, we will look at two basic tennis movements: forehand and backhand. In order to acheieve this, we will create a convulutional neural network (CNN). 

### Volumetrics: Creating a digestible image/video file through trial and error
* Data
    * Pro Right Hander, Pro Left Hander, Non Pro right hander
* 2 clippings: backhand, forehand
    * ~500 frames each
    * 500 * 224 * 224 * 3 = 75 m * 2 clippings = 150 m data points  * 3 players = 450 m data points
    * ~1 GB in size
* Model
    * 1, 5 , 25 and 50 epochs
    * 7, 35, 175 and 360 mins to train and save each model

### Train: Creating a key for tennis movements
In order to create a successful video classifier, we had to create sample videos to recognize the two different movements and then use test videos to see if our code is sound. The training video recorded the two movements separately; the test video recorded the two movements together. Given our time and expertise we will classify any accuracy and probability testing above at an average accuracy (~70%) to be succesful. 

Our sample and testing clips were recorded in house by the team. The basic code is meant to be transferrable regardless of expert level, gender, and race. 

![Process](https://github.com/ffidai/Final_Project/blob/master/MD/process_overview.png)

### Predict: Forehand vs. Backhand

Now that the two training videos have been created, we will run various test videos. The test videos will be from a mix of players,specifically varying in race, gender, and expert level.

The model will also create an exportable Excel file of movements to visualize the forehand and backhand count per player. This will help single out outliers in our data to improve the code and processes for future development. 

![Analysis](https://github.com/ffidai/Final_Project/blob/master/MD/analysis.png)

### CNN

![Model](https://github.com/ffidai/Final_Project/blob/master/MD/CNN.jpeg)
https://medium.com/@RaghavPrabhu/understanding-of-convolutional-neural-network-cnn-deep-learning-99760835f148

## Tools Used

* Deep/Machine Learning: Tensorflow, Keras, ResNet 50 (Residual Networks), SKLearn
  * achieve image/video classification
* Python & Flask
  * run the demo on a web page
* Pandas & Tableau 
  * export data in Excel to visualize trend results
  * this can help improve the code overtime to see variances between players/movements
* HTML/CSS/Boostrap
  * house the data in a digestible format

## Demo
Now we will showcase the model in a live demo. 

## Findings
Classify the intersection of model running time & accuracy of the data of a usable, practical model given the output of the test videos. 

* Pro Right
    * 1 , **_5_** & 25 epochs
* Pro Left
    * 1 , 5 epochs
* Non Pro right
    * 1 , 5 , **_25_** & 50 epochs

## Potential Next Steps
We outlined an array of use cases for sports video classification. Key callouts below: 

* Personalized coaching
* Specialized sports medicine and physical therapy
* Player analytics
* Live, extractable data

Building on the starter code:
* Identify “waiting” player motions
* Variations of forehand and backhand movements, add in other key movements
* Professional camera to achieve clear frames
* Professional player for the training video to show clear, precise movements



____
Farah, Pooja, Vaidy