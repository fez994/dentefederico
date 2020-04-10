## 1.0 Introduction   

I spent the last week studying machine learning, algorithms, data analysis and statistics. I’m a firm believer that the best way to learn something is by doing. This is the result of my first machine learning project using Scikit-Learn, MatPlotLib and Pandas.  

Objectives:  

- Using Machine Learning to predict the strength level of an Athlete based on Gender, Bodyweight, Weight Class and TotalKg lifted.   

- Better understand the level of athletes that partake in Powerlifting competition 

## 1.1. Context  

The dataset that I used is a snapshot of the OpenPowerlifting Database as of April 2019. OepnPowerlifting is creating a public-domain archive of powerlifting history. Powerlifting is a sport in which competitors compete to lift the most weight for their class in three separate barbell lifts: Squat, Bench and Deadlift.  

## 2.0 Summarize Data

Machine learning algorithms learn from data. It is critical that I feed them the right data for the problem i want to solve. Even if I have good data, I need to make sure that it is in a useful scale, format and even that meaningful features are included.
Let's take a quick look into the OpenPowerlifting data: 

```
import pandas as pd

dataset = pd.read_csv('openpowerlifting.csv', low_memory=False)
print(dataset.shape)
print(dataset.head(10))

```

```
(1423354, 37)
               Name Sex Event Equipment   Age AgeClass Division  BodyweightKg  ... IPFPoints  Tested  Country  Federation        Date  MeetCountry  MeetState       MeetName 
0      Abbie Murphy   F   SBD     Wraps  29.0    24-34     F-OR          59.8  ...    511.15     NaN      NaN     GPC-AUS  2018-10-27    Australia        VIC  Melbourne Cup 
1       Abbie Tuong   F   SBD     Wraps  29.0    24-34     F-OR          58.5  ...    595.65     NaN      NaN     GPC-AUS  2018-10-27    Australia        VIC  Melbourne Cup 
2    Ainslee Hooper   F     B       Raw  40.0    40-44     F-OR          55.4  ...    313.97     NaN      NaN     GPC-AUS  2018-10-27    Australia        VIC  Melbourne Cup 
3   Amy Moldenhauer   F   SBD     Wraps  23.0    20-23     F-OR          60.0  ...    547.04     NaN      NaN     GPC-AUS  2018-10-27    Australia        VIC  Melbourne Cup 
4      Andrea Rowan   F   SBD     Wraps  45.0    45-49     F-OR         104.0  ...    550.08     NaN      NaN     GPC-AUS  2018-10-27    Australia        VIC  Melbourne Cup 
5     April Alvarez   F   SBD     Wraps  37.0    35-39     F-OR          74.0  ...    596.18     NaN      NaN     GPC-AUS  2018-10-27    Australia        VIC  Melbourne Cup 
6        Ash Morgan   F   SBD     Wraps  23.0    20-23     F-OR          59.8  ...    612.23     NaN      NaN     GPC-AUS  2018-10-27    Australia        VIC  Melbourne Cup 
7   Belinda Moloney   F   SBD     Wraps  35.0    35-39     F-OR          80.4  ...    575.85     NaN      NaN     GPC-AUS  2018-10-27    Australia        VIC  Melbourne Cup 
8   Briony Williams   F   SBD     Wraps  36.0    35-39     F-OR         108.0  ...    716.65     NaN      NaN     GPC-AUS  2018-10-27    Australia        VIC  Melbourne Cup 
9  Brooke Kowalczyk   F   SBD     Wraps  37.0    35-39     F-OR          74.8  ...    762.42     NaN      NaN     GPC-AUS  2018-10-27    Australia        VIC  Melbourne Cup 
```

1423354 instances, 37 attributes 
