I spent the past couple of weeks studying machine learning, algorithms, data analysis and statistics. I’m a firm believer that the best way to learn something is by doing. This is the result of my first machine learning project using Scikit-Learn, MatPlotLib and Pandas.  

Objectives:  

- Using Machine Learning to predict the strength level of an Athlete based on Gender, Bodyweight, Weight Class and TotalKg lifted.   

- Better understand the level of athletes that partake in Powerlifting competition 

## Context  

The dataset that I used is a snapshot of the OpenPowerlifting Database as of April 2019. OpenPowerlifting is creating a public-domain archive of powerlifting history. Powerlifting is a sport in which competitors compete to lift the most weight for their class in three separate barbell lifts: Squat, Bench and Deadlift.  

## Data Prep

Machine learning algorithms learn from data. It is critical that you feed them the right data for the problem you want to solve. Even if you have good data, you need to make sure that it is in a useful scale, format and even that meaningful features are included.
Let's take a quick look into the OpenPowerlifting data: 

```
import pandas as pd

dataset = pd.read_csv('openpowerlifting.csv', low_memory=False)
print(dataset.shape)
print(dataset.head(10))

```
Result: 

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

Powerlifting is a pretty fragmented sport. This means that there are hundreds if not thousands of different federations, each one with their own rules regarding weight classes, equipment and drug tests. This is evident by taking a quick look at the unique weight classes in the data.

```
print('Number of unique weightclass: ' + str(dataset['WeightClassKg'].nunique()))

Result: 
Number of unique weightclasses: 224
```

224 unique weight classes are way too much, i'll reduce them to 15:


Man:  59 kg, 66 kg, 74 kg, 83 kg, 93 kg, 105 kg, 120 kg, 120 kg+



Women: 43 kg, 47 kg, 52 kg, 57 kg, 63 kg, 72 kg, 84 kg, 84 kg+

Those are the weightclasses currently used by the IPF (International Powerlifting Federation) 

## Selecting Data

You need to consider what data you need to address the question or problem you are working on. For this project, I'll only consider athletes that train RAW (No equipment), and I'll include both Tested and Non-tested divisions. This is because the difference between natural athletes and enhanced athletes is not that big, and sometimes tested athletes are even stronger than untested ones. The only column that I'm going to keep for training are Gender, BodyweightKg, TotalKg. 

### Strength Classification

Andrius Virbičianskas published tables (like the one down below) with athletes level based on the same data that i'm using. The separation between classes was done using standard deviations. For example World Class represents 0.63% of all lifters (that’s usually in World TOP 100 in respective weight categories) and is +2.5 std from mean.

![alt text](https://i.imgur.com/DSoHewV.png "0.1")

```
#Filtering data for RAW totals ONLY
dataset = dataset[dataset['Equipment'].map(len) < 4]

# Bodyweight Classing by Sex, using IPF standards
male = dataset[dataset['Sex']=='M']
female = dataset[dataset['Sex'] == 'F']

def male_weight_class(x):
    if x <= 59:
        return 59
    if x <= 66 and x > 59:
        return 66
    if x <= 74 and x > 66:
        return 74
    if x <= 83 and x > 74:
        return 83
    if x <= 93 and x > 83:
        return 93
    if x <= 105 and x > 94:
        return 105
    if x <= 120 and x > 105:
        return 120
    if x > 120:
        return 121
        
def female_weight_class(x):
    if x <= 43:
        return 43
    if x <= 47 and x > 43:
        return 47
    if x <= 52 and x > 47:
        return 52
    if x <= 57 and x > 52:
        return 57
    if x <= 63 and x > 57:
        return 63
    if x <= 72 and x > 63:
        return 72
    if x <= 84 and x > 72:
        return 84
    if x > 84:
        return 85

#Adding WeightClasses 
male['WeightClassKg'] = male['BodyweightKg'].apply(male_weight_class)
female['WeightClassKg'] = female['BodyweightKg'].apply(female_weight_class)

dataset = pd.concat([male, female])

## Strength classification, this is a snippet of the code, i did this for every weight class bot male and female 
def strengthStandards(y):

#-120kg Class
    if y['Sex'] == 'M' and y['WeightClassKg'] == 120 and y['TotalKg'] <= 305 or y['TotalKg'] < 375:
        return 'Untrained'
    if y['Sex'] == 'M' and y['WeightClassKg'] == 120 and y['TotalKg'] >= 375 and y['TotalKg'] < 500:
        return 'Beginner'
    if y['Sex'] == 'M' and y['WeightClassKg'] == 120 and y['TotalKg'] >= 500 and y['TotalKg'] < 607:
        return 'Intermediate'
    if y['Sex'] == 'M' and y['WeightClassKg'] == 120 and y['TotalKg'] >= 607 and y['TotalKg'] < 712:
        return 'Advanced'
    if y['Sex'] == 'M' and y['WeightClassKg'] == 120 and y['TotalKg'] >= 712 and y['TotalKg'] < 815:
        return 'Master'
    if y['Sex'] == 'M' and y['WeightClassKg'] == 120 and y['TotalKg'] >= 815 and y['TotalKg'] < 870:
        return 'Elite'
    if y['Sex'] == 'M' and y['WeightClassKg'] == 120 and y['TotalKg'] >= 870:
        return 'World Class'


```

Now it's just a simple matter of exporting cleaned data to a new file.


```
dataset["StrengthLevel"] = dataset.apply(strengthStandards, axis=1)
keep_col = ['Sex', 'BodyweightKg','TotalKg','WeightClassKg','StrengthLevel']
new_file = dataset[keep_col]

new_file.to_csv('cleandata.csv', index=False)
```

The data is now ready, let's take a quick look at it. 

```
  Sex  BodyweightKg  TotalKg  WeightClassKg StrengthLevel
0   M          82.1    317.5           83.0     Untrained
1   M          98.6    510.0          105.0  Intermediate
2   M          88.1    510.0           93.0  Intermediate
3   M         116.7    175.0          120.0     Untrained
4   M          67.1    535.0           74.0      Advanced
5   M          82.5    345.0           83.0      Beginner
6   M         117.7    685.0          120.0      Advanced
7   M         109.1    200.0          120.0     Untrained
8   M         113.2    130.0          120.0     Untrained
9   M          98.1    195.0          105.0     Untrained

```

And that's it, 5 attributes with only the pieces of information that we need. It's time to fulfill the objectives that we set ourselves at the start of the article: Better understand the level of athletes that partake in Powerlifting competition. 

## Strength Level of Powerlifters 

Let's group all the athletes into a nice graph so that we can look at the data in an easy to understand way. 


```
slevel = dataset.groupby(['StrengthLevel']).size()
slevel.plot.bar()
plt.tight_layout()
plt.show()

```

![alt text](https://i.imgur.com/CGW4VYV.png "0.2")

It's evident that the vast majority of people that compete in powerlifting are **untrained**, suggesting that people start competing pretty early on in their careers. This could be true for different reasons: 

Powerlifting is a pretty cheap sport (compared to something like golf it is super cheap) but there are some costs associated with it. These things include:

- The entry fee for the competition 
- Joining a federation for a year 
- Purchasing a singlet to compete in 
- Cost of travel to and from the competition including a potential hotel stay

Furthermore, the powerlifting community tends to encourage athletes to compete early to build experience to perform better in future competitions.

Another interesting aspect of the data is that the number of advanced athletes is greater than the intermediates.

**My final take:** 

- The vast majority of people that competes are beginners (Technically untrained, but at the end of the day they are the same thing)
- There are more advanced athletes than intermediate, this might suggest the idea that some athletes prefer to put on hold competing until they reach a decent level of strength. 


# Algorithms Evaluation 

Now it is time to create some models of the data and estimate their accuracy on unseen data.
Here is what i'm going to do: 

1. Separate out a validation dataset.
2. Setup the test harness to use 10-fold cross validation.
3. Build 3 different models to predict an athlete strength level
4. Select the best model.

I'll split the loaded dataset into two, 80% of which I will use to train our models and 20% that I'll hold back as a validation dataset.
```
# Replacing Male and Female gender with 0 and 1 so that our algorithms can process the information
dataset['Sex'].replace(['M','F'],[0,1],inplace=True)

#Split-out validation dataset
array = dataset.values
X = array[:, 0:4]
Y = array[:, 4]
validation_size = 0.20
seed = 7
X_train, X_validation, Y_train, Y_validation = train_test_split(X, Y, test_size=validation_size, random_state=seed)
```

I'll use 10-fold cross validation to estimate accuracy. This will split our dataset into 10 parts, train on 9 and test on 1 and repeat for all combinations of train-test splits. I'm using the metric of accuracy to evaluate models. This is a ratio of the number of correctly predicted instances divided by the total number of instances in the dataset multiplied by 100 to give a
percentage (e.g. 95% accurate).

I don't know which algorithms would be good on this problem or what configurations to use so let's evaluate three different algorithms:

- Logistic Regression (LR).
- Linear Discriminant Analysis (LDA).
- k-Nearest Neighbors (KNN).

I reset the random number seed before each run to ensure that the evaluation of each algorithm is performed using exactly the same data splits. It ensures the results are directly comparable. Let's build and evaluate our models:

```
models = []
models.append(('LR', LogisticRegression()))
models.append(('LDA', LinearDiscriminantAnalysis()))
models.append(('KNN', KNeighborsClassifier()))

results = []
names = []
for name, model in models:
    kfold = KFold(n_splits=10, random_state=seed)
    cv_results = cross_val_score(model, X_train, Y_train, cv=kfold, scoring='accuracy')
    results.append(cv_results)
    names.append(name)
    msg = "%s: %f (%f)" % (name, cv_results.mean(), cv_results.std())
    print(msg)


```




```
LR: 0.794054 (0.003679)
LDA: 0.828183 (0.002426)
KNN: 0.998550 (0.000220)
```





