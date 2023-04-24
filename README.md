# EX-05-Feature-Generation


## AIM
To read the given data and perform Feature Generation process and save the data to a file. 

# Explanation
Feature Generation (also known as feature construction, feature extraction or feature engineering) is the process of transforming features into new features that better relate to the target.
 

# ALGORITHM
### STEP 1
Read the given Data
### STEP 2
Clean the Data Set using Data Cleaning Process
### STEP 3
Apply Feature Generation techniques to all the feature of the data set
### STEP 4
Save the data to the file


# CODE
~~~
Program Developed:Dharshini S
Register number:212220220009
~~~
## Data.csv
~~~
import pandas as pd
df=pd.read_csv("data.csv")
df

from sklearn.preprocessing import LabelEncoder, OrdinalEncoder
from sklearn.preprocessing import OneHotEncoder

oe=OrdinalEncoder()
df1=df.copy()

df1["City"] = oe.fit_transform(df1[["City"]])
df1["bin_1"] = oe.fit_transform(df1[["bin_1"]])
df1["Ord_1"] = oe.fit_transform(df1[["Ord_1"]])
df1["Ord_2"] = oe.fit_transform(df1[["Ord_2"]])
df1["bin_2"] = oe.fit_transform(df1[["bin_2"]])

df2=df.copy()

#feature scaling
from sklearn.preprocessing import MinMaxScaler
sc=MinMaxScaler()
df2=pd.DataFrame(sc.fit_transform(df1),columns=['id', 'bin_1', 'bin_2', 'City', 'Ord_1','Ord_2','Target'])
df2

from sklearn.preprocessing import StandardScaler
sc1=StandardScaler()
df3=pd.DataFrame(sc1.fit_transform(df1),columns=['id', 'bin_1', 'bin_2', 'City', 'Ord_1','Ord_2','Target'])
df3

from sklearn.preprocessing import MaxAbsScaler
sc2=MaxAbsScaler()
df4=pd.DataFrame(sc2.fit_transform(df1),columns=['id', 'bin_1', 'bin_2', 'City', 'Ord_1','Ord_2','Target'])
df4

from sklearn.preprocessing import RobustScaler
sc3=RobustScaler()
df5=pd.DataFrame(sc3.fit_transform(df1),columns=['id', 'bin_1', 'bin_2', 'City', 'Ord_1','Ord_2','Target'])
df5
~~~
## Encoding.csv
~~~
import pandas as pd
qf=pd.read_csv("encoding.csv")
qf

from sklearn.preprocessing import LabelEncoder, OrdinalEncoder
from sklearn.preprocessing import OneHotEncoder

oe=OrdinalEncoder()

qf1=qf.copy()


qf1["bin_1"] = oe.fit_transform(qf1[["bin_1"]])
qf1["nom_0"] = oe.fit_transform(qf1[["nom_0"]])
qf1["ord_2"] = oe.fit_transform(qf1[["ord_2"]])
qf1["bin_2"] = oe.fit_transform(qf1[["bin_2"]])

#feature scaling
from sklearn.preprocessing import MinMaxScaler
sc=MinMaxScaler()
qf0=pd.DataFrame(sc.fit_transform(qf1),columns=['id', 'bin_1', 'bin_2', 'nom_0','ord_2'])
qf0   

from sklearn.preprocessing import StandardScaler
sc1=StandardScaler()
qf2=pd.DataFrame(sc1.fit_transform(qf1),columns=['id', 'bin_1', 'bin_2', 'nom_0','ord_2'])
qf2

from sklearn.preprocessing import MaxAbsScaler
sc2=MaxAbsScaler()
qf3=pd.DataFrame(sc2.fit_transform(qf1),columns=['id', 'bin_1', 'bin_2', 'nom_0','ord_2'])
qf3

from sklearn.preprocessing import RobustScaler
sc3=RobustScaler()
qf4=pd.DataFrame(sc3.fit_transform(qf1),columns=['id', 'bin_1', 'bin_2', 'nom_0','ord_2'])
qf4
~~~
## Titanic_dataset.csv
~~~
import pandas as pd
rf=pd.read_csv("titanic.csv")
rf

#removing unwanted data
rf.drop("Name",axis=1,inplace=True)
rf.drop("Ticket",axis=1,inplace=True)
rf.drop("Cabin",axis=1,inplace=True)  

rf["Age"]=rf["Age"].fillna(rf["Age"].median())
rf["Embarked"]=rf["Embarked"].fillna(rf["Embarked"].mode()[0])

rf.isnull().sum()

rf1=rf.copy()

from sklearn.preprocessing import LabelEncoder, OrdinalEncoder
embark=['S','C','Q']
oe=OrdinalEncoder()

e1=OrdinalEncoder(categories=[embark])
rf1['Embarked'] = e1.fit_transform(rf[['Embarked']])
rf1['Sex'] = oe.fit_transform(rf[['Sex']])
rf1

#feature scaling
from sklearn.preprocessing import MinMaxScaler
sc=MinMaxScaler()
rf0=pd.DataFrame(sc.fit_transform(rf1),columns=['PassengerId', 'Survived', 'Pclass', 'Sex','Age','SibSp','Parch','Fare','Embarked'])
rf0

from sklearn.preprocessing import StandardScaler
sc1=StandardScaler()
rf3=pd.DataFrame(sc1.fit_transform(rf1),columns=['Passenger','Survived','Pclass','Sex','Age','SibSp','Parch','Fare','Embarked'])
rf3

from sklearn.preprocessing import MaxAbsScaler
sc2=MaxAbsScaler()
rf4=pd.DataFrame(sc2.fit_transform(rf1),columns=['Passenger','Survived','Pclass','Sex','Age','SibSp','Parch','Fare','Embarked'])
rf4

from sklearn.preprocessing import RobustScaler
sc3=RobustScaler()
rf5=pd.DataFrame(sc3.fit_transform(rf1),columns=['Passenger','Survived','Pclass','Sex','Age','SibSp','Parch','Fare','Embarked'])
rf5
~~~
# OUPUT
## Data.csv:
### Initial dataset:
![image](https://user-images.githubusercontent.com/113699377/233903074-e58d7a33-4d66-4518-953f-f604a192d290.png)

### Encoded dataset:
![image](https://user-images.githubusercontent.com/113699377/233903111-7e26d9de-4c53-4e8f-b13b-210f3c816f24.png)

### Data scaling using MinMaxScaler:
![image](https://user-images.githubusercontent.com/113699377/233903175-45a1ab0b-244a-44a0-a335-a06628a0cf8e.png)
### Data scaling using StandardScalar:
![image](https://user-images.githubusercontent.com/113699377/233903206-bc893183-8994-4bef-84e8-d7217bb95e0d.png)
### Data scaling using MaxAbsScaler:
![image](https://user-images.githubusercontent.com/113699377/233903252-6daf42f7-455e-4c59-9e63-527917d726f6.png)
### Data scaling using RobustScaler:
![image](https://user-images.githubusercontent.com/113699377/233903378-e75cf51f-49f6-43b4-8418-0f2906019acc.png)
## Encoding.csv:
### Initial dataset:
![image](https://user-images.githubusercontent.com/113699377/233903416-ce83167d-a73a-4841-81f7-2489ee2c4bc8.png)
### Encoded dataset:
![image](https://user-images.githubusercontent.com/113699377/233903479-041e3ed4-622f-45fd-885a-5d993cf9f2f7.png)
### Data scaling using MinMaxScaler:
![image](https://user-images.githubusercontent.com/113699377/233903531-f1d59779-7701-4a58-8af2-0cb5205baf97.png)
### Data scaling using StandardScalar
![image](https://user-images.githubusercontent.com/113699377/233903634-11a92f01-6c89-4aae-9184-a06d4c9171e0.png)
### Data scaling using MaxAbsScaler:
![image](https://user-images.githubusercontent.com/113699377/233903684-ed6f2de6-9e7a-41b2-b1ff-c7d8aac7037d.png)
## Titanic_dataset.csv:
### Initial dataset:
![image](https://user-images.githubusercontent.com/113699377/233904195-7a3d1164-0f16-4ee4-9cea-e20f99fe38e0.png)
### isnull.sum():
![image](https://user-images.githubusercontent.com/113699377/233904284-563c7fdb-1225-4855-a69f-29b307ac282c.png)
### Encoded dataset:
![image](https://user-images.githubusercontent.com/113699377/233904347-88fa8821-3a32-450a-b964-03e4c6bb021e.png)
### Data scaling using MinMaxScaler:
![image](https://user-images.githubusercontent.com/113699377/233904397-cd367f55-5d86-478c-be2a-bb34ab41794d.png)
### Data scaling using StandardScalar:
![image](https://user-images.githubusercontent.com/113699377/233904511-aeb2acc1-ceed-424e-a79a-3e2a838713d0.png)

### Data scaling using MaxAbsScaler:
![image](https://user-images.githubusercontent.com/113699377/233904549-39257c62-5886-4479-914e-2307517b68b8.png)
### Data scaling using RobustScaler:
![image](https://user-images.githubusercontent.com/113699377/233904621-49002c6d-c384-4541-b90e-de83f4617bb2.png)
# Result:
~~~
Feature Generation process and Feature Scaling process is applied to the given data frames sucessfully.
~~~


