from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.tree import DecisionTreeClassifier

train_data = pd.read_csv("../input/titanic/train.csv")
test_data = pd.read_csv("../input/titanic/test.csv")

#1 - préparation data 

#stratégie d imputation pour les valeurs numériques
numerical_transformer = SimpleImputer(strategy='constant')

#stratégie d imputation mais surtout d encodage pour les variables categoricales
categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('onehot', OneHotEncoder(handle_unknown='ignore'))
])


#TODO faire tourner la methode pour la profondeur des arbres/Under- Overfitting pour le prochain modele

#definition du modele

#1 - les propriétés à analyser
'''
features = ['Pclass', 'Sex', 'Age', 'Fare', 'Cabin','Embarked']
XTrain = train_data[features]
Xtest = test_data[features]

'''
#selection des colonnes objets , soit categorical
categorical_cols = [cname for cname in train_data.columns if train_data[cname].nunique() < 10 and train_data[cname].dtype == "object"]
#selection des colonnes numericlas
numerical_cols = [cname for cname in train_data.columns if train_data[cname].dtype in ['int64', 'float64']]
my_cols = categorical_cols + numerical_cols
X_train_inter = train_data[my_cols].copy()
#X_train_defintif = X_train_inter.drop(['Survived'],axis=1)

#bidouillage de mes morts
#X_train_defintif.columns
#df1 = df[['a','d']]
X_test_Dtest_data_Def = test_data[['Sex','Embarked','PassengerId','Pclass','Age','SibSp','Parch','Fare']]

#2 la propriété à prédire

yTrain  = train_data.Survived
yTest =  train_data.Survived


#application imputation , encoding
preprocessor = ColumnTransformer(
    transformers=[
        ('num', numerical_transformer, numerical_cols),
        ('cat', categorical_transformer, categorical_cols)
    ])

#le modele:
model = DecisionTreeClassifier(random_state=0)

#association modele modele et modele data
clf = Pipeline(steps=[('preprocessor', preprocessor),
                      ('model', model)
                     ])

#entrainer
clf.fit(X_train_inter, yTrain)
#predire
preds = clf.predict(X_test_Dtest_data_Def)

print("toto")