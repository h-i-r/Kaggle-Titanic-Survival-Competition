import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from sklearn.linear_model import BayesianRidge
from sklearn import preprocessing
import scipy.stats as stats
from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
import squarify

def training_features(df_titanic):

    # PassengerId
    print(df_titanic['PassengerId'].nunique())
    df_titanic['PassengerId'] = df_titanic['PassengerId'].astype(int)

    # Survived
    df_titanic['Survived'] = df_titanic['Survived'].astype(int)
    labels = ['Not Survived', 'Survived']
    values = df_titanic["Survived"].value_counts()
    colors = ['mediumvioletred', 'orchid']
    plt.pie(values, labels=labels, colors=colors, startangle=90, autopct='%.1f%%', shadow=True, explode=[.04, .04])
    plt.title('Survival Percentage')
    plt.show()

    # Pclass
    df = df_titanic.groupby('Pclass').size().reset_index(name='counts')
    labels = ['Lower', 'Middle', 'Upper']
    sizes = df['counts'].values.tolist()
    colors = [plt.cm.Spectral(i / float(len(labels))) for i in range(len(labels))]
    squarify.plot(sizes=sizes, label=labels, color=colors)
    plt.title('Pclass Concentration Split')
    plt.show()

    # Name
    print(df_titanic['Name'].nunique())
    df_titanic['Name'].replace(".*Countess.*|.*Mme.*|.*Mrs. .*", 'Mrs', regex=True, inplace=True)
    df_titanic['Name'].replace(".*Mr. .*|.*Mr .*", 'Mr', regex=True, inplace=True)
    df_titanic['Name'].replace(".*Mlle.*|.*Miss..*|.*Ms.*", 'Miss', regex=True, inplace=True)
    df_titanic['Name'].replace(".*Master..*", 'Master', regex=True, inplace=True)
    df_titanic['Name'].replace(".*Dr.*|.*Rev..*|.*Col..*|.*Major..*|.*Capt..*", 'Professional Title', regex=True, inplace=True)

    for i in df_titanic['Name']:
        if i not in ['Mrs', 'Mr', 'Miss', 'Master', 'Professional Title']:
            df_titanic = df_titanic.replace(i, 'Others')

    print(df_titanic['Name'].value_counts())

    # Sex
    labels = ['Male', 'Female']
    values = df_titanic["Sex"].value_counts()
    colors = ['steelblue', 'palevioletred']
    plt.pie(values, labels=labels, colors=colors, startangle=90, autopct='%.1f%%', shadow=True, explode=[.04, .04])
    plt.title('Gender Percentage')
    plt.show()

    # SibSp
    print(df_titanic['SibSp'].sort_values().unique())

    # Parch
    print(df_titanic['Parch'].sort_values().unique())

    # familyMembers
    familyMembers = pd.DataFrame(df_titanic['Parch'] + df_titanic['SibSp'])
    df_titanic = pd.DataFrame(pd.concat([df_titanic, familyMembers], axis=1))
    df_titanic.rename(columns={0: 'familyMembers'}, inplace=True)

    # Ticket
    match = re.compile('[\d]')
    ticketInfo = pd.DataFrame(df_titanic['Ticket'])
    ticketInfo.rename(columns={'Ticket': 'ticketInfo'}, inplace=True)

    for i in range(0, len(df_titanic['Ticket'])):
        if re.match(match, df_titanic.loc[i, 'Ticket']) == None:
            ticketInfo.replace(to_replace=ticketInfo.iloc[i], value='Type 1', inplace=True)
        else:
            ticketInfo.replace(to_replace=ticketInfo.iloc[i], value='Type 2', inplace=True)
    print(ticketInfo.value_counts())
    df_titanic = pd.DataFrame(pd.concat([df_titanic, ticketInfo], axis=1))

    # Fare
    print(df_titanic['Fare'].value_counts())

    # farePerFamily
    farePerFamily = pd.DataFrame(df_titanic['Fare']/(df_titanic['familyMembers']+1))
    df_titanic = pd.DataFrame(pd.concat([df_titanic, farePerFamily], axis=1))
    df_titanic.rename(columns={0: 'farePerFamily'}, inplace=True)

    # N U L L S

    # Embarked
    df_Embarked = pd.DataFrame(df_titanic['Embarked'])
    si = SimpleImputer(missing_values=np.nan, strategy='most_frequent')
    df_Embarked = pd.DataFrame(si.fit_transform(df_Embarked))
    df_Embarked.rename(columns={0: 'Embarked'}, inplace=True)
    df_titanic.drop(columns='Embarked', inplace=True, axis=1)
    df_titanic = pd.concat([df_titanic, df_Embarked], axis=1)
    df_titanic["Embarked"].value_counts().plot(kind='bar', color='yellow', title='Embarked Breakdown', legend=True)
    plt.show()

    # Cabin
    df_titanic['Cabin'] = df_titanic['Cabin'].str[0:1]
    df_titanic['Cabin'].fillna('U', inplace=True)
    print(df_titanic['Cabin'].value_counts())
    df_titanic["Cabin"].value_counts().plot(kind='bar', color='cyan', title='Cabin Breakdown', legend=True)
    plt.show()

    # Cabin x ticketInfo
    Cabin_ticketInfo = pd.DataFrame(df_titanic['Cabin']+df_titanic['ticketInfo'])
    df_titanic = pd.DataFrame(pd.concat([df_titanic, Cabin_ticketInfo], axis=1))
    df_titanic.rename(columns={0: 'Cabin_ticketInfo'}, inplace=True)
    print(df_titanic['Cabin_ticketInfo'].value_counts())

    # Label Encoding of Categorical Columns
    df_titanic_cat = pd.DataFrame(df_titanic[['Pclass', 'Name', 'Sex', 'SibSp', 'Parch', 'Ticket', 'Fare', 'Embarked', 'familyMembers', 'ticketInfo', 'Cabin', 'Cabin_ticketInfo']])
    df_titanic = df_titanic.drop(columns=['Pclass', 'Name', 'Sex', 'SibSp', 'Parch', 'Ticket', 'Fare', 'Embarked', 'familyMembers', 'ticketInfo', 'Cabin', 'Cabin_ticketInfo'], axis=1)

    le = LabelEncoder()
    for i in df_titanic_cat.columns:
        df_titanic_cat[i] = le.fit_transform(df_titanic_cat[i])

    df_titanic_cat = df_titanic_cat.astype(int)
    df_titanic = pd.concat([df_titanic,df_titanic_cat], axis=1)

    # Age
    df_titanic_columns = df_titanic.columns
    imputer = IterativeImputer(BayesianRidge())
    df_titanic = pd.DataFrame(imputer.fit_transform(df_titanic))
    df_titanic.columns = df_titanic_columns
    df_titanic['Age'] = np.round(df_titanic['Age'], 1)

    # age_bins
    age_bins = pd.DataFrame(pd.qcut(df_titanic['Age'], q=4, labels=[0, 1, 2, 3]))
    age_bins.rename(columns={'Age': 'ageBins'}, inplace=True)
    age_bins = age_bins.astype(int)
    df_titanic = pd.concat([df_titanic, age_bins], axis=1)

    # numerical
    numerical = pd.DataFrame(df_titanic['Age'] * df_titanic['Fare'])
    df_titanic = pd.DataFrame(pd.concat([df_titanic, numerical], axis=1))
    df_titanic.rename(columns={0: 'numerical'}, inplace=True)

    # Standard Scaling
    df_titanic_continous = pd.DataFrame(df_titanic[['Fare', 'farePerFamily', 'numerical']])
    df_titanic_continous = pd.DataFrame(preprocessing.StandardScaler().fit_transform(df_titanic_continous))
    df_titanic_continous.rename(columns={0: 'Fare', 1: 'farePerFamily', 2: 'numerical'}, inplace=True)

    # Plot 'Fare'
    plt.subplot(1, 2, 1)
    fit1 = stats.norm.pdf(df_titanic['Fare'], np.mean(df_titanic['Fare']), np.std(df_titanic['Fare']))
    plt.plot(df_titanic['Fare'], fit1, 'cx')
    plt.title('Original')

    plt.subplot(1, 2, 2)
    fit2 = stats.norm.pdf(df_titanic_continous['Fare'], np.mean(df_titanic_continous['Fare']), np.std(df_titanic_continous['Fare']))
    plt.plot(df_titanic_continous['Fare'], fit2, 'mx')
    plt.title('Standard Scaling')

    plt.tight_layout()
    plt.show()

    df_titanic_continous = np.round(df_titanic_continous, 2)
    df_titanic.drop(columns=['Ticket', 'Fare', 'farePerFamily', 'numerical'], inplace=True, axis=1)
    df_titanic = pd.concat([df_titanic, df_titanic_continous], axis=1)

    return df_titanic

def testing_features(X_test):

    # Name
    X_test['Name'].replace(".*Countess.*|.*Mme.*|.*Mrs. .*", 'Mrs', regex=True, inplace=True)
    X_test['Name'].replace(".*Mr. .*|.*Mr .*", 'Mr', regex=True, inplace=True)
    X_test['Name'].replace(".*Mlle.*|.*Miss..*|.*Ms.*", 'Miss', regex=True, inplace=True)
    X_test['Name'].replace(".*Master..*", 'Master', regex=True, inplace=True)
    X_test['Name'].replace(".*Dr.*|.*Rev..*|.*Col..*|.*Major..*|.*Capt..*", 'Professional Title', regex=True, inplace=True)

    for i in X_test['Name']:
        if i not in ['Mrs', 'Mr', 'Miss', 'Master', 'Professional Title']:
            X_test = X_test.replace(i, 'Others')

    # familyMembers
    familyMembers = pd.DataFrame(X_test['Parch'] + X_test['SibSp'])
    X_test = pd.DataFrame(pd.concat([X_test, familyMembers], axis=1))
    X_test.rename(columns={0: 'familyMembers'}, inplace=True)

    # Ticket
    match = re.compile('[\d]')
    ticketInfo = pd.DataFrame(X_test['Ticket'])
    ticketInfo.rename(columns={'Ticket': 'ticketInfo'}, inplace=True)

    for i in range(0, len(X_test['Ticket'])):
        if re.match(match, X_test.loc[i, 'Ticket']) == None:
            ticketInfo.replace(to_replace=ticketInfo.iloc[i], value='Type 1', inplace=True)
        else:
            ticketInfo.replace(to_replace=ticketInfo.iloc[i], value='Type 2', inplace=True)

    X_test = pd.DataFrame(pd.concat([X_test, ticketInfo], axis=1))

    # Fare
    df_Fare = pd.DataFrame(X_test['Fare'])
    si = SimpleImputer(missing_values=np.nan, strategy='most_frequent')
    df_Fare = pd.DataFrame(si.fit_transform(df_Fare))
    df_Fare.rename(columns={0: 'Fare'}, inplace=True)
    X_test = X_test.drop(columns='Fare', axis=1)
    X_test = pd.concat([X_test, df_Fare], axis=1)

    # farePerFamily
    farePerFamily = pd.DataFrame(X_test['Fare']/(X_test['familyMembers']+1))
    X_test = pd.DataFrame(pd.concat([X_test, farePerFamily], axis=1))
    X_test.rename(columns={0: 'farePerFamily'}, inplace=True)

    # Cabin
    X_test['Cabin'] = X_test['Cabin'].str[0:1]
    X_test['Cabin'].fillna('U', inplace=True)
    print(X_test['Cabin'].value_counts())

    # Cabin x ticketInfo
    Cabin_ticketInfo = pd.DataFrame(X_test['Cabin']+X_test['ticketInfo'])
    X_test = pd.DataFrame(pd.concat([X_test, Cabin_ticketInfo], axis=1))
    X_test.rename(columns={0: 'Cabin_ticketInfo'}, inplace=True)
    print(X_test['Cabin_ticketInfo'].value_counts())

    # Label Encoding of Categorical Columns
    X_test_cat = pd.DataFrame(X_test[['Pclass', 'Name', 'Sex', 'SibSp', 'Parch', 'Ticket', 'Fare', 'Embarked', 'familyMembers', 'ticketInfo', 'Cabin', 'Cabin_ticketInfo']])
    X_test = X_test.drop(columns=['Pclass', 'Name', 'Sex', 'SibSp', 'Parch', 'Ticket', 'Fare', 'Embarked', 'familyMembers', 'ticketInfo', 'Cabin', 'Cabin_ticketInfo'], axis=1)

    le = LabelEncoder()
    for i in X_test_cat.columns:
        X_test_cat[i] = le.fit_transform(X_test_cat[i])
    X_test = pd.concat([X_test, X_test_cat], axis=1)

    # Age
    X_test_columns = X_test.columns
    imputer = IterativeImputer(BayesianRidge())
    X_test = pd.DataFrame(imputer.fit_transform(X_test))
    X_test.columns = X_test_columns
    X_test['Age'] = np.round(X_test['Age'], 1)

    # age_bins
    age_bins = pd.DataFrame(pd.qcut(X_test['Age'], q=4, labels=[0, 1, 2, 3]))
    age_bins.rename(columns={'Age': 'ageBins'}, inplace=True)
    X_test = pd.concat([X_test, age_bins], axis=1)

    # numerical
    numerical = pd.DataFrame(X_test['Age']*X_test['Fare'])
    X_test = pd.DataFrame(pd.concat([X_test, numerical], axis=1))
    X_test.rename(columns={0: 'numerical'}, inplace=True)

    # Mean Normalization
    X_test_continous = pd.DataFrame(X_test[['Fare', 'farePerFamily', 'numerical']])
    X_test_continous = pd.DataFrame(preprocessing.normalize(X_test_continous))
    X_test_continous.rename(columns={0: 'Fare', 1: 'farePerFamily', 2: 'numerical'}, inplace=True)

    X_test.drop(columns=['Fare', 'farePerFamily', 'Ticket', 'numerical'], inplace=True, axis=1)
    X_test = pd.concat([X_test, X_test_continous], axis=1)

    return X_test

def main():

    # training dataset
    df_titanic = pd.read_csv("train.csv")
    print(df_titanic.shape)
    print(df_titanic.dtypes)
    print(df_titanic.head(25))
    print(df_titanic.isnull().sum())
    # training features
    df_titanic = training_features(df_titanic)
    # Export training data
    df_titanic.to_csv("df_titanic.csv", index=False)

    # test data
    test_set = pd.read_csv("test.csv")
    # test features
    test_set = testing_features(test_set)

    # M O D E L
    X_train = pd.DataFrame(df_titanic.loc[:, ['Sex', 'numerical', 'Age', 'Fare', 'farePerFamily', 'Name', 'Pclass', 'Cabin_ticketInfo', 'SibSp', 'ageBins', 'Embarked']])
    y_train = pd.DataFrame(df_titanic['Survived'])
    X_test = pd.DataFrame(test_set.loc[:, ['Sex', 'numerical', 'Age', 'Fare', 'farePerFamily', 'Name', 'Pclass', 'Cabin_ticketInfo', 'SibSp', 'ageBins', 'Embarked']])

    # Random Forest Classifier
    randomForestClassifier = RandomForestClassifier()
    randomForestClassifier.fit(X_train, y_train.values.ravel())
    y_pred = randomForestClassifier.predict(X_test)

    # Evaluate
    accuracy_train = randomForestClassifier.score(X_train, y_train)
    print('accuracy_train', accuracy_train)

    # Submission
    Survived = pd.DataFrame(y_pred, columns=['Survived'])
    final_submission = pd.concat([test_set['PassengerId'], Survived], axis=1)
    final_submission = final_submission.astype(int)
    final_submission = final_submission.to_csv('submission.csv', index=False)

if __name__ == "__main__" :
    main()
