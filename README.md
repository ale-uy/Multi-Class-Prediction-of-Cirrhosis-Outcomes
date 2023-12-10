___

<h1 style="background-color: red; color: black; font-family: cursive; font-size: 300%; text-align: center; border-radius: 50px 50px">Multi-Class Prediction of Cirrhosis Outcomes</h1>

---

We will use boosting and stacking classifier models, we will apply Box-Cox transformation, we will under/oversample classes, we will calibrate probabilities, we will use class weights and much more, it will be interesting!

Utilizaremos modelos clasificadores de boosting y stacking, aplicaremos transformacion Box-Cox, haremos sub/sobremuestreo de clases, calibraremos probabilidades, usaremos pesos de clases y mucho mas ¡será interesante!

>By [Ale uy](https://www.kaggle.com/lasm1984)

___

<h1 id="goto0" style="background-color:orangered;font-family:cursive;color:black;font-size:250%;text-align:center;border-radius: 50px 50px;">Table of Contents</h1>

0. [Table of Contents](#goto0)

1. [Notebook Description](#goto1)

2. [Loading Libraries](#goto2)

3. [Reading Data Files](#goto3)

4. [Data Exploration](#goto4)

5. [Individuals Modeling](#goto5)

    5a. [Logistic Model](#goto5a)

    5b. [XGB Model](#goto5b)

    5c. [LGBM Model](#goto5c)

    5d. [CAT Model](#goto5d)

    5e. [NN Model](#goto5e)

6. [Staking Models](#goto6)

    6a. [Voting Model](#goto6a)

7. [Conclusions](#goto7)

<h1 id="goto1" style="background-color:orangered;font-family:cursive;color:black;font-size:250%;text-align:center;border-radius: 50px 50px;">Notebook Description</h1>

[Back to Table of Contents](#goto0)

> ### **ENGLISH**

<u>**Goal**</u>: Utilize clinical features for predicting survival state of patients with liver cirrhosis. The survival states include 0 = D (death), 1 = C (censored), 2 = CL (censored due to liver transplantation).

**Dataset Description**

The dataset for this competition (both train and test) was generated from a deep learning model trained on the [Cirrhosis Patient Survival Prediction dataset](https://www.kaggle.com/datasets/joebeachcapital/cirrhosis-patient-survival-prediction). Feature distributions are close to, but not exactly the same, as the original. Feel free to use the original dataset as part of this competition, both to explore differences as well as to see whether incorporating the original in training improves model performance.

**For what purpose was the dataset created?**

Cirrhosis results from prolonged liver damage, leading to extensive scarring, often due to conditions like hepatitis or chronic alcohol consumption. The data provided is sourced from a Mayo Clinic study on primary biliary cirrhosis (PBC) of the liver carried out from 1974 to 1984.

**Files**

``train.csv`` - the training dataset; Status is the categorical target; C (censored) indicates the patient was alive at N_Days, CL indicates the patient was alive at N_Days due to liver a transplant, and D indicates the patient was deceased at N_Days.

``test.csv`` - the test dataset; your objective is to predict the probability of each of the three Status values, e.g., Status_C, Status_CL, Status_D.

---

> ### **Español**

<u>**Objetivo**</u>: utilizar características clínicas para predecir el estado de supervivencia de pacientes con cirrosis hepática. Los estados de supervivencia incluyen 0 = D (muerte), 1 = C (censurado), 2 = CL (censurado debido a un trasplante de hígado).

**Descripción del conjunto de datos**

El conjunto de datos para esta competencia (tanto de entrenamiento como de prueba) se generó a partir de un modelo de aprendizaje profundo entrenado en el [conjunto de datos de predicción de supervivencia del paciente con cirrosis] (https://www.kaggle.com/datasets/joebeachcapital/cirrhosis-patient-survival-prediction ). Las distribuciones de funciones son cercanas, pero no exactamente iguales, a las del original. Siéntase libre de utilizar el conjunto de datos original como parte de esta competencia, tanto para explorar diferencias como para ver si la incorporación del original en el entrenamiento mejora el rendimiento del modelo.

**¿Con qué propósito se creó el conjunto de datos?**

La cirrosis es el resultado de un daño hepático prolongado, que provoca cicatrices extensas, a menudo debido a afecciones como la hepatitis o el consumo crónico de alcohol. Los datos proporcionados provienen de un estudio de Mayo Clinic sobre cirrosis biliar primaria (CBP) del hígado realizado entre 1974 y 1984.

**Archivos**

``train.csv`` - el conjunto de datos de entrenamiento; El estatus es el objetivo categórico; C (censurado) indica que el paciente estaba vivo en N_Days, CL indica que el paciente estaba vivo en N_Days debido a un trasplante de hígado y D indica que el paciente falleció en N_Days.

``test.csv`` - el conjunto de datos de prueba; su objetivo es predecir la probabilidad de cada uno de los tres valores de Estado, por ejemplo, Estado_C, Estado_CL, Estado_D.

---

>Reference: [Walter Reade, Ashley Chow. (2023). Multi-Class Prediction of Cirrhosis Outcomes. Kaggle.](https://www.kaggle.com/competitions/playground-series-s3e26)

<h1 id="goto2" style="background-color:orangered;font-family:cursive;color:black;font-size:250%;text-align:center;border-radius: 50px 50px;">Loading Libraries</h1>

[Back to Table of Contents](#goto0)

#### Basic Tools | Herramientas Básicas


```python
import pandas as pd; pd.set_option("display.max_columns", 30)
import numpy as np

import matplotlib.pyplot as plt; plt.style.use("fivethirtyeight")
import seaborn as sns; sns.set(style="whitegrid"); sns.set_palette("husl")
import plotly.express as px

import warnings; warnings.filterwarnings("ignore")
```

#### Advanced Tools | Herramientas Avanzadas


```python
from scipy.cluster.hierarchy import linkage, dendrogram
from scipy.spatial.distance import squareform
from sklearn.metrics import log_loss
from scipy.stats import boxcox
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import ADASYN
from imblearn.under_sampling import ClusterCentroids
from sklearn.utils.class_weight import compute_class_weight
from sklearn.calibration import CalibratedClassifierCV
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import VotingClassifier
```

<h1 id="goto3" style="background-color:orangered;font-family:cursive;color:black;font-size:250%;text-align:center;border-radius: 50px 50px;">Reading Data Files</h1> 

[Back to Table of Contents](#goto0)

#### Competition Data | Datos de la Competición


```python
train = pd.read_csv('train.csv')
test = pd.read_csv('test.csv')

print('The dimension of the train dataset is:', train.shape)
print('The dimension of the test dataset is:', test.shape)
```


```python
train.describe().T
```


```python
train.info()
```


```python
test.describe().T
```


```python
test.info()
```

#### Separate '`id`' and set the goal '`Status`' | Separar `id` y marcar el objetivo `Status`


```python
train_id = train['id']
train.drop('id', axis=1, inplace=True)

test_id = test['id']
test.drop('id', axis=1, inplace=True)

TARGET = 'Status'
```

<h1 id="goto4" style="background-color:orangered;font-family:cursive;color:black;font-size:250%;text-align:center;border-radius: 50px 50px;">Data Exploration</h1>

[Back to Table of Contents](#goto0)

## Dataset Plots | Gráficos de los Datos

#### Categorical Plots | Gráficos de Categóricas


```python
try:    
    categorical_columns = train.select_dtypes('O').columns
    # Calculate the number of categorical columns and rows to organize the plots
    num_columns = len(categorical_columns)
    rows = (num_columns + 1) // 2
    # Create the figure and axes for the plots
    _, ax = plt.subplots(nrows=rows, ncols=2, figsize=(16, rows * 6))
    ax = ax.flat
    # Generate horizontal bar charts for each categorical variable
    for i, col in enumerate(categorical_columns):
        train[col].value_counts().plot.barh(ax=ax[i])
        ax[i].set_title(col, fontsize=12, fontweight="bold")
        ax[i].tick_params(labelsize=12)
    # Adjust the layout and display the plots
    plt.tight_layout()
    plt.show()
except ValueError:
    print('There are no Categorical Features')
```

#### Numerical Features Density Function | Función de Densidad de Caracteristicas Numéricas


```python
# Get the list of numerical columns in your DataFrame
numeric_columns = train.select_dtypes(include=['float', 'int', 'bool']).columns

# Define the plot size and the number of rows and columns in the grid
num_plots = len(numeric_columns)
rows = (num_plots + 1) // 2  # Calculate the number of rows needed (two plots per row)
cols = 2  # Two plots per row
_, axes = plt.subplots(nrows=rows, ncols=cols, figsize=(8 * cols, 6 * rows))

# Iterate through the numerical features and create the density plots
for i, feature_name in enumerate(numeric_columns):
    row_idx, col_idx = divmod(i, cols)  # Calculate the current row and column index
    sns.histplot(data=train, x=feature_name, kde=True, ax=axes[row_idx, col_idx])
    axes[row_idx, col_idx].set_title(f'{feature_name}')
    axes[row_idx, col_idx].set_xlabel('Value')
    axes[row_idx, col_idx].set_ylabel('Density')

# Adjust the spacing between subplots
plt.tight_layout()

# Show the plots
plt.show()
```

### Numercal Features Box Plots (Interactive) | Gráficos de Caja de Características Numéricas (Interactivos)


```python
# Melt the DataFrame to have all numerical variables in a single column
df_melted = pd.melt(train.select_dtypes(include=['float', 'int', 'bool']))

# Define a custom color palette
custom_colors = px.colors.qualitative.Plotly  # You can change this to any other palette

# Generate a combined box plot with the custom color palette
fig = px.box(df_melted, x='variable', y='value', color='variable', color_discrete_sequence=custom_colors)
fig.update_layout(title='Box Plots')
fig.show()
```

#### Numerical Feature Correlations | Correlaciones de Caracteristicas Numéricas


```python
train_ = train.select_dtypes(include=['float', 'int', 'bool']).columns

corr = train[train_].corr(method='spearman')
plt.figure(figsize=(12, 10))
sns.heatmap(corr, linewidth=0.5, annot=True, cmap="RdBu", vmin=-1, vmax=1)
```

#### Numerical Feature grouping | Grupos de Caracteristicas Numéricas


```python
train_ = train.select_dtypes(include=['float', 'int', 'bool']).columns

corr = train[train_].corr(method = "spearman")
link = linkage(squareform(1 - abs(corr)), "complete")
plt.figure(figsize = (8, 8), dpi = 400)
dendro = dendrogram(link, orientation='right', labels=train_)
plt.show()
```

## Mathematical Analysis | Análisis Matemático

#### Nulls Values | Valores Faltantes


```python
train.isna().sum().sort_values(ascending=False) / train.shape[0] * 100
```


```python
test.isna().sum().sort_values(ascending=False) / test.shape[0] * 100
```

#### Duplicate Values | Valores Duplicados


```python
train.duplicated().sum()
```


```python
test.duplicated().sum()
```

#### Distribution of Values in the Target | Distribución de Valores en el Objetivo


```python
train.Status.value_counts()
```

#### Kurtosis and Skew Analysis | Análisis de Curtosis y Sesgo

* Kurtosis:
    * ``Leptokurtic (positive kurtosis)``: Indicates that the tails of the distribution are heavier than they would be in a normal distribution. This implies that there are more extreme values present.
    * ``Platykurtic (negative kurtosis)``: Indicates that the tails of the distribution are lighter than they would be in a normal distribution. This suggests that there are fewer extreme values present.
* Skew:
    * ``Positive skew``: Indicates that the right tail of the distribution is longer or thicker than the left. Most of the data is concentrated on the left side and there are extreme values on the right side.
    * ``Negative skewness``: Indicates that the left tail of the distribution is longer or thicker than the right. Most of the data is concentrated on the right side and there are extreme values on the left side.

* Curtosis:
    * ``Leptocúrtica (positive curtosis)``: Indica que las colas de la distribución son más pesadas de lo que serían en una distribución normal. Esto implica que hay más valores extremos presentes.
    * ``Platicúrtica (negative curtosis)``: Indica que las colas de la distribución son más ligeras de lo que serían en una distribución normal. Esto sugiere que hay menos valores extremos presentes.
* Sesgo:
    * ``Sesgo positivo``: Indica que la cola derecha de la distribución es más larga o gruesa que la izquierda. La mayoría de los datos se concentran en la parte izquierda y hay valores extremos en la parte derecha.
    * ``Sesgo negativo``: Indica que la cola izquierda de la distribución es más larga o gruesa que la derecha. La mayoría de los datos se concentran en la parte derecha y hay valores extremos en la parte izquierda.


```python
train_ = train.select_dtypes(include=['float', 'int']).columns
test_ = test.select_dtypes(include=['float', 'int']).columns

pd.DataFrame({'train_kurtosis': train[train_].kurtosis(), 'test_kurtosis': test[test_].kurtosis()})
```


```python
pd.DataFrame({'train_skew': train[train_].skew(), 'test_skew': test[test_].skew()})
```

## Apply Transformations | Aplicar Transformaciones

#### Rename Target and Stage Values | Renombrar Valores del Target y Stage


```python
names_map = {
 'C': 0,
 'CL': 1,
 'D': 2,
 1.0: 'one',
 2.0:'two',
 3.0:'three',
 4.0:'four'
}
```


```python
train[TARGET] = train[TARGET].replace(names_map)
train['Stage'] = train['Stage'].replace(names_map)
test['Stage'] = test['Stage'].replace(names_map)
```

#### Convert categorical to dummy | Convertir categóricas a dummy


```python
Status = train[TARGET]

train = pd.get_dummies(train.drop(columns='Status'), drop_first=True)
train[TARGET] = Status

test = pd.get_dummies(test, drop_first=True)
```

---

> ### Optional | Opcional

#### There are two interesting options for ``Age`` | Hay dos opciones interesantes para ``Age``:

* **Convert days to years | Transformar días a años**
* Use the standard scale for cancer patients | Utilizar la escala standard para pacientes con cáncer
    * [Standard Populations - 19 Age Groups](https://seer.cancer.gov/stdpopulations/stdpop.19ages.html)


```python
# Convert days to years
# train['Age'] = train['Age'] // 365.25
# test['Age'] = test['Age'] // 365.25
```

#### If the Data Does not Follow a Normal Distribution | Si los Datos no Siguen una Distribución Normal

* Apply Box-Cox | Aplicar Box-Cox

*This transformation also helps us with heteroscedasticity and outliers* | *Esta transformación también nos ayuda con la heterocedasticidad y  valor atípicos*


```python
train_ = train.drop(columns=[TARGET])
numeric_cols = train_.select_dtypes(include=['int64', 'float64']).columns

for col in numeric_cols:
    train[col], lambda_ = boxcox(train_[col])
    test[col] = boxcox(test[col], lambda_)
```

* OR Apply Standard Scaler | O Aplicar Escalador Estándar


```python
# scaler = StandardScaler()

# train_ = train.drop(columns=[TARGET])

# numeric_cols = train_.select_dtypes(include=['int64', 'float64']).columns

# train[numeric_cols] = scaler.fit_transform(train[numeric_cols])
# test[numeric_cols] = scaler.fit(test[numeric_cols])

# del(train_)
```

* Or MinMaxScaler | O Escala min/max


```python
# scaler = MinMaxScaler()

# train_ = train.drop(columns=[TARGET])

# numeric_cols = train_.select_dtypes(include=['int64', 'float64']).columns

# scaler.fit(train[numeric_cols])

# train[numeric_cols] = scaler.transform(train[numeric_cols])
# test[numeric_cols] = scaler.transform(test[numeric_cols])

# del(train_)
```

---

> **Important:** *It is a good idea to regenerate the previous graphs and statistics once the transformations have been applied.* | *Es una buena idea volver a generar los gráficos y estadisticos anteriores una vez aplicadas las transformaciones*

<h1 id="goto5" style="background-color:orangered;font-family:cursive;color:black;font-size:250%;text-align:center;border-radius: 50px 50px;">Individual Modeling</h1>

[Back to Table of Contents](#goto0)

#### Metric Models | Metrica de Modelos


```python
def apply_metrics(y_test, y_pred):
    """
    Calculates metrics.

    Parameters:
        y_test (array-like): True values of the target variable (ground truth).
        y_pred (array-like): Predicted values by the model.

    Returns:
        pandas DataFrame: A DataFrame containing the metrics and their respective values.
    """

    Log_Loss = log_loss(y_test, y_pred)

    metric_df = pd.DataFrame({
        'Metric': ['Log Loss Error'],
        'Value': [Log_Loss]
    })

    return metric_df
```

#### Create Dependent Variable and Array of Independent Variables | Crear Variable Dependiente y Matriz de Variables Independientes


```python
y = train[TARGET]
X = train.drop(columns=[TARGET])

X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, test_size=0.1, random_state=42)
```

#### Apply Weights to '``Status``' Classes | Aplicar Pesos a las Clases de '``Status``'


```python
class_weights = compute_class_weight(class_weight='balanced', classes=np.unique(y_train), y=y_train)

weights_dict = {cls: weight for cls, weight in zip(np.unique(y_train), class_weights)}
```

---

> ### Optional | Opcional

#### Balancing minority classes | Equilibrar las clases minoritarias

* Oversampling | Sobremuestreo


```python
# X_train, y_train = ADASYN(random_state=42).fit_resample(X_train, y_train)
```


```python
# y_train.value_counts()
```

* Undersampling | Submuestreo


```python
# cc = ClusterCentroids()
# X_train, y_train = cc.fit_resample(X_train, y_train)
```


```python
# y_train.value_counts()
```

---

<h2 id="goto5a" style="background-color:darkorange;font-family:cursive;color:black;font-size:150%;text-align:center;border-radius: 50px 50px;">Logistic Model</h2>

[Back to Models](#goto5)

#### Create and train the model | Crear y Entrenar el Modelo


```python
logistic_model = LogisticRegression(multi_class='multinomial', solver='lbfgs', class_weight=weights_dict)
logistic_model.fit(X_train, y_train)
```

#### Predict Test Values | Predecir Valores de Test


```python
y_pred = logistic_model.predict_proba(X_test)
```

#### Measure Performance | Medir el Rendimiento


```python
apply_metrics(y_test, y_pred)
```

---

> ### Optional | Opcional

#### More Precise Calibration of Probabilities | Calibración más Precisa de las Probabilidades


```python
calibrated_logistic = CalibratedClassifierCV(logistic_model, method='sigmoid') # isotonic
calibrated_logistic.fit(X_train, y_train)
```

#### Calibrated Class Probabilities | Probabilidades de Clase Calibradas


```python
calibrated_y_pred = calibrated_logistic.predict_proba(X_test)
```

#### Measure Performance | Medir el Rendimiento


```python
apply_metrics(y_test, calibrated_y_pred)
```

#### Graph Importance of Features | Graficar Importancia de las Variables


```python
sns.barplot(x = list(abs(logistic_model.coef_[0])), y = list(X.columns))
```

---

#### Submission and Scoring | Presentación y Puntuación


```python
submission = pd.read_csv('sample_submission.csv')

submission[submission.columns[0]] = test_id

# submission[submission.columns[1:]] = logistic_model.predict_proba(test)
submission[submission.columns[1:]] = calibrated_logistic.predict_proba(test)
```


```python
submission.to_csv('logistic.csv', index = False)
```

<p style="color: red; font-size: 150%"><b>
CONCLUSION: Logistic Model (Box-Cox) is a Bad option.
</b></p>

<p style="color: red; font-size: 150%"><b>
>> Scoring Log Loss: 0.5674
</b></p>

<h2 id="goto5b" style="background-color:darkorange;font-family:cursive;color:black;font-size:150%;text-align:center;border-radius: 50px 50px;">Extreme Gradient Boosting Model</h2>

<div style="font-family: cursive">

[Back to Models](#goto5)

</div>

#### Create and train the model | Crear y Entrenar el Modelo


```python
xgb_model = XGBClassifier(objective='multi:softmax', num_class=3) # multi_logloss y multi:softprob
xgb_model.fit(X_train, y_train)
```

#### Predict Test Values | Predecir Valores de Test


```python
y_pred = xgb_model.predict_proba(X_test)
```

#### Measure Performance | Medir el Rendimiento


```python
apply_metrics(y_test, y_pred)
```

---

> ### Optional | Opcional

#### More Precise Calibration of Probabilities | Calibración más Precisa de las Probabilidades


```python
calibrated_xgb = CalibratedClassifierCV(xgb_model, method='sigmoid')
calibrated_xgb.fit(X_train, y_train)
```

#### Calibrated Class Probabilities | Probabilidades de Clase Calibradas


```python
calibrated_y_pred = calibrated_xgb.predict_proba(X_test)
```

#### Measure Performance | Medir el Rendimiento


```python
apply_metrics(y_test, calibrated_y_pred)
```

#### Graph Importance of Features | Graficar Importancia de las Variables


```python
sns.barplot(x = list(xgb_model.feature_importances_), y = list(X.columns))
```

---

#### Submission and Scoring | Presentación y Puntuación


```python
submission = pd.read_csv('sample_submission.csv')

submission[submission.columns[0]] = test_id

# submission[submission.columns[1:]] = xgb_model.predict_proba(test)
submission[submission.columns[1:]] = calibrated_xgb.predict_proba(test)
```


```python
submission.to_csv('xgboost.csv', index = False)
```

<p style="color: orange; font-size: 150%"><b>
CONCLUSION: XGB Model (Box-Cox) is a Regular option.
</b></p>

<p style="color: orange; font-size: 150%"><b>
>> Scoring Log Loss: 0.44352
</b></p>

<h2 id="goto5c" style="background-color:darkorange;font-family:cursive;color:black;font-size:150%;text-align:center;border-radius: 50px 50px;">Light Gradient Boosting Machine</h2>

<div style="font-family: cursive">

[Back to Models](#goto5)

</div>

#### Create and train the model | Crear y Entrenar el Modelo


```python
lgb_model = LGBMClassifier(objective='multiclass', num_class=3, class_weight=weights_dict) # multi_logloss
lgb_model.fit(X_train, y_train)
```

#### Predict Test Values | Predecir Valores de Test


```python
y_pred = lgb_model.predict_proba(X_test)
```

#### Measure Performance | Medir el Rendimiento


```python
apply_metrics(y_test, y_pred)
```

---

> ### Optional | Opcional

#### More Precise Calibration of Probabilities | Calibración más Precisa de las Probabilidades


```python
calibrated_lgb = CalibratedClassifierCV(lgb_model, method='sigmoid')
calibrated_lgb.fit(X_train, y_train)
```

#### Calibrated Class Probabilities | Probabilidades de Clase Calibradas


```python
calibrated_y_pred = calibrated_lgb.predict_proba(X_test)
```

#### Measure Performance | Medir el Rendimiento


```python
apply_metrics(y_test, calibrated_y_pred)
```

#### Graph Importance of Features | Graficar Importancia de las Variables


```python
sns.barplot(x = list(lgb_model.feature_importances_), y = list(lgb_model.feature_name_))
```

---

#### Submission and Scoring | Presentación y Puntuación


```python
submission = pd.read_csv('sample_submission.csv')

submission[submission.columns[0]] = test_id

# submission[submission.columns[1:]] = lgbm_model.predict_proba(test)
submission[submission.columns[1:]] = calibrated_lgb.predict_proba(test)
```


```python
submission.to_csv('lgbm.csv', index = False)
```

<p style="color: green; font-size: 150%"><b>
CONCLUSION: LGBM Model (Box-Cox) is a Good option.
</b></p>

<p style="color: green; font-size: 150%"><b>
>> Scoring Log Loss: 0.43799
</b></p>

<h2 id="goto5d" style="background-color:darkorange;font-family:cursive;color:black;font-size:150%;text-align:center;border-radius: 50px 50px;">CAT Boosting Model</h2>

<div style="font-family: cursive">

[Back to Models](#goto5)

</div>

#### Create and train the model | Crear y Entrenar el Modelo


```python
cat_model = CatBoostClassifier(loss_function='MultiClass', class_weights=weights_dict)
cat_model.fit(X_train, y_train, verbose=False)
```

#### Predict Test Values | Predecir Valores de Test


```python
y_pred = cat_model.predict_proba(X_test)
```

#### Measure Performance | Medir el Rendimiento


```python
apply_metrics(y_test, y_pred)
```

---

> ### Optional | Opcional

#### More Precise Calibration of Probabilities | Calibración más Precisa de las Probabilidades


```python
# The output is deleted to make the notebook more readable
calibrated_cat = CalibratedClassifierCV(cat_model, method='sigmoid')
calibrated_cat.fit(X_train, y_train)
```

#### Calibrated Class Probabilities | Probabilidades de Clase Calibradas


```python
calibrated_y_pred = calibrated_cat.predict_proba(X_test)
```

#### Measure Performance | Medir el Rendimiento


```python
apply_metrics(y_test, calibrated_y_pred)
```

#### Graph Importance of Features | Graficar Importancia de las Variables


```python
sns.barplot(x = list(cat_model.feature_importances_), y = list(cat_model.feature_names_))
```

---

#### Submission and Scoring | Presentación y Puntuación


```python
submission = pd.read_csv('sample_submission.csv')

submission[submission.columns[0]] = test_id

# submission[submission.columns[1:]] = cat_model.predict_proba(test)
submission[submission.columns[1:]] = calibrated_cat.predict_proba(test)
```


```python
submission.to_csv('cat.csv', index = False)
```

<p style="color: green; font-size: 150%"><b>
CONCLUSION: CAT Model (Box-Cox) is a Good option.
</b></p>

<p style="color: green; font-size: 150%"><b>
>> Scoring Log Loss: 0.43495
</b></p>

<h2 id="goto5e" style="background-color:darkorange;font-family:cursive;color:black;font-size:200%;text-align:center;border-radius: 50px 50px;">Neural Network Model</h2>

<div style="font-family: cursive">

[Back to Models](#goto5)

</div>

#### Create and train the model | Crear y Entrenar el Modelo


```python
nn_model = MLPClassifier(random_state=42, max_iter=300)
nn_model.fit(X_train, y_train)
```

#### Predict Test Values | Predecir Valores de Test


```python
y_pred = nn_model.predict_proba(X_test)
```

#### Measure Performance | Medir el Rendimiento


```python
apply_metrics(y_test, y_pred)
```

---

> ### Optional | Opcional

#### More Precise Calibration of Probabilities | Calibración más Precisa de las Probabilidades


```python
calibrated_nn = CalibratedClassifierCV(nn_model, method='sigmoid')
calibrated_nn.fit(X_train, y_train)
```

#### Calibrated Class Probabilities | Probabilidades de Clase Calibradas


```python
calibrated_y_pred = calibrated_nn.predict_proba(X_test)
```

#### Measure Performance | Medir el Rendimiento


```python
apply_metrics(y_test, calibrated_y_pred)
```

---

#### Submission and Scoring | Presentación y Puntuación


```python
submission = pd.read_csv('sample_submission.csv')

submission[submission.columns[0]] = test_id

# submission[submission.columns[1:]] = cat_model.predict_proba(test)
submission[submission.columns[1:]] = calibrated_nn.predict_proba(test)
```


```python
submission.to_csv('nn.csv', index = False)
```

<p style="color: orange; font-size: 150%"><b>
CONCLUSION: NN Model (Box-Cox) is a Regular option.
</b></p>

<p style="color: orange; font-size: 150%"><b>
>> Scoring Log Loss: 0.51842
</b></p>

<h1 id="goto6" style="background-color:orangered;font-family:cursive;color:black;font-size:250%;text-align:center;border-radius: 50px 50px;">Stacking Models</h1>

<div style="font-family: cursive">

[Back to Table of Contents](#goto0)

</div>

#### Create a ``Voting Classifier`` meta estimator with the previous boosting models and a ``Neural Network`` incorporating the predictions of the boosting models to the original Dataset.

#### Crea un meta estimador ``Clasificador de Votaciones`` con los modelos boosting anteriores y una ``Red Neuronal`` incorporando las predicciones de los modelos boosting al Dataset original.

<h2 id="goto6a" style="background-color:darkorange;font-family:cursive;color:black;font-size:200%;text-align:center;border-radius: 50px 50px;">Voting Classifier Model</h2>

<div style="font-family: cursive">

[Back to Stacking Models](#goto6)

</div>

#### Create and train the model | Crear y Entrenar el Modelo


```python
# The output is deleted to make the notebook more readable
voting_clf = VotingClassifier(estimators=[('xgb', calibrated_xgb), ('lgb', calibrated_lgb), ('cat', calibrated_cat)], voting='soft') # ('log', logistic_model)
voting_clf.fit(X_train, y_train)
```

#### Predict Test Values | Predecir Valores de Test


```python
y_pred = voting_clf.predict_proba(X_test)
```

#### Measure Performance | Medir el Rendimiento


```python
apply_metrics(y_test, y_pred)
```

---

### Submission and Scoring | Presentación y Puntuación


```python
# Test a submission
submission = pd.read_csv('sample_submission.csv')

submission[submission.columns[0]] = test_id

submission[submission.columns[1:]] = voting_clf.predict_proba(test)
```


```python
submission.to_csv('stack_cls.csv', index = False)
```

<p style=color:green;font-size:150%><b>
CONCLUSION: Voting Classifier (Box-Cox) is a Best option.
</b></p>

<p style=color:green;font-size:150%><b>
* Scoring Log Loss: 0.43386
</b></p>

<h1 id="goto7" style="background-color:orangered;font-family:cursive;color:black;font-size:250%;text-align:center;border-radius: 50px 50px;">Conclusions</h1>

<div style="font-family: cursive">

[Back to Table of Contents](#goto0)

</div>

**The Best Result:**

* The Voting Model presents the best performance:
    * Box-Cox transformation
    * Calibrated probabilities
    * Weight balancing
  
**It Might be Interesting:**

* Try a Special Treatment for the 'Age' Feature [**✓**]
* Standardize features [**✓**]
* Use the Original Dataset as Test [**✓**]
* Use Class Weights [**✓**]
* Convert 'Stage' to categorical [**✓**]
* Apply Class Balancing to 'CL' [**X**]
* Use NN Model as Metamodel [**X**]
* Try a Special Treatment for the 'N-days' Feature
* Customize Model Hyperparameters
