import pandas as pd
import numpy as np
import tensorflow as tf
import tempfile

training = pd.read_csv("train.csv")
potential_training = training[['Pclass', 'Sex', 'Age', 'Fare', 'SibSp', 'Parch', 'Survived']]

test_data = pd.read_csv("test.csv")
test_features = test_data[['Pclass', 'Sex', 'Age', 'Fare', 'SibSp', 'Parch']]

test_features.Age.fillna(test_features.Age.median(), inplace=True)
test_features.Fare.fillna(test_features.Fare.median(), inplace=True)
test_features.Sex.fillna(test_features.Sex.mode(), inplace = True)

test_features.Age -= test_features.Age.min()
test_features.Age /= (test_features.Age.max()+0.001)

test_features.Fare -= test_features.Fare.min()
test_features.Fare /= (test_features.Fare.max()+0.001)

test_features.SibSp -= test_features.SibSp.min()
test_features.SibSp /= (test_features.SibSp.max()+0.001)

test_features.Parch -= test_features.Parch.min()
test_features.Parch /= (test_features.Parch.max()+0.001)

test_features.Pclass =  test_features.Pclass.astype(str)



no_missing = potential_training.dropna(how='any')
#no_missing["Pclass"].apply(lambda x: str(x))

no_missing.Age -= no_missing.Age.min()
no_missing.Age /= (no_missing.Age.max()+0.001)

no_missing.Fare -= no_missing.Fare.min()
no_missing.Fare /= (no_missing.Fare.max()+0.001)

no_missing.SibSp -= no_missing.SibSp.min()
no_missing.SibSp /= (no_missing.SibSp.max()+0.001)

no_missing.Parch -= no_missing.Parch.min()
no_missing.Parch /= (no_missing.Parch.max()+0.001)

#test_features.Pclass =  test_features.Pclass.astype(str)



no_missing.Pclass = no_missing.Pclass.astype(str)

print no_missing

msk = np.random.rand(len(no_missing)) < 0.80
df_train = no_missing[msk]
df_test = no_missing[~msk]


CATEGORICAL_COLUMNS = ['Pclass', 'Sex']
CONTINUOUS_COLUMNS = ['Age', 'Fare', 'SibSp', 'Parch']
LABEL_COLUMN = "Survived"

def input_fn(df):
      # Creates a dictionary mapping from each continuous feature column name (k) to
      # the values of that column stored in a constant Tensor.
      continuous_cols = {k: tf.constant(df[k].values)
                         for k in CONTINUOUS_COLUMNS}
      # Creates a dictionary mapping from each categorical feature column name (k)
      # to the values of that column stored in a tf.SparseTensor.
      categorical_cols = {k: tf.SparseTensor(
                      indices=[[i, 0] for i in range(df[k].size)],
                      values=df[k].values,
                      shape=[df[k].size, 1])
                          for k in CATEGORICAL_COLUMNS}
      # Merges the two dictionaries into one.
      feature_cols = dict(continuous_cols.items() + categorical_cols.items())
      # Converts the label column into a constant Tensor.
      label = tf.constant(df[LABEL_COLUMN].values)
      # Returns the feature columns and the label.
      return feature_cols, label


def input_fn_t(df):
      # Creates a dictionary mapping from each continuous feature column name (k) to
      # the values of that column stored in a constant Tensor.
      continuous_cols = {k: tf.constant(df[k].values)
                         for k in CONTINUOUS_COLUMNS}
      # Creates a dictionary mapping from each categorical feature column name (k)
      # to the values of that column stored in a tf.SparseTensor.
      categorical_cols = {k: tf.SparseTensor(
                      indices=[[i, 0] for i in range(df[k].size)],
                      values=df[k].values,
                      shape=[df[k].size, 1])
                          for k in CATEGORICAL_COLUMNS}
      # Merges the two dictionaries into one.
      feature_cols = dict(continuous_cols.items() + categorical_cols.items())
      # Converts the label column into a constant Tensor.


      return feature_cols

def train_input_fn():
    return input_fn(df_train)

def eval_input_fn():
    return input_fn(df_test)

def eval_input_fn_t():
    return input_fn_t(df_test)


Sex = tf.contrib.layers.sparse_column_with_keys(column_name="Sex", keys=["Female", "Male"])
Pclass = tf.contrib.layers.sparse_column_with_keys(column_name="Pclass", keys=["1", "2", "3"])

Age = tf.contrib.layers.real_valued_column("Age")
Fare = tf.contrib.layers.real_valued_column("Fare")

model_dir = tempfile.mkdtemp()
m = tf.contrib.learn.LinearClassifier(feature_columns=[Sex,Pclass,Age,Fare],
                                      optimizer=tf.train.FtrlOptimizer(
                                                learning_rate=0.01,
                                                l1_regularization_strength=1.0,
                                                l2_regularization_strength=1.0),
                                        model_dir=model_dir)

print "Fitting"
m.fit(input_fn=train_input_fn, steps=5000)
print "Finished"


results = m.evaluate(input_fn=eval_input_fn, steps=1)
for key in sorted(results):
      print "%s: %s" % (key, results[key])

ids = test_data.PassengerId.values
df_test = test_features

preds = m.predict(input_fn=eval_input_fn_t)

print len(preds)

f = open("submission.csv","w")
f.write("PassengerId,Survived\n")
for i,p in zip(ids,preds):
      print i,p
      f.write("%d,%d\n"% (i,p))

f.close()
