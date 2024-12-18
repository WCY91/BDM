from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler,normalize
import tensorflow as tf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dropout,Layer,Flatten,LayerNormalization,TextVectorization,Dense,LeakyReLU,Input,Reshape
from sklearn.metrics  import mean_squared_error

salaries_data = pd.read_csv("cleaned_dataset/salaries_and_scores.csv")
salaries_data=salaries_data.drop('playerName',axis=1)
time_y  = salaries_data["MP"]
salaries_data.iloc[:, 0:17] = StandardScaler().fit_transform(salaries_data.iloc[:, 0:17])
print(salaries_data)

X = salaries_data[['PTS', 'PF', 'TOV', 'AST', 'STL','BLK','TRB','FG','FGA']].values
X = normalize(X, norm="l1")
money_y= y  = salaries_data["target"]
time_y= time_y / 4 #make time label
time_value = salaries_data['MP'].values
X_train, X_test, y_train, y_test, time_y_train, time_y_test,time_value_y_train,time_value_y_test = train_test_split(
    X, y, time_y,time_value,test_size=0.2, random_state=3
)

# method use the transformer encoder to predict
class MultiHeadSelfAttention(Layer):
    def __init__(self,embed_dim,num_heads=8):
        super(MultiHeadSelfAttention,self).__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.projection_dim = embed_dim // num_heads
        self.query_dense = Dense(embed_dim)
        self.key_dense = Dense(embed_dim)
        self.value_dense = Dense(embed_dim)
        self.combine_dense = Dense(embed_dim)

    def attention(self,query,key,value):
        score = tf.matmul(query,key,transpose_b=True)
        dim_key = tf.cast(tf.shape(key)[-1],tf.float32)
        scaled_score = score / tf.math.sqrt(dim_key)
        weights = tf.nn.softmax(scaled_score,axis=1)
        output = tf.matmul(weights,value)
        return output,weights
    
    def split_heads(self,x,batch_size):
        x = tf.reshape(x, (batch_size,-1,self.num_heads,self.projection_dim))
        return tf.transpose(x,perm=[0,2,1,3])
    
    def call(self,inputs):
        batch_size = tf.shape(inputs)[0]
        query = self.query_dense(inputs)
        key = self.key_dense(inputs)
        value = self.value_dense(inputs)

        query = self.split_heads(query,batch_size)
        key = self.split_heads(key,batch_size)
        value = self.split_heads(value,batch_size)

        output, weights = self.attention(query,key,value)
        output = tf.transpose(output, perm=[0,2,1,3])
        concat_attention = tf.reshape(output, (batch_size, -1, self.embed_dim))
        output = self.combine_dense(concat_attention)
        return output
    
class TransformerBlock(Layer):

  def __init__(self,embed_dim,num_heads,ff_dim,rate=0.1):
    super(TransformerBlock,self).__init__()
    self.att= MultiHeadSelfAttention(embed_dim,num_heads)
    self.fnn = tf.keras.Sequential(
        [Dense(ff_dim,activation='relu'),Dense(embed_dim)]
    )
    self.layernorm1 = LayerNormalization(epsilon = 1e-6)
    self.layernorm2 = LayerNormalization(epsilon=1e-6)
    self.dropout1 = Dropout(rate)
    self.dropout2 = Dropout(rate)

  def call(self,inputs,training):
    attn_output = self.att(inputs)
    attn_output = self.dropout1(attn_output,training=training)
    out1 = self.layernorm1(inputs+attn_output)
    ffn_output=self.fnn(out1)
    ffn_output=self.dropout2(ffn_output,training=training)
    return self.layernorm2(out1+ffn_output)
  
class TransformerEncoder(Layer):
  def __init__(self,num_layers,embed_dim,num_heads,ff_dim,rate=0.1):
    super(TransformerEncoder,self).__init__()
    self.num_layers = num_layers
    self.embed_dim = embed_dim
    self.enc_layers = [TransformerBlock(embed_dim,num_heads,ff_dim,rate) for _ in range(num_layers)]
    self.dropout = Dropout(rate)

  def call(self,inputs,training=False):
    x = inputs
    for i in range(self.num_layers):
      x = self.enc_layers[i](x,training = training)
    return x


embed_dim = 32
num_heads = 8
ff_dim = 128
num_layers = 2
time_classes = len(set(time_y))
time_y_train = tf.keras.utils.to_categorical(time_y_train,num_classes= time_classes)
time_y_test_categorical = tf.keras.utils.to_categorical(time_y_test, num_classes=time_classes)
money_classes = 25

def build_regression_model():
  model = Sequential([
    Input(shape=(9,)),        
    Reshape((9, 1)),          
    Dense(embed_dim),        
    TransformerEncoder(num_layers, embed_dim, num_heads, ff_dim),
    Flatten(),
    Dense(128, activation='relu'),
    LeakyReLU(),
    Dropout(0.3),
    Dense(64, activation='relu'),
    Dropout(0.3),
    Dense(32,activation='relu'),
    Dense(16,activation='relu'),
    Dense(1)
  ])
  model.compile(optimizer='adam', loss='mse', metrics=['mse','mae'])
  return model

def build_classify_model(class_num):
  model = Sequential([
    Input(shape=(9,)),        
    Reshape((9, 1)),          
    Dense(embed_dim),        
    TransformerEncoder(num_layers, embed_dim, num_heads, ff_dim),
    Flatten(),
    Dense(128, activation='relu'),
    LeakyReLU(),
    Dropout(0.3),
    Dense(64, activation='relu'),
    Dropout(0.3),
    Dense(32,activation='relu'),
    Dense(16),
    Dense(class_num, activation='softmax')
  ])
  model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
  return model


# time_clf_model =  build_classify_model(time_classes)
# history = time_clf_model.fit(X_train, time_y_train, validation_split=0.1, epochs=80, batch_size=32)
# y_pred = time_clf_model.predict(X_test)
# loss, accuracy = time_clf_model.evaluate(X_test, time_y_test)
# time_clf_model.save('weights/time_clf_transformer.h5')

# time value predict model
# X = salaries_data[['PTS', 'PF', 'TOV', 'AST', 'STL','BLK','TRB','FG','FGA']].values
# X = normalize(X, norm="l1")
# y = salaries_data['MP'].values
# X_train, X_test, y_train, y_test = train_test_split(
#     X, y,test_size=0.2, random_state=3
# )
# time_model = build_regression_model()
# history = time_model.fit(X_train,y_train,validation_split=0.1, epochs=80, batch_size=32)
# y_pred = time_model.predict(X_test)
# plt.figure(figsize=(8, 6))
# plt.scatter(y_test, y_pred, alpha=0.7)
# plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', linewidth=2)
# plt.xlabel('Actual Values')
# plt.ylabel('Predicted Values')
# plt.title('Actual vs Predicted')
# plt.show()
# time_model.save('weights/time_reg_transformer.h5')






