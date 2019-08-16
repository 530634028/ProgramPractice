


# import numpy as np
import keras
from keras.models import Sequential
from keras.layers import Dense



# 对于具有 10 个类的单输入模型（多分类分类）：
model = Sequential()
model.add(Dense(32, activation='relu', input_dim=100))
model.add(Dense(10, activation='softmax'))
model.compile(optimizer='rmsprop',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# 生成虚拟数据
import numpy as np
data = np.random.random((1000, 100))
labels = np.random.randint(10, size=(1000, 1))

# 将标签转换为分类的 one-hot 编码
one_hot_labels = keras.utils.to_categorical(labels, num_classes=10)

# 训练模型，以 32 个样本为一个 batch 进行迭代
model.fit(data, one_hot_labels, epochs=10000, batch_size=32)





# # 对于具有 2 个类的单输入模型（二进制分类）：
# model = Sequential()
# model.add(Dense(32, activation='relu', input_dim=100))
# model.add(Dense(1, activation='sigmoid'))
# model.compile(optimizer='rmsprop',
#               loss='binary_crossentropy',
#               metrics=['accuracy'])
#
# # 生成虚拟数据
# import numpy as np
# data = np.random.random((1000, 100))
# labels = np.random.randint(2, size=(1000, 1))
#
# # 训练模型，以 32 个样本为一个 batch 进行迭代
# model.fit(data, labels, epochs=10, batch_size=32)