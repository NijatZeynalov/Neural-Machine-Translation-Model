#import libraries
import matplotlib.pyplot as plt
from pickle import load
import tensorflow as tf
from numpy import array
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.utils import to_categorical
#from tensorflow.keras.utils.vis_utils import plot_model
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Embedding
from tensorflow.keras.layers import RepeatVector
from tensorflow.keras.layers import TimeDistributed
from tensorflow.keras.callbacks import ModelCheckpoint

#load a clean dataset
def load_cl_sn(fn):
    return load(open(fn, 'rb'))

#fit a tokenizer
def cr_tokenizer(lns):
    tokenizer = Tokenizer()
    tokenizer.fit_on_texts(lns)
    return tokenizer

#max sentence length
def max_length(lns):
    return max(len(ln.split()) for ln in lns)

#encode and pad sequences
def encode_seqs(tokenizer, lng, lns):
    X = tokenizer.texts_to_sequences(lns)
    X = pad_sequences(X, maxlen=lng, padding='post')
    return X

#one hot encode target sequence

def encode_output(seqs, vocab_size):
    ylist = list()
    for sq in seqs:
        encoded = to_categorical(sq, num_classes=vocab_size)
        ylist.append(encoded)
    y = array(ylist)
    y = y.reshape(seqs.shape[0], seqs.shape[1], vocab_size)
    return y

#define NMT model
def def_model(src_vcb, tar_vcb, src_ts, tar_ts, n_units):
    model = Sequential()
    model.add(Embedding(src_vcb, n_units, input_length=src_ts, mask_zero=True))
    model.add(LSTM(n_units))
    model.add(RepeatVector(tar_ts))
    model.add(LSTM(n_units, return_sequences=True))
    model.add(TimeDistributed(Dense(tar_vcb, activation='softmax')))
    # compile model
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics = ['accuracy'])
    # summarize defined model
    model.summary()
    tf.keras.utils.plot_model(model, to_file='model.png', show_shapes=True)
    return model
# load datasets
dataset = load_cl_sn('english-azerbaijani-both.pkl')
train = load_cl_sn('english-azerbaijani-train.pkl')
test = load_cl_sn('english-azerbaijani-test.pkl')
# prepare english tokenizer
eng_tokenizer = cr_tokenizer(dataset[:, 0])
eng_vocab_size = len(eng_tokenizer.word_index) + 1
eng_length = max_length(dataset[:, 0])
print('English Vocabulary Size: %d' % eng_vocab_size)
print('English Max Length: %d' % (eng_length))
# prepare german tokenizer
aze_tokenizer = cr_tokenizer(dataset[:, 1])
aze_vocab_size = len(aze_tokenizer.word_index) + 1
aze_length = max_length(dataset[:, 1])
print('Azerbaijani Vocabulary Size: %d' % aze_vocab_size)
print('Azerbaijani Max Length: %d' % (aze_length))
# prepare training data
trainX = encode_seqs(aze_tokenizer, aze_length, train[:, 1])
trainY = encode_seqs(eng_tokenizer, eng_length, train[:, 0])
trainY = encode_output(trainY, eng_vocab_size)
# prepare validation data
testX = encode_seqs(aze_tokenizer, aze_length, test[:, 1])
testY = encode_seqs(eng_tokenizer, eng_length, test[:, 0])
testY = encode_output(testY, eng_vocab_size)
# define model
model = def_model(aze_vocab_size, eng_vocab_size, aze_length, eng_length, 256)
# fit model
checkpoint = ModelCheckpoint('model.h5', monitor='val_loss', verbose=1, save_best_only=True, mode='min')
history = model.fit(trainX, trainY, epochs=30, batch_size=64, validation_data=(testX, testY), callbacks=[checkpoint], verbose=2)

acc = history.history['accuracy']
val_acc = history.history['val_accuracy']
loss = history.history['loss']
val_loss = history.history['val_loss']

epochs = range(len(acc))

plt.plot(epochs, acc, 'r', label='Training accuracy')
plt.plot(epochs, val_acc, 'b', label='Validation accuracy')
plt.title('Training and validation accuracy')
plt.legend(loc=0)
plt.figure()
plt.show()