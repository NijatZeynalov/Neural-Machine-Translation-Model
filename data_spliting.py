from pickle import load
from pickle import dump
from numpy.random import shuffle

#load a clean dataset
def load_cl_sentences(fn):
    return load(open(fn, 'rb'))

#save a list of clean sentences to file
def save_cl_data(sn, fn):
    dump(sn, open(fn, 'wb'))
    print('saved: %s' %fn)

#load
raw_ds = load_cl_sentences('english-azerbaijani.pkl')

n_sn = 2215
ds = raw_ds[:n_sn, :]

#random shuffle
shuffle(ds)

train, test = ds[:2000], ds[2000:]

#save
save_cl_data(ds, 'english-azerbaijani-both.pkl')
save_cl_data(ds, 'english-azerbaijani-train.pkl')
save_cl_data(ds, 'english-azerbaijani-test.pkl')