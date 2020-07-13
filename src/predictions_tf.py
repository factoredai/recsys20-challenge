import tensorflow as tf
import pandas as pd
import os
from collections import defaultdict
import numpy as np
import pickle
from tqdm import tqdm
from utils.dataset import (get_csvs, names_to_idx, create_type_list,
                           build_dataset)
import tensorflow.keras
from utils.layers import FM_pro, EncoderLayer
from utils.optimizers import WarmMaxConvSchedule


print('working_dir', os.getcwd())

model_path = './models/final_model_reduced_50.4/'
submission_path = './data/submission-final-complete/'
BATCH_SIZE = 2048

checkpoint_path = os.path.join(model_path, 'tf_model01-1.96.h5')
h_params_path = os.path.join(model_path, "hparams.pickle")


model = tf.keras.models.load_model(
    checkpoint_path, custom_objects={
        "FM_pro": FM_pro,
        "EncoderLayer": EncoderLayer,
        "WarmMaxConvSchedule": WarmMaxConvSchedule
    })

with open(h_params_path, "rb") as input_file:
    hparams = pickle.load(input_file)

labels_to_pred = list(hparams["labels"].keys())

print('labels_to_pred', labels_to_pred)
# dataloaders
filenames_submission = get_csvs(submission_path)

types_list, names2idx = create_type_list(filenames_submission[0], labels_to_pred)
col_idx = names_to_idx(filenames_submission[0], labels_to_pred)


dataset_sub = build_dataset(filenames_submission, types_list, col_idx,
                            hparams["inputs"], mode="submission",
                            epochs=1, batch_size=BATCH_SIZE)


decoder = np.vectorize(lambda x: x.decode('UTF-8'))
df = pd.DataFrame(columns=['id1', 'id2'] + labels_to_pred)
rows = defaultdict(list)

for batch in tqdm(dataset_sub):
    ids = batch[0]['id_cols']
    rows['id1'] += list(decoder(ids.numpy()[:, 1]))
    rows['id2'] += list(decoder(ids.numpy()[:, 2]))

df['id1'] = rows['id1']
df['id2'] = rows['id2']

predictions = model.predict(dataset_sub, verbose=1)

for label in labels_to_pred:
    df[label] = predictions[label][:, 0]

for label in labels_to_pred:
    df[['id1', 'id2', label]].to_csv('preds_' + label + '.csv', index=False,
                                     header=False)
