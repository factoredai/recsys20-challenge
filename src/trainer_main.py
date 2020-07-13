import tensorflow as tf
import os
import sys
import json
import pickle
from argparse import ArgumentParser
from utils.builder import compile_model
from utils.dataset import (get_csvs, names_to_idx, create_type_list,
                           build_dataset)

# Setup parameters
parser = ArgumentParser()
parser.add_argument(
    '--model_version',
    type=str
)

# INPUTS
model_name = "final_model_reduced"
model_version = parser.parse_args().model_version
train_data = "train-final-complete"
val_data = "val-final-complete"

# training hparams
BATCH_SIZE = 2048
EPOCHS = 6
patience = 12
n_vals_per_epoch = 14

model_folder_name = model_name + '_' + model_version

hparams = {"model_name": "final_model_best_50_0",
           "inputs": [
               "txt", "id_cols", "author_bool_cols", "engager_bool_cols", "interaction_bool_cols",
               "tweet_num_cols", "tweet_cat_cols", "author_num_cols", "hsh", "dom",
               "engager_num_cols", "interaction_num_cols", "author_cat_cols", "engager_cat_cols",
               "bucket", "engager_topic_cat", "engager_topic_count", "engager_topic_prop",
               "engager_hist_num_cols", "bucket_2", "new_features", "engaging_follows_engaged"
            ],
           "metrics": ["PR_AUC", "BCE"],
           "labels": {
               "indicator_retweet": {
                   "loss": {
                       "name": "bce",  # "sce_focal_loss", "weighted_bce"
                       "params": {
                            "ce_weight": 1.0, "rce_weight": 0.3,
                            "gamma": 0.1, "class_weight": {1: 1.0043, 0: 1.0}
                            # "class_weights": {0: 0.5617626532, 1: 4.547753572}
                        }
                    },
                   "loss_weight": None
                },
               "indicator_reply": {
                    "loss": {
                        "name": "bce",  # "sce_focal_loss", "weighted_bce"
                        "params": {
                            "ce_weight": 1.0, "rce_weight": 0.3,
                            "gamma": 0.1, "class_weight": {1: 1.0043, 0: 1.0}
                            # "class_weights": {0: 0.5132905152, 1: 19.31040705}
                        }
                    },
                    "loss_weight": None
                },
               "indicator_retweet_with_comment": {
                   "loss": {
                       "name": "bce",  # "sce_focal_loss", "weighted_bce"
                       "params": {
                            "ce_weight": 1.0, "rce_weight": 0.3,
                            "gamma": 0.1, "class_weight": {1: 1.0043, 0: 1.0}
                            # "class_weights": {0: 0.5037173608,1: 67.75201453}
                        }
                    },
                   "loss_weight": None
                },
               "indicator_like": {
                   "loss": {
                       "name": "bce",  # "sce_focal_looss", "weighted_bce"
                       "params": {
                            "ce_weight": 1.0, "rce_weight": 0.3,
                            "gamma": 0.1, "class_weight": {1: 1.0043, 0: 1.0}
                            # "class_weights": {0: 0.8859675773, 1: 1.1477228}
                        }
                    },
                   "loss_weight": None
                },
            },
           "optimizer": {
               "name": "ws_max_conv",
               "params": {
                   "lr_max": 0.01,
                   "lr_conv": 0.002,
                   "warmup_steps": 200
                }
            }
           }

model_params = {
    "embedding_users_dim": 128,
    "dr_num_bool": 0.4,
    "dr_combined": 0.3,
    "dr_dl": 0.25
}

# Data
home = os.getcwd()
print(home)
data_path = os.path.join(os.getcwd(), "data")
train_path = os.path.join(data_path, train_data)
valid_path = os.path.join(data_path, val_data)

# Models
models_path = os.path.join(os.getcwd(), "models")
current_model_path = os.path.join(models_path, model_folder_name)
if not(os.path.exists(current_model_path)):
    os.mkdir(current_model_path)
else:
    print(f"Path already exists {current_model_path}")
    sys.exit()
checkpoint_path = os.path.join(current_model_path, 'tf_model')
model_h5_path = os.path.join(current_model_path, 'tf_model.h5')

# Logs
logs_path = os.path.join(models_path, "logs")
current_logs_path = os.path.join(logs_path, model_name + '_' + model_version)


# RUN MODEL
counter_path = os.path.join(train_path, 'counter.txt')
if not os.path.exists(counter_path):
    print('You need to count the number of files in the train_path first')
    print("python models/rows_counter.py")
else:
    with open(counter_path, 'r') as f:
        number_lines = int(f.readline())
    print('training with {} lines'.format(number_lines))

with open('metadata.json', 'r') as f:
    metadata = json.load(f)

train_filenames = get_csvs(train_path)
valid_filenames = get_csvs(valid_path)

STEPS_PER_EPOCH = number_lines/BATCH_SIZE//n_vals_per_epoch
real_EPOCHS = EPOCHS * n_vals_per_epoch
patience = patience

labels_to_pred = hparams["labels"].keys()
hparams["model_params"] = model_params
hparams["metadata"] = metadata

with open(os.path.join(current_model_path, "hparams.pickle"), "wb") as output_file:
    pickle.dump(hparams, output_file)

types_list, names2idx = create_type_list(train_filenames[0], labels_to_pred)
col_idx = names_to_idx(train_filenames[0], labels_to_pred)

with open('names2.idx.json', 'w') as f:
    json.dump(names2idx, f, indent=4)

with open('col_idx.json', 'w') as f:
    json.dump(col_idx, f, indent=4)


train_ds = build_dataset(train_filenames, types_list, col_idx, hparams["inputs"],
                         mode="train", epochs=EPOCHS, batch_size=BATCH_SIZE)
val_ds = build_dataset(valid_filenames, types_list, col_idx, hparams["inputs"],
                       mode="val", batch_size=BATCH_SIZE)

model = compile_model(hparams)
model.summary(line_length=120, positions=[.5, .65, .77, 1.])

# Callbacks
cp_callback = tf.keras.callbacks.ModelCheckpoint(
    checkpoint_path+'{epoch:02d}-{val_loss:.2f}.h5',
    save_best_only=True,
    save_weights_only=False,
    monitor='val_loss',  # val_loss
    mode='min',
    verbose=1
)
es_callback = tf.keras.callbacks.EarlyStopping(
    monitor='val_loss',  # val_loss
    patience=patience,
    mode='min',
    restore_best_weights=True
)
nan_callback = tf.keras.callbacks.TerminateOnNaN()
tensorboard_callback = tf.keras.callbacks.TensorBoard(
    log_dir=current_logs_path,
    update_freq=100,
    histogram_freq=1,
    embeddings_freq=1,
    profile_batch=100
)


history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=real_EPOCHS,
    steps_per_epoch=STEPS_PER_EPOCH,
    callbacks=[
        cp_callback,
        es_callback,
        nan_callback,
        tensorboard_callback
    ]
)


model.save(model_h5_path)
