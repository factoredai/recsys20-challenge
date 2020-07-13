import os
import numpy as np
import pandas as pd
import torch
from transformers import BertModel
from tqdm import tqdm


def pad(x, to=150, sep='\t'):
    """Pads/truncates an array to a constant length.

    Arguments:
        x {str} -- string with the array values

    Keyword Arguments:
        to {int} -- length to pad the array to (default: {150})
        sep {str} -- separator character to transform string to array (default: {'\t'})

    Returns:
        list -- array with fixed length
    """
    x = x.split(sep)
    return list(np.pad(x, (0, max(0, to - len(x))), mode='constant')[:to].astype(np.int64))


def array_to_str(arr):
    """Transforms an array to a string for easier serialization.

    Arguments:
        arr {numpy array} -- array to transform to string

    Returns:
        str -- string with the comma delimited values of the input array
    """
    return ','.join(arr.astype(np.str))


def get_bert_embeddings(path, save_at='./embeddings', n_samples=None, batch_size=1024,
                        token_sep='\t', max_len=150, colnames=['id', 'tokens'],
                        bert_to_use='bert-base-multilingual-cased'):
    """Obtains the BERT embeddings of a dataset comprised of id and tokens.

    Arguments:
        path {str} -- path of the dataset

    Keyword Arguments:
        save_at {str} -- path to save results at (default: {'./embeddings'})
        n_samples {int} -- total number of samples, used for tqdm. Optional (default: {None})
        batch_size {int} -- batch size to use per worker (default: {1024})
        token_sep {str} -- separator char for the token values (default: {'\t'})
        max_len {int} -- maximum length to truncate/pad the embeddings to (default: {150})
        colnames {list} -- column names of the dataset (default: {['id', 'tokens']})
        bert_to_use {str} -- Huggingface Transformer model to use
                             (default: {'bert-base-multilingual-cased'})
    """

    # verify GPU availability
    assert torch.cuda.is_available(), 'CUDA is unavailable. Processing is prohibitively expensive without GPU.'
    assert len(colnames) == 2, 'File should only have two columns: one for the id and one for the tokens.'
    device = torch.device("cuda:0")
    n_available_gpus = torch.cuda.device_count()
    print('Found', n_available_gpus, ' available GPUs')

    # setup calculated parameters
    chunksize = batch_size * n_available_gpus
    if n_samples:
        steps = int(np.ceil(n_samples/chunksize))
        print(f'Using data at {path} - total records: {n_samples}')
    else:
        steps = None
    if not os.path.exists(save_at):
        os.mkdir(save_at)

    # optionally distribute model
    model = BertModel.from_pretrained('bert-base-multilingual-cased')
    if n_available_gpus > 1:
        model = torch.nn.DataParallel(model)
    model = model.to(device)

    # start process
    df = pd.read_csv(path, names=colnames, chunksize=chunksize)
    failed_batches = 0

    pad_fn = lambda t: pad(t, max_len, token_sep)
    if steps:
        for i in tqdm(range(steps)):
            try:
                batch_df = next(df)
                padded_seqs = np.array([row for row in batch_df[colnames[1]].map(pad_fn)])
                tokens = torch.tensor(padded_seqs).to(device)
                with torch.no_grad():
                    z = model(tokens)[1]
                    z = z.cpu().numpy().astype(np.float16)
                z = np.array([array_to_str(row) for row in z])
                z = np.expand_dims(z, axis=1)
                result = np.concatenate((np.expand_dims(np.array(batch_df[colnames[0]]), 1), z), axis=1)
                np.savetxt(f'{save_at}/part_{i}.txt', result, fmt="%s")
            except:
                failed_batches += 1
    else:
        for batch_df in tqdm(df):
            try:
                padded_seqs = np.array([row for row in batch_df[colnames[1]].map(pad_fn)])
                tokens = torch.tensor(padded_seqs).to(device)
                with torch.no_grad():
                    z = model(tokens)[1]
                    z = z.cpu().numpy().astype(np.float16)
                z = np.array([array_to_str(row) for row in z])
                z = np.expand_dims(z, axis=1)
                result = np.concatenate((np.expand_dims(np.array(batch_df[colnames[0]]), 1), z), axis=1)
                np.savetxt(f'{save_at}/part_{i}.txt', result, fmt="%s")
            except:
                failed_batches += 1

    print(f'Total failed batches: {failed_batches}')


if __name__ == '__main__':
    files = [
        's3://bucket-name/data/textEncodings/test-tweets.csv',
        's3://bucket-name/data/textEncodings/submission-tweets.csv',
        's3://bucket-name/data/textEncodings/train-tweets.csv'
    ]
    saving_dirs = ['./test-embeddings', './submission-embeddings', './train-embeddings']

    for i in range(len(files)):
        get_bert_embeddings(files[i], save_at=saving_dirs[i], colnames=['tweet_id', 'text_tokens'])
