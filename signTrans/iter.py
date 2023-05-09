# -*- coding: utf-8 -*-
"""
Transfer Learning Project

author: Hao Zhou

"""


from torchtext.legacy import data

from signTrans.utils import load_cfg

cfg = load_cfg()


def make_iter( 
        dataset: data.Dataset,
        batch_size: int = 128,
        # batch_type: str = "sentence",
        train: bool = True,
        shuffle: bool = True,
        ):
    # batch_size_fn = token_batch_size_fn if batch_type == "token" else None
    batch_size_fn = None

    # print(batch_size)

    if train:
        shuffle = True
    else:
        shuffle = False

    if train:

        if cfg['model_type'] == 'multiseq':
            
            # optionally shuffle and sort during training
            data_iter = data.BucketIterator(
                repeat=False,
                sort=False,
                dataset=dataset,
                batch_size=batch_size,
                batch_size_fn=batch_size_fn,
                train=True,
                sort_within_batch=True,
                sort_key=lambda x: len(x.src_long),
                shuffle=shuffle,
            )
        else:
            # optionally shuffle and sort during training
            data_iter = data.BucketIterator(
                repeat=False,
                sort=False,
                dataset=dataset,
                batch_size=batch_size,
                batch_size_fn=batch_size_fn,
                train=True,
                sort_within_batch=True,
                sort_key=lambda x: len(x.src),
                shuffle=shuffle,
            )
            
    else:
        # don't sort/shuffle for validation/inference
        data_iter = data.BucketIterator(
            repeat=False,
            dataset=dataset,
            batch_size=batch_size,
            batch_size_fn=batch_size_fn,
            train=False,
            sort=False,
        )

    return data_iter