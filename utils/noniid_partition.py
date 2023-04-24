import logging

import numpy as np


def non_iid_partition_with_dirichlet_distribution(label_list,
                                                  client_num,
                                                  classes,
                                                  alpha,
                                                  task='classification'):
    net_dataidx_map = {}
    K = classes

    # For multiclass labels, the list is ragged and not a numpy array
    N = len(label_list) if task == 'segmentation' else label_list.shape[0]

    # guarantee the minimum number of sample in each client
    min_size = 0
    while min_size < 10:
        idx_batch = [[] for _ in range(client_num)]

        if task == 'segmentation':
            # Unlike classification tasks, here, one instance may have multiple categories/classes
            for c, cat in enumerate(classes):
                if c > 0:
                    idx_k = np.asarray([np.any(label_list[i] == cat) and not np.any(
                        np.in1d(label_list[i], classes[:c])) for i in
                                        range(len(label_list))])
                else:
                    idx_k = np.asarray(
                        [np.any(label_list[i] == cat) for i in range(len(label_list))])

                # Get the indices of images that have category = c
                idx_k = np.where(idx_k)[0]
                idx_batch, min_size = partition_class_samples_with_dirichlet_distribution(N, alpha, client_num,
                                                                                          idx_batch, idx_k)
        else:
            # for each classification in the dataset
            for k in range(K):
                # get a list of batch indexes which are belong to label k
                idx_k = np.where(label_list == k)[0]
                idx_batch, min_size = partition_class_samples_with_dirichlet_distribution(N, alpha, client_num,
                                                                                          idx_batch, idx_k)
    for i in range(client_num):
        np.random.shuffle(idx_batch[i])
        net_dataidx_map[i] = idx_batch[i]

    return net_dataidx_map