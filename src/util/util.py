import torch
from collections import OrderedDict


class HeavyForward(object):

    @classmethod
    def concat_split_net_outs(cls, split_net_outs):
        """
        Concat the pieces of network's output into a large network output dictionary
        """
        
        net_out = OrderedDict()

        net_out_keys = split_net_outs[0].keys()

        for key in net_out_keys:

            if isinstance(split_net_outs[0][key], torch.Tensor):
                
                net_out[key] = torch.unsqueeze(torch.cat([split_out[key]
                                                          for split_out in split_net_outs],
                                                         dim=0),
                                               dim=0).detach().cpu()

            elif isinstance(split_net_outs[0][key], list):

                extended_list = []
                for split_out in split_net_outs:
                    extended_list.extend(split_out[key])

                net_out[key] = [extended_list]

            else:

                appended_list = []
                for split_out in split_net_outs:
                    appended_list.append(split_out[key])

                net_out[key] = [appended_list]

        return net_out

    @classmethod
    def batch_to_split_batches(cls, batch, split_batch_size=1):
        """
        Split batch into pieces depending on the "split_info" in batch
        """

        split_info = batch.get("split_info", None)
        if split_info is None:

            return batch

        else:

            target_key = split_info["target_key"]
            if isinstance(target_key, list):
                target_key = target_key[0]

            n_splits = batch[target_key].shape[1]

            n_split_batches = n_splits // split_batch_size \
                if (n_splits % split_batch_size) == 0 else (n_splits // split_batch_size) + 1

            split_batches = []

            for i_split in range(n_split_batches):

                single_split = OrderedDict()

                for key, value in batch.items():

                    if key == target_key:
                        single_split[key] = value[0, split_batch_size * i_split:split_batch_size * (i_split + 1)]
                    else:
                        single_split[key] = value

                split_batches.append(single_split)

            return split_batches, target_key
