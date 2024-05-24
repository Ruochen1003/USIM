import numpy as np
import torch
import pandas as pd
import torch.nn.utils.rnn as rnn_utils

class interaction():
    def __init__(self,interaction = None):
        self.interaction = dict()
        if isinstance(interaction, dict):
            self.interaction['user'] = [torch.tensor(user) for user in list(interaction['user'])]
            #self.interaction['user'] = rnn_utils.pad_sequence(self.interaction['user'], batch_first=True, padding_value=-1)
            #self.interaction['mask'] = ~torch.lt(self.interaction['user'], 0)
            self.interaction['item'] = torch.tensor(list(interaction['item']))
            if 'item_content' in interaction.keys():
                self.interaction['item_content'] = torch.tensor(np.array(interaction['item_content']))
            else:
                self.interaction['item_content'] = None
        else:
            self.interaction['user'] = None
            self.interaction['item'] = None
            self.interaction['content'] = None
            self.interaction['mask'] = None

    def __getitem__(self, key):
        if isinstance(key, slice):
            sliced_dict = interaction()
            for k in self.interaction.keys():
                if self.interaction[k] is not None:
                    sliced_dict.interaction[k] = self.interaction[k][key]
                else:
                    sliced_dict.interaction[k] = None
            return sliced_dict
        else:
            return self.interaction[key]

    def _convert_to_tensor(self,data):
        """This function can convert common data types (list, pandas.Series, numpy.ndarray, torch.Tensor) into torch.Tensor.

        Args:
            data (list, pandas.Series, numpy.ndarray, torch.Tensor): Origin data.

        Returns:
            torch.Tensor: Converted tensor from `data`.
        """
        elem = data[0]
        if isinstance(elem, (float, int, np.float, np.int64)):
            new_data = torch.as_tensor(data)
        elif isinstance(elem, (list, tuple, pd.Series, np.ndarray, torch.Tensor)):
            seq_data = [torch.as_tensor(d) for d in data]
            new_data = rnn_utils.pad_sequence(seq_data, batch_first=True)
        else:
            raise ValueError(f'[{type(elem)}] is not supported!')
        if new_data.dtype == torch.float64:
            new_data = new_data.float()
        return new_data

    def to(self,device):
        for key,value in self.interaction.items():
            if self.interaction[key] is not None:
                if isinstance(self.interaction[key],list):
                    self.interaction[key] = [user.to(device) for user in self.interaction[key]]
                else:
                    self.interaction[key] = self.interaction[key].to(device)

    def __len__(self):
        return len(self.interaction['user'])


