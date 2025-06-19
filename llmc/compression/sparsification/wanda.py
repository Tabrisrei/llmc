import torch
import torch.nn as nn
from loguru import logger

from llmc.utils.registry_factory import ALGO_REGISTRY

from .base_blockwise_sparsification import BaseBlockwiseSparsification


@ALGO_REGISTRY
class Wanda(BaseBlockwiseSparsification):
    def __init__(self, model, sparsity_config, input, padding_mask, config):
        super().__init__(model, sparsity_config, input, padding_mask, config)
        self.add_sparse_config()

    @torch.no_grad()
    def add_sparse_config(self):
        self.prefix = self.model.block_name_prefix
        special_config = self.sparsity_config['special']

        self.pattern = special_config.get('pattern', False)
        self.granularity = special_config.get('granularity', False)
        self.prunen = special_config.get('prunen', 0)
        self.prunem = special_config.get('prunem', 0)

        assert self.pattern in ['unstructured', 'structured', 'semi_structured'], \
            f'pattern {self.pattern} should be either unstructured, structured or semi_structured'

        if self.pattern == 'structured':
            assert self.granularity in ['channel_wise', 'block_wise'], \
                f'granularity {self.granularity} should be either channel or row'

        if self.pattern == 'semi_structured':
            assert self.prunen > 0 and self.prunem > 0 and self.prunen < self.prunem, \
                f'prunen {self.prunen} and prunem {self.prunem} should be greater than 0 and prunen < prunem'
            
    @torch.no_grad()
    def get_row_scale(self, layer, act):
        if len(act.shape) == 2:
            act = act.unsqueeze(0)
        nsamples = act.shape[0]
        if isinstance(layer, nn.Linear):
            if len(act.shape) == 3:
                act = act.reshape((-1, act.shape[-1]))
            act = act.t()

        columns = layer.weight.data.shape[1]

        scaler_row = torch.zeros((columns), device=layer.weight.device)

        act = act.type(torch.float32).to(scaler_row.device)
        scaler_row += torch.norm(act, p=2, dim=1) ** 2 / nsamples
        return scaler_row

    @torch.no_grad()
    def subset_transform(
        self,
        subset,
        input_feat,
        *subset_kwargs,
    ):

        layers_dict = subset['layers']
        # layers_dict = subset
        # input_name = subset['input'][0]

        # layers = list(layers_dict.values())

        for layer_name, layer in layers_dict.items():
            scaler_row = self.get_row_scale(layer, input_feat[layer_name][0])
            W_metric = torch.abs(layer.weight.data) * torch.sqrt(
                scaler_row.reshape((1, -1))
            )

            # W_mask = (
            #     torch.zeros_like(W_metric) == 1
            # )  # initialize a mask to be all False

            if self.pattern == 'structured' and self.granularity == 'channel_wise':
                W_mask = torch.zeros_like(W_metric, dtype=torch.bool)
                W_metric_channel = torch.mean(W_metric, dim=1, keepdim=True)
                sort_res = torch.sort(W_metric_channel, dim=0, stable=True)
                indices = sort_res.indices[ : int(W_metric_channel.shape[0] * self.sparsity), : ]
                W_mask[indices.squeeze(-1), :] = True
            elif self.pattern == 'semi_structured':
                # structured n:m sparsity
                sep_len = W_metric.shape[1]
                if sep_len % self.prunem != 0:
                    pad_len = self.prunem - (sep_len % self.prunem)
                    W_metric = torch.nn.functional.pad(W_metric, (0, pad_len), value=0)

                chunked_W_metric = W_metric.reshape(
                    W_metric.shape[0], -1, self.prunem
                )  # reshape to (batch_size, num_chunks, chunk_size)

                chunked_sort_res = torch.sort(chunked_W_metric, dim=-1, stable=True)
                # chunked_indices = chunked_sort_res[1][ : , : , : int(chunked_W_metric.shape[-1] * self.sparsity)]
                chunked_indices = chunked_sort_res.indices[ : , : , : self.prunen]
                chunked_W_mask = torch.zeros_like(chunked_W_metric, dtype=torch.bool)
                chunked_W_mask.scatter_(2, chunked_indices, True)
                
                W_mask = chunked_W_mask.reshape(W_metric.shape)
                W_mask = W_mask.reshape(W_metric.shape)[:, :sep_len]

                # for ii in range(0, W_metric.shape[1], self.prunem):
                #     # if ii % self.prunem == 0:
                #     tmp = W_metric[:,ii:(ii+self.prunem)].float()
                #     W_mask.scatter_(1,ii+torch.topk(tmp, self.prunen, dim=1, largest=False)[1], True)
            elif self.pattern == 'unstructured':
                W_mask = torch.zeros_like(W_metric, dtype=torch.bool)
                sort_res = torch.sort(W_metric, dim=-1, stable=True)
                # indices = sort_res[1][:, : int(W_metric.shape[1] * self.sparsity)]
                indices = sort_res.indices[:, : int(W_metric.shape[1] * self.sparsity)]

                W_mask.scatter_(1, indices, True)

            layer.weight.data[W_mask] = 0  # set weights to zero
