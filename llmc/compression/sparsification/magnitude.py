import torch
from loguru import logger

from llmc.utils.registry_factory import ALGO_REGISTRY

from .base_blockwise_sparsification import BaseBlockwiseSparsification


@ALGO_REGISTRY
class Magnitude(BaseBlockwiseSparsification):
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
    def subset_transform(
        self,
        subset,
        input_feat,
        subset_kwargs,
    ):
        layers_dict = subset['layers']

        layers = list(layers_dict.values())
        for layer in layers:
            W = layer.weight.data
            W_metric = torch.abs(W)
            if not self.semi_structured:
                thresh = torch.sort(W_metric.flatten().cuda())[0][
                    int(W.numel() * self.sparsity)
                ].cpu()
                W_mask = W_metric <= thresh
            else:
                W_metric_shape_origin = W_metric.shape
                chunked_W_metric = W_metric.reshape(W_metric_shape_origin[0], -1, W_metric.shape[-1])
                chunked_sort_res = torch.sort(chunked_W_metric, dim=-1, stable=True)
                chunked_indices = chunked_sort_res.indices[ : , : , : self.prunen]
                chunked_W_mask = torch.zeros_like(chunked_W_metric, dtype=torch.bool)
                chunked_W_mask.scatter_(1, chunked_indices, True)
                
                W_mask = chunked_W_mask.reshape(W_metric_shape_origin)

        W[W_mask] = 0
