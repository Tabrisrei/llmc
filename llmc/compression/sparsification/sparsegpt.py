import copy
import functools
import math
import os
from abc import ABCMeta, abstractmethod
from collections import defaultdict

import torch
import torch.distributed as dist
import torch.nn as nn
import transformers
from loguru import logger

from llmc.utils.registry_factory import ALGO_REGISTRY

from .base_blockwise_sparsification import BaseBlockwiseSparsification
from llmc.compression.quantization.module_utils import (_LLMC_LINEAR_TYPES_, _TRANSFORMERS_LINEAR_TYPES_,
                           FakeQuantLinear, RotateLinear)


@ALGO_REGISTRY
class SparseGPT(BaseBlockwiseSparsification):
    def __init__(
        self, model, quant_config, input, padding_mask, config, modality='language'
    ):
        super().__init__(model, quant_config, input, padding_mask, config)
        self.dev = torch.device('cuda')
        self.model_dtype = next(self.model.model.parameters()).dtype
        self.add_sparse_config()
        self.layers_cache = {}
        # self.collect_model_qparams()

    @torch.no_grad()
    def add_sparse_config(self):
        self.prefix = self.model.block_name_prefix
        special_config = self.sparsity_config['special']

        self.true_sequential = special_config['true_sequential']
        # self.static_groups = special_config['static_groups']
        # self.actorder = special_config['actorder']
        self.percdamp = special_config['percdamp']
        self.blocksize = special_config['blocksize']

        # self.owq = special_config.get('owq', False)
        self.chunk_num = special_config.get('chunk_num', 1)

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


    # @torch.no_grad()
    # def block_transform(self, block, input_feat, block_kwargs):
    #     # if self.online_rotate:
    #     #     self.replace_rotate_linears(block)
    #     # if self.owq and not hasattr(self, 'n_out_dict'):
    #     #     named_linears = self.model.get_block_linears(block)
    #     #     self.n_out_dict = {}
    #     #     for i, name in enumerate(named_linears.keys()):
    #     #         self.n_out_dict[name] = self.n_outs[i]
    #     super().block_transform(block, input_feat, block_kwargs)

    @torch.no_grad()
    def subset_transform(
        self,
        subset,
        input_feat,
        subset_kwargs,
    ):
        layers_dict = subset['layers']
        for name in layers_dict:
            layer = layers_dict[name]
            if not isinstance(
                layer, tuple(_LLMC_LINEAR_TYPES_ + _TRANSFORMERS_LINEAR_TYPES_)
            ):
                continue
            self.layer_transform(layer, name)
            self.free(name)

    @torch.no_grad()
    def layer_transform(self, layer, name):
        self.initialize_qparams_and_prepare_weights(layer, name)
        W, Hinv = self.process_hessian_and_weights(layer, name)
        self.update_layer_with_transformed_weights(layer, W, Hinv, name)

    def initialize_qparams_and_prepare_weights(self, layer, name):
        # self.qparams = {}
        self.columns = self.layers_cache[name]['columns']
        # self.n_out = self.n_out_dict[name] if self.owq else 0
        # self.n_nonout = self.columns - self.n_out


    def process_hessian_and_weights(self, layer, name):
        W = layer.weight.data.clone()
        if isinstance(layer, nn.Conv2d):
            W = W.flatten(1)
        elif isinstance(layer, transformers.Conv1D):
            W = W.t()

        W = W.float()
        H = self.layers_cache[name]['H']
        del self.layers_cache[name]['H']

        dead = torch.diag(H) == 0
        H[dead, dead] = 1
        W[:, dead] = 0

        damp = self.percdamp * torch.mean(torch.diag(H))
        diag = torch.arange(self.columns, device=self.dev)
        H[diag, diag] += damp
        H = torch.linalg.cholesky(H)
        H = torch.cholesky_inverse(H)
        H = torch.linalg.cholesky(H, upper=True)

        return W, H

    def update_layer_with_transformed_weights(self, layer, W, Hinv, name):
        Losses = torch.zeros(W.shape[0], device=self.dev)
        tmp = torch.zeros_like(W, device=self.dev, dtype=self.model_dtype)

        self.weight_transform(W, Hinv, Losses, tmp)
        torch.cuda.synchronize()
        logger.info(f'error {torch.sum(Losses).item()}')

        if isinstance(layer, transformers.Conv1D):
            tmp = tmp.t()

        layer.weight.data = tmp.reshape(layer.weight.shape).to(self.model_dtype)

    @torch.no_grad()
    def weight_transform(self, W, Hinv, Losses, tmp):
        for i1 in range(0, self.columns, self.blocksize):
            i2 = min(i1 + self.blocksize, self.columns)
            count = i2 - i1

            W1, Hinv1 = W[:, i1:i2].clone(), Hinv[i1:i2, i1:i2]
            tmp1, Err1, Losses1 = (
                torch.zeros_like(W1),
                torch.zeros_like(W1),
                torch.zeros_like(W1),
            )

            tmp_block = W1 ** 2 / (torch.diag(Hinv1).reshape((1, -1))) ** 2

            if self.pattern == 'structured' and self.granularity == 'channel_wise':
                tmp_block_channel = torch.mean(tmp_block, dim=1, keepdim=True)
                sort_res = torch.sort(tmp_block_channel, dim=0, stable=True)
                indices = sort_res.indices[: int(tmp_block_channel.shape[0] * self.sparsity), :]
                mask1 = torch.zeros_like(tmp_block, dtype=torch.bool)
                mask1[indices.squeeze(1), :] = True
                pass
            elif self.pattern == 'semi_structured':
                # reshape into [rows, num_chunks, prunem]
                # mask1 = torch.zeros_like(W1) == 1
                # tmp_block = W1 ** 2 / (torch.diag(Hinv1).reshape((1, -1))) ** 2
                sep_len = tmp_block.shape[1]
                if sep_len % self.prunem != 0:
                    pad_len = self.prunem - (sep_len % self.prunem)
                    tmp_block = torch.nn.functional.pad(tmp_block, (0, pad_len), value=0)

                chunked_tmp_block = tmp_block.reshape(tmp_block.shape[0], -1, self.prunem)
                chunked_sort_res = torch.sort(chunked_tmp_block, dim=-1, stable=True)
                chunked_indices = chunked_sort_res.indices[:, :, :self.prunen]
                mask1 = torch.zeros_like(chunked_tmp_block, dtype=torch.bool)
                mask1.scatter_(2, chunked_indices, True)
                mask1 = mask1.reshape(tmp_block.shape)[:, :sep_len]

            elif self.pattern == 'unstructured':
                # tmp_block = W1 ** 2 / (torch.diag(Hinv1).reshape((1, -1))) ** 2
                thresh = torch.sort(tmp_block.flatten())[0][int(tmp_block.numel() * self.sparsity)]
                mask1 = tmp_block <= thresh
            else:
                mask1 = torch.zeros_like(W1, dtype=torch.bool)
                logger.info(f'Can not recognize the sparsity type, using all weights.')

            for i in range(count):
                w, d = W1[:, i], Hinv1[i, i]

                # if self.pattern == 'semi_structured' and self.prunen != 0 and i % self.prunem == 0:
                #     tmp_block = W1[:, i:(i + self.prunem)] ** 2 / (torch.diag(Hinv1)[i:(i + self.prunem)].reshape((1, -1))) ** 2
                #     mask1.scatter_(1, i + torch.topk(tmp_block, self.prunen, dim=1, largest=False)[1], True)

                q = w.clone()
                q[mask1[:, i]] = 0

                tmp1[:, i] = q
                # Losses1[:, i] = ((w - q) ** 2) / (2 * d**2)
                Losses1[:, i] = ((w - q) ** 2) / (d**2)
                err1 = (w - q) / d
                W1[:, i:] -= err1.unsqueeze(1).matmul(Hinv1[i, i:].unsqueeze(0))
                Err1[:, i] = err1

            # tmp[:, i1:i2], Losses[:, i1:i2] = tmp1, Losses1
            tmp[:, i1:i2] = tmp1
            # Losses[:, i1:i2] += torch.sum(Losses1, 1) / 2
            Losses += torch.sum(Losses1, 1) / 2
            W[:, i2:] -= Err1.matmul(Hinv[i1:i2, i2:])

    @torch.no_grad()
    def cache_input_hook(self, m, inp, out, name, feat_dict):
        if isinstance(m, tuple(_LLMC_LINEAR_TYPES_ + _TRANSFORMERS_LINEAR_TYPES_)):
            self.add_batch(self.named_layers[name], name, inp[0].data, out.data)
        # if self.act_static:
        #     super().cache_input_hook(m, inp, out, name, feat_dict)

    @torch.no_grad()
    def add_batch(self, layer, name, inp, out):
        world_size = int(os.environ['WORLD_SIZE'])
        if len(inp.shape) == 2:
            inp = inp.unsqueeze(0)
        tmp = inp.shape[0]
        if isinstance(
            layer, (nn.Linear, transformers.Conv1D)
        ):
            if len(inp.shape) == 3:
                inp = inp.reshape((-1, inp.shape[-1]))
            inp = inp.t()
        # if isinstance(layer, nn.Conv2d):
        #     unfold = nn.Unfold(
        #         layer.kernel_size,
        #         dilation=layer.dilation,
        #         padding=layer.padding,
        #         stride=layer.stride,
        #     )
        #     inp = unfold(inp)
        #     inp = inp.permute([1, 0, 2])
        #     inp = inp.flatten(1)

        assert inp.shape[1] % self.chunk_num == 0, \
            f'Error: inp.shape[1] ({inp.shape[1]}) cannot be evenly divided by chunk_num.'
        chunks = torch.chunk(inp, self.chunk_num, dim=1)

        self.layers_cache[name]['H'] *= self.layers_cache[name]['nsamples'] / (
            self.layers_cache[name]['nsamples'] + tmp
        )
        self.layers_cache[name]['nsamples'] += tmp

        for chunk in chunks:
            chunk = math.sqrt(2 / self.layers_cache[name]['nsamples']) * chunk.float()
            self.layers_cache[name]['H'] += chunk.matmul(chunk.t())

        dist.all_reduce(self.layers_cache[name]['H'], op=dist.ReduceOp.SUM)
        dist.all_reduce(torch.tensor(self.layers_cache[name]['nsamples']).cuda(),
                        op=dist.ReduceOp.SUM)
        self.layers_cache[name]['H'] /= world_size

    @torch.no_grad()
    def layer_init(self, layer, name):
        W = layer.weight.data.clone()
        if isinstance(layer, nn.Conv2d):
            W = W.flatten(1)
        if isinstance(layer, transformers.Conv1D):
            W = W.t()
        self.layers_cache[name]['H'] = torch.zeros(
            (W.shape[1], W.shape[1]), device=self.dev
        )
        self.layers_cache[name]['nsamples'] = 0
        self.layers_cache[name]['columns'] = W.shape[1]

    @torch.no_grad()
    def subset_init(self, subset):
        self.named_layers = subset['layers']
        for name in self.named_layers:
            self.layers_cache[name] = {}
            self.layer_init(self.named_layers[name], name)

    @torch.no_grad()
    def block_init(self, block):
        self.named_layers = self.model.get_block_linears(block)
        for name in self.named_layers:
            self.layers_cache[name] = {}
            self.layer_init(self.named_layers[name], name)

    @torch.no_grad()
    def free(self, name):
        self.H = None
        self.Losses = None
        self.Trace = None
        del self.layers_cache[name]
        torch.cuda.empty_cache()

    # @torch.no_grad()
    # def ready(self):
    #     if 'scale' not in self.qparams:
    #         return False
    #     return torch.all(self.qparams['scale'] != 0)
