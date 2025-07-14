import functools
from functools import wraps
from types import MethodType

import torch

from llmc.utils.registry_factory import TOKEN_REDUCTION_REGISTRY

from .token_reduction_module import TokenReductionModule
from .utils import prefill_wrapper


@TOKEN_REDUCTION_REGISTRY.register('FastV')
class FastV(TokenReductionModule):
    def __init__(self, config, model, blocks):
        super().__init__(config, model, blocks)
        self.add_sparse_config()
        self.register_reduction_modules()

    def add_sparse_config(self):

        self.pruning_loc = self.special_config['pruning_loc']
        self.special_config['image_token_length'] = \
            self.model.pruning_config['image_token_length']
        self.special_config['IMAGE_TOKEN_INDEX'] = \
            self.model.pruning_config['IMAGE_TOKEN_INDEX']
        self.special_config['attn_scores'] = None

        self.pruning_paras = self.special_config

    def register_reduction_modules(self):

        @prefill_wrapper
        def input_hook(module, input_args, pruning_paras):
            input_ids = input_args[0]
            image_token_idxs = (input_ids[0] ==
                                pruning_paras['vision_token_index']).nonzero(as_tuple=True)[0]
            pruning_paras['image_token_start_index'] = image_token_idxs[0].item()

            return input_args

        def input_hook_llava(fn, pruning_paras):
            @wraps(fn)
            def wrapper(self, *args, **kwargs):
                if len(args) == 0:
                    return fn(*args, **kwargs)
                input_args = args[0]
                if hasattr(input_args[0], 'shape') and input_args[0].shape[0] == 1:
                    return fn(*args, **kwargs)

                input_ids = args[0]
                attention_mask = args[2]
                token_indices = \
                    input_ids[0][attention_mask[0]] == pruning_paras['IMAGE_TOKEN_INDEX']
                pruning_paras['image_token_start_index'] = torch.where(token_indices)[0][0].item()

                outputs = fn(*args, **kwargs)
                return outputs
            return wrapper

        def update_output_attentions_hook(module, args, kwargs, pruning_paras):
            kwargs['output_attentions'] = True
            pruning_paras['attn_scores'] = module.__class__.forward(module, *args, **kwargs)[1]
            kwargs['output_attentions'] = False
            return args, kwargs

        @prefill_wrapper
        def fastv_pruning_hook(module, args, kwargs, pruning_paras):

            rate = pruning_paras['rate']
            image_token_start_index = pruning_paras['image_token_start_index']
            image_token_length = pruning_paras['image_token_length']

            hidden_states = args[0]
            causal_mask = kwargs['attention_mask']

            device = hidden_states.device
            # last_layer_attention = layer_outputs[1]
            last_layer_attention = pruning_paras['attn_scores']
            # compute average attention over different head
            last_layer_attention_avg = torch.mean(last_layer_attention, dim=1)[0]
            # generate new attention mask based on the average attention,
            # sample the top ATTENTION_RANK tokens with highest attention
            last_layer_attention_avg_last_tok = last_layer_attention_avg[-1]
            # get the attention in image token
            last_layer_attention_avg_last_tok_image = \
                last_layer_attention_avg_last_tok[image_token_start_index:
                                                  image_token_start_index + image_token_length]
            # get the indexes of the top ATTENTION_RANK tokens
            top_attention_rank_index = \
                last_layer_attention_avg_last_tok_image.topk(
                    round(image_token_length * (1 - rate))).indices + image_token_start_index
            # keep index
            keep_indexs = torch.cat(
                (
                    torch.arange(image_token_start_index, device=device),
                    top_attention_rank_index,
                    torch.arange(image_token_start_index + image_token_length,
                                 hidden_states.shape[1], device=device)
                )
            )

            # sort index
            keep_indexs = keep_indexs.sort().values
            # filter hidden states &
            hidden_states = hidden_states[:, keep_indexs, :]
            # update position ids
            position_ids = keep_indexs.unsqueeze(0)
            # update attention mask
            if causal_mask is not None:
                causal_mask = causal_mask[:, :, :hidden_states.shape[1], :hidden_states.shape[1]]
                kwargs['attention_mask'].resize_as_(causal_mask).copy_(causal_mask.clone())
            kwargs['cache_position'].resize_as_(position_ids.squeeze(0)).copy_(
                position_ids.squeeze(0).clone())
            kwargs['position_ids'].resize_as_(position_ids).copy_(position_ids.clone())

            position_embeddings = kwargs['position_embeddings']
            new_pe0 = position_embeddings[0][:, keep_indexs, :].clone()
            new_pe1 = position_embeddings[1][:, keep_indexs, :].clone()
            position_embeddings[0].resize_as_(new_pe0).copy_(new_pe0)
            position_embeddings[1].resize_as_(new_pe0).copy_(new_pe1)

            return (hidden_states,), kwargs

        if self.model.__class__.__name__ == 'LlavaHf':
            self.model.embed_tokens.register_forward_pre_hook(
                functools.partial(input_hook, pruning_paras=self.pruning_paras)
            )
        elif self.model.__class__.__name__ == 'Llava':
            hook_fn = input_hook_llava(
                self.model.vlm_model.prepare_inputs_labels_for_multimodal,
                self.pruning_paras
            )
            self.model.vlm_model.prepare_inputs_labels_for_multimodal = MethodType(
                hook_fn, self.model.vlm_model
            )

        self.blocks[self.pruning_loc - 1].register_forward_pre_hook(
            functools.partial(update_output_attentions_hook, pruning_paras=self.pruning_paras),
            with_kwargs=True
        )

        self.blocks[self.pruning_loc].register_forward_pre_hook(
            functools.partial(fastv_pruning_hook, pruning_paras=self.pruning_paras),
            with_kwargs=True
        )
