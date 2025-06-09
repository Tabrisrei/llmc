
class TokenReductionModule:
    def __init__(self, config, model, blocks):
        self.config = config
        self.model = model
        self.blocks = blocks
        self.set_sparse_config()

    def set_sparse_config(self):
        self.special_config = self.config.get('special', {})
        self.special_config['is_video_model'] = self.model.pruning_config['is_video_model']
        # vision_token can be image or video
        if self.special_config['is_video_model']:
            self.special_config['vision_token_index'] = self.model.pruning_config[
                'video_token_index'
            ]
            self.special_config['vision_token_length'] = self.model.pruning_config[
                'video_token_length'
            ]
        else:
            self.special_config['vision_token_index'] = self.model.pruning_config[
                'image_token_index'
            ]
            self.special_config['vision_token_length'] = self.model.pruning_config[
                'image_token_length'
            ]

    def register_reduction_modules(self):
        pass
