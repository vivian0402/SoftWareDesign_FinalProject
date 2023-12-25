import os

from . import constants
from .modeling_CTBert import CTBertClassifier, CTBertRegression
from .modeling_CTBert import CTBertForCL, TableGPTForMask
from .trainer_utils import CTBertCollatorForCL

class BaseCTBertModel:
    def __init__(self, **kwargs):
        self.categorical_columns = kwargs.get('categorical_columns')
        self.numerical_columns = kwargs.get('numerical_columns')
        self.binary_columns = kwargs.get('binary_columns')
        self.hidden_dim = kwargs.get('hidden_dim', 128)
        self.num_layer = kwargs.get('num_layer', 2)
        self.num_attention_head = kwargs.get('num_attention_head', 8)
        self.hidden_dropout_prob = kwargs.get('hidden_dropout_prob', 0)
        self.ffn_dim = kwargs.get('ffn_dim', 256)
        self.activation = kwargs.get('activation', 'relu')
        self.vocab_freeze = kwargs.get('vocab_freeze', False)
        self.device = kwargs.get('device', 'cuda')
        self.task_type = kwargs.get('task_type')
        self.model_type = kwargs.get('model_type')
        self.checkpoint = kwargs.get('checkpoint')
        self.mlm_probability = kwargs.get('mlm_probability', 0.15)
        self.projection_dim = kwargs.get('projection_dim', 128)
        self.supervised = kwargs.get('supervised', False)
        self.num_partition = kwargs.get('num_partition')
        self.overlap_ratio = kwargs.get('overlap_ratio')
        self.ignore_duplicate_cols = kwargs.get('ignore_duplicate_cols', True)

    def load_model(self, model):
        if self.checkpoint is not None:
            model.load(self.checkpoint)
    
    def create_model(self):
        if self.task_type == 'pretrain_CL_ds':
            model = BuildContrastiveLearner(**self.__dict__)
        elif self.task_type == 'pretrain_mask_ds' or self.task_type == 'pretrain_mask':
            model = BuildMaskFeaturesLearner(**self.__dict__)
        elif self.task_type == 'fintune' or self.task_type == 'scratch':
            if self.model_type == 'regression':
                model = BuildRegression(**self.__dict__)
            elif self.model_type == 'classify':
                model = BuildClassifier(**self.__dict__)
            else:
                raise ValueError(f"Unknown model type: {self.model_type}")
        else:
            raise ValueError(f"Unknown task: {self.task_type}")
        return model.build()

class BuildClassifier(BaseCTBertModel):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.feature_extractor = kwargs.get('feature_extractor')
        self.num_class = kwargs.get('num_class', 2)
        self.use_bert = kwargs.get('use_bert', True)

    def build(self) -> CTBertClassifier:
        model = CTBertClassifier(
            categorical_columns = self.categorical_columns,
            numerical_columns = self.numerical_columns,
            binary_columns = self.binary_columns,
            hidden_dim = self.hidden_dim,
            num_layer = self.num_layer,
            num_attention_head = self.num_attention_head,
            hidden_dropout_prob = self.hidden_dropout_prob,
            ffn_dim = self.ffn_dim,
            activation = self.activation,
            vocab_freeze = self.vocab_freeze,
            device = self.device,
            feature_extractor = self.feature_extractor,
            num_class=self.num_class,
            use_bert=self.use_bert
        )
        
        self.load_model(model)

        return model

class BuildRegression(BaseCTBertModel):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.feature_extractor = kwargs.get('feature_extractor')

    def build(self) ->  CTBertRegression:
        model =  CTBertRegression(
            categorical_columns = self.categorical_columns,
            numerical_columns = self.numerical_columns,
            binary_columns = self.binary_columns,
            hidden_dim = self.hidden_dim,
            num_layer = self.num_layer,
            num_attention_head = self.num_attention_head,
            hidden_dropout_prob = self.hidden_dropout_prob,
            ffn_dim = self.ffn_dim,
            activation = self.activation,
            vocab_freeze = self.vocab_freeze,
            device = self.device,
            feature_extractor = self.feature_extractor,
            num_class=self.num_class,
            use_bert=self.use_bert,
        )
        
        self.load_model(model)

        return model

class BuildContrastiveLearner(BaseCTBertModel):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.projection_dim = kwargs.get('projection_dim')
        self.num_partition = kwargs.get('num_partition')
        self.overlap_ratio = kwargs.get('overlap_ratio')
        self.supervised = kwargs.get('supervised')
        self.ignore_duplicate_cols = kwargs.get('ignore_duplicate_cols')

    def build(self): 
        model = CTBertForCL(
            categorical_columns = self.categorical_columns,
            numerical_columns = self.numerical_columns,
            binary_columns = self.binary_columns,
            hidden_dim = self.hidden_dim,
            num_layer = self.num_layer,
            num_attention_head = self.num_attention_head,
            hidden_dropout_prob = self.hidden_dropout_prob,
            ffn_dim = self.ffn_dim,
            activation = self.activation,
            vocab_freeze = self.vocab_freeze,
            device = self.device,
            num_partition= self.num_partition,
            supervised=self.supervised,
            projection_dim=self.projection_dim,
            overlap_ratio=self.overlap_ratio,
        )

        # build collate function for contrastive learning
        collate_fn = CTBertCollatorForCL(
            categorical_columns=self.categorical_columns,
            numerical_columns=self.numerical_columns,
            binary_columns=self.binary_columns,
            overlap_ratio=self.overlap_ratio,
            num_partition=self.num_partition,
            ignore_duplicate_cols=self.ignore_duplicate_cols
        )

        if self.checkpoint is not None:
            collate_fn.feature_extractor.load(os.path.join(self.checkpoint, constants.EXTRACTOR_STATE_DIR))

        return model, collate_fn

class BuildMaskFeaturesLearner(BaseCTBertModel):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def build(self): 
        model = TableGPTForMask(
            categorical_columns = self.categorical_columns,
            numerical_columns = self.numerical_columns,
            binary_columns = self.binary_columns,
            hidden_dim = self.hidden_dim,
            num_layer = self.num_layer,
            num_attention_head = self.num_attention_head,
            hidden_dropout_prob = self.hidden_dropout_prob,
            ffn_dim = self.ffn_dim,
            activation = self.activation,
            vocab_freeze = self.vocab_freeze,
            device = self.device,
            mlm_probability=self.mlm_probability,
            projection_dim=self.projection_dim,
        )

        self.load_model(model)

        return model