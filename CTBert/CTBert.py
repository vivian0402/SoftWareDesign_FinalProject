import pdb
import os

from . import constants
from .modeling_CTBert import CTBertClassifier, CTBertFeatureExtractor, CTBertFeatureProcessor, CTBertRegression
from .modeling_CTBert import CTBertForCL, TableGPTForMask
from .modeling_CTBert import CTBertInputEncoder, CTBertModel
from .trainer_utils import CTBertCollatorForCL
from .data_loader import DataLoader

dev = 'cuda'
def build_classifier(
    categorical_columns=None,
    numerical_columns=None,
    binary_columns=None,
    feature_extractor=None,
    num_class=2,
    hidden_dim=128,
    num_layer=2,
    num_attention_head=8,
    hidden_dropout_prob=0,
    ffn_dim=256,
    activation='relu',
    vocab_freeze=False,
    use_bert=True,
    device=dev,
    checkpoint=None,
    **kwargs) -> CTBertClassifier:
    '''Build a :class:`CTBert.modeling_CTBert.CTBertClassifier`.

    Parameters
    ----------
    categorical_columns: list 
        a list of categorical feature names.

    numerical_columns: list
        a list of numerical feature names.

    binary_columns: list
        a list of binary feature names, accept binary indicators like (yes,no); (true,false); (0,1).
    
    feature_extractor: CTBertFeatureExtractor
        a feature extractor to tokenize the input tables. if not passed the model will build itself.

    num_class: int
        number of output classes to be predicted.

    hidden_dim: int
        the dimension of hidden embeddings.
    
    num_layer: int
        the number of transformer layers used in the encoder.
    
    num_attention_head: int
        the numebr of heads of multihead self-attention layer in the transformers.

    hidden_dropout_prob: float
        the dropout ratio in the transformer encoder.

    ffn_dim: int
        the dimension of feed-forward layer in the transformer layer.
    
    activation: str
        the name of used activation functions, support ``"relu"``, ``"gelu"``, ``"selu"``, ``"leakyrelu"``.
    
    device: str
        the device, ``"cpu"`` or ``"cuda:0"``.
    
    checkpoint: str
        the directory to load the pretrained CTBert model.

    Returns
    -------
    A CTBertClassifier model.

    '''
    model = CTBertClassifier(
        categorical_columns = categorical_columns,
        numerical_columns = numerical_columns,
        binary_columns = binary_columns,
        feature_extractor = feature_extractor,
        num_class=num_class,
        hidden_dim=hidden_dim,
        num_layer=num_layer,
        num_attention_head=num_attention_head,
        hidden_dropout_prob=hidden_dropout_prob,
        ffn_dim=ffn_dim,
        activation=activation,
        vocab_freeze=vocab_freeze,
        use_bert=use_bert,
        device=device,
        **kwargs,
    )
    
    if checkpoint is not None:
        model.load(checkpoint)

    return model


def build_regression(
    categorical_columns=None,
    numerical_columns=None,
    binary_columns=None,
    feature_extractor=None,
    hidden_dim=128,
    num_layer=2,
    num_attention_head=8,
    hidden_dropout_prob=0,
    ffn_dim=256,
    activation='relu',
    vocab_freeze=False,
    device=dev,
    checkpoint=None,
    **kwargs) ->  CTBertRegression:
    '''Build a :class:`CTBert.modeling_CTBert.CTBertClassifier`.

    Parameters
    ----------
    categorical_columns: list 
        a list of categorical feature names.

    numerical_columns: list
        a list of numerical feature names.

    binary_columns: list
        a list of binary feature names, accept binary indicators like (yes,no); (true,false); (0,1).
    
    feature_extractor: CTBertFeatureExtractor
        a feature extractor to tokenize the input tables. if not passed the model will build itself.

    num_class: int
        number of output classes to be predicted.

    hidden_dim: int
        the dimension of hidden embeddings.
    
    num_layer: int
        the number of transformer layers used in the encoder.
    
    num_attention_head: int
        the numebr of heads of multihead self-attention layer in the transformers.

    hidden_dropout_prob: float
        the dropout ratio in the transformer encoder.

    ffn_dim: int
        the dimension of feed-forward layer in the transformer layer.
    
    activation: str
        the name of used activation functions, support ``"relu"``, ``"gelu"``, ``"selu"``, ``"leakyrelu"``.
    
    device: str
        the device, ``"cpu"`` or ``"cuda:0"``.
    
    checkpoint: str
        the directory to load the pretrained CTBert model.

    Returns
    -------
    A CTBertClassifier model.

    '''
    model =  CTBertRegression(
        categorical_columns = categorical_columns,
        numerical_columns = numerical_columns,
        binary_columns = binary_columns,
        feature_extractor = feature_extractor,
        hidden_dim=hidden_dim,
        num_layer=num_layer,
        num_attention_head=num_attention_head,
        hidden_dropout_prob=hidden_dropout_prob,
        ffn_dim=ffn_dim,
        activation=activation,
        vocab_freeze=vocab_freeze,
        device=device,
        **kwargs,
    )
    
    if checkpoint is not None:
        model.load(checkpoint)

    return model


def build_contrastive_learner(
    categorical_columns=None,
    numerical_columns=None,
    binary_columns=None,
    projection_dim=128,
    num_partition=3,
    overlap_ratio=0.5,
    supervised=True,
    hidden_dim=128,
    num_layer=2,
    num_attention_head=8,
    hidden_dropout_prob=0,
    ffn_dim=256,
    activation='relu',
    device=dev,
    checkpoint=None,
    ignore_duplicate_cols=True,
    vocab_freeze=True,
    **kwargs,
    ): 
    model = CTBertForCL(
        categorical_columns = categorical_columns,
        numerical_columns = numerical_columns,
        binary_columns = binary_columns,
        num_partition= num_partition,
        hidden_dim=hidden_dim,
        num_layer=num_layer,
        num_attention_head=num_attention_head,
        hidden_dropout_prob=hidden_dropout_prob,
        supervised=supervised,
        ffn_dim=ffn_dim,
        projection_dim=projection_dim,
        overlap_ratio=overlap_ratio,
        activation=activation,
        vocab_freeze=vocab_freeze,
        device=device,
    )
    if checkpoint is not None:
        model.load(checkpoint)
    
    # build collate function for contrastive learning
    collate_fn = CTBertCollatorForCL(
        categorical_columns=categorical_columns,
        numerical_columns=numerical_columns,
        binary_columns=binary_columns,
        overlap_ratio=overlap_ratio,
        num_partition=num_partition,
        ignore_duplicate_cols=ignore_duplicate_cols
    )
    if checkpoint is not None:
        collate_fn.feature_extractor.load(os.path.join(checkpoint, constants.EXTRACTOR_STATE_DIR))

    return model, collate_fn


def build_mask_features_learner(
    categorical_columns=None,
    numerical_columns=None,
    binary_columns=None,
    mlm_probability=0.15,
    projection_dim=128,
    hidden_dim=128,
    num_layer=2,
    num_attention_head=8,
    hidden_dropout_prob=0,
    ffn_dim=256,
    activation='relu',
    vocab_freeze=True,
    device=dev,
    checkpoint=None,
    **kwargs,
    ): 
    model = TableGPTForMask(
        categorical_columns = categorical_columns,
        numerical_columns = numerical_columns,
        binary_columns = binary_columns,
        mlm_probability=mlm_probability,
        hidden_dim=hidden_dim,
        num_layer=num_layer,
        num_attention_head=num_attention_head,
        hidden_dropout_prob=hidden_dropout_prob,
        ffn_dim=ffn_dim,
        projection_dim=projection_dim,
        activation=activation,
        vocab_freeze=vocab_freeze,
        device=device,
    )
    if checkpoint is not None:
        model.load(checkpoint)

    return model