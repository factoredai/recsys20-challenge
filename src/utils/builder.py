from utils import models
import tensorflow as tf
from utils.losses import weighted_bce_loss, sce_focal_loss, sce_loss
from utils.optimizers import WarmMaxConvSchedule


def build_model(hparams):
    """
    Functional model to try

    Return:
    -------
    model : tf.keras.Model
        Not compiled keras model
    """
    model_name = hparams["model_name"]
    model2use = getattr(models, model_name)
    model = model2use(hparams)
    return model


def build_metrics(hparams):
    """
    Metrics to build

    Return:
    -------
    metrics : list(tf.keras.metrics)
        Metrics to be used for compiling
    """
    metrics = []
    for metrics_name in hparams["metrics"]:
        if metrics_name == "PR_AUC":
            metrics.append(tf.keras.metrics.AUC(curve='PR', name=metrics_name))
        elif metrics_name == "BCE":
            metrics.append(tf.keras.metrics.BinaryCrossentropy(name=metrics_name))
    return metrics


def build_loss(hparams):
    """
    Loss to build

    Return:
    -------
    dict_loss : dict(labels: loss)
        Dictionary of losses to be used for each label
    dict_loss_weights : dict(labels: loss_weight)
        Dictionary of weights for each loss
    """
    dict_labels = hparams["labels"]
    dict_loss = {}
    dict_loss_weights = {}
    for key, val in dict_labels.items():
        dict_loss_weights[key] = dict_labels[key]["loss_weight"]
        dict_loss_name = dict_labels[key]["loss"]["name"]
        dict_loss_params = dict_labels[key]["loss"]["params"]
        if dict_loss_name == "bce":
            dict_loss[key] = "binary_crossentropy"
        elif dict_loss_name == "weighted_bce":
            dict_loss[key] = weighted_bce_loss(**dict_loss_params)
        elif dict_loss_name == "sce_focal_loss":
            dict_loss[key] = sce_focal_loss(**dict_loss_params)
        elif dict_loss_name == "sce_loss":
            dict_loss[key] = sce_loss(**dict_loss_params)
        else:
            print(f"Loss {dict_loss_name} not available")
    return dict_loss, dict_loss_weights


def build_optimizer(hparams):
    """
    Optimizer to build

    Return:
    -------
    optimizer : tf.keras.Optimizer
        Optimizer to use
    """
    optimizer_dict = hparams["optimizer"]
    optimizer_name = optimizer_dict["name"]
    optimizer_params = optimizer_dict["params"]
    if optimizer_name == "adam":
        optimizer = tf.keras.optimizers.Adam(
            learning_rate=optimizer_params["lr"],
            beta_1=0.9, beta_2=0.999,
            epsilon=1e-07, amsgrad=False,
            name='adam'
        )
    elif optimizer_name == "ws_max_conv":
        learning_rate = WarmMaxConvSchedule(
            optimizer_params['lr_max'],
            optimizer_params['lr_conv'],
            optimizer_params['warmup_steps']
        )
        optimizer = tf.keras.optimizers.Adam(
            learning_rate=learning_rate,
            beta_1=0.9,
            beta_2=0.999,
            epsilon=1e-07,
            amsgrad=False,
            name='adam_warmmaxconv'
        )
    else:
        print(f"Optimizer {optimizer_name} doesn't exist")
    return optimizer


def compile_model(hparams):
    """
    Compile selected model
    """
    model = build_model(hparams)
    metrics = build_metrics(hparams)
    loss, loss_weights = build_loss(hparams)
    optimizer = build_optimizer(hparams)
    model.compile(
        optimizer=optimizer,
        loss=loss,
        metrics=metrics,
        loss_weights=loss_weights
    )
    return model
