import tensorflow as tf
import numpy as np


def sce_focal_loss(ce_weight=1.0, rce_weight=0.3, gamma=0.1, class_weight=None):
    """
    Symmetric crossentropy focal loss with class weights for binary crossentropy

    Params:
    -------
    ce_weight : float
        Crossentropy weight
    rce_weight : float
        Reverse crossentropy weight
    gamma : float
        Gamma hyperparamter from focal loss
    class_weight : dict(0:float, 1:float)
        Dictionary of weights for class 0 and 1

    Return:
    -------
    sce_focal : fn
        Custom loss function for keras model.
    """
    if class_weight:
        if not(0 in class_weight):
            print("0 is not a class in class_weight")
        if not(1 in class_weight):
            print("1 is not a class in class_weight")
    print('class_weight', class_weight)

    def sce_focal(y_true, y_pred):
        """
        y_true: Ground truth values. shape = [batch_size, 1]
        y_pred: The predicted values. shape = [batch_size, 1]
        """
        tf.debugging.assert_all_finite(y_true, "y_true contains NaNs", name=None)
        tf.debugging.assert_all_finite(y_pred, "y_pred contains NaNs", name=None)
        _epsilon = tf.keras.backend.epsilon()
        y_pred = tf.clip_by_value(y_pred, _epsilon, 1. - _epsilon)
        y_true = tf.clip_by_value(y_true, _epsilon, 1. - _epsilon)
        ce = tf.keras.losses.binary_crossentropy(y_true, y_pred, from_logits=False)
        tf.debugging.assert_all_finite(ce, "ce contains NaNs", name=None)
        rce = tf.keras.losses.binary_crossentropy(y_pred, y_true, from_logits=False)
        tf.debugging.assert_all_finite(ce, "rce contains NaNs", name=None)
        factor = tf.squeeze(tf.math.pow(tf.abs(y_true-y_pred), gamma))  # Focal
        tf.debugging.assert_all_finite(factor, "factor contains NaNs", name=None)
        focal_ce = tf.multiply(factor, ce)
        focal_rce = tf.multiply(factor, rce)
        if class_weight:
            class_0 = tf.squeeze(tf.multiply(tf.abs(1.0-y_true), class_weight[0]))
            class_1 = tf.squeeze(tf.multiply(y_true, class_weight[1]))
            weight = tf.add(class_0, class_1)
            focal_ce = tf.multiply(weight, focal_ce)
            focal_rce = tf.multiply(weight, focal_rce)
        focal_ce = tf.reduce_mean(focal_ce)
        focal_rce = tf.reduce_mean(focal_rce)
        loss = ce_weight*focal_ce + rce_weight*focal_rce
        return loss
    return sce_focal


def weighted_bce_loss(class_weight=None, reduction_factor=0.9):
    if class_weight:
        if not(0 in class_weight):
            print("0 is not a class in class_weight")
        if not(1 in class_weight):
            print("1 is not a class in class_weight")
    reduction_factor = np.sqrt(reduction_factor)
    class_weight[0] = class_weight[0]/reduction_factor
    class_weight[1] = class_weight[0]*reduction_factor
    print('class_weight', class_weight)

    def weighted_bce(y_true, y_pred):
        """
        y_true: Ground truth values. shape = [batch_size, 1]
        y_pred: The predicted values. shape = [batch_size, 1]
        """
        _epsilon = tf.keras.backend.epsilon()
        y_pred = tf.clip_by_value(y_pred, _epsilon, 1. - _epsilon)
        bce = tf.keras.losses.binary_crossentropy(y_true, y_pred, from_logits=False)
        class_0 = tf.squeeze(tf.multiply(tf.abs(1.0-y_true), class_weight[0]))
        class_1 = tf.squeeze(tf.multiply(y_true, reduction_factor * class_weight[1]))
        weight = tf.add(class_0, class_1)
        weighted_bce_ = tf.multiply(weight, bce)
        loss = tf.reduce_mean(weighted_bce_)
        return loss
    return weighted_bce


def sce_loss(ce_weight=1.0, rce_weight=0.1, class_weight=None):
    """
    Symmetric crossentropy focal loss with class weights for binary crossentropy

    Params:
    -------
    ce_weight : float
        Crossentropy weight
    rce_weight : float
        Reverse crossentropy weight
    class_weight : dict(0:float, 1:float)
        Dictionary of weights for class 0 and 1

    Return:
    -------
    sce : fn
        Custom loss function for keras model.
    """
    if class_weight:
        if not(0 in class_weight):
            print("0 is not a class in class_weight")
        if not(1 in class_weight):
            print("1 is not a class in class_weight")
    print('class_weight', class_weight)

    def sce(y_true, y_pred):
        """
        y_true: Ground truth values. shape = [batch_size, 1]
        y_pred: The predicted values. shape = [batch_size, 1]
        """
        _epsilon = tf.keras.backend.epsilon()
        y_pred = tf.clip_by_value(y_pred, _epsilon, 1. - _epsilon)
        y_true = tf.clip_by_value(y_true, _epsilon, 1. - _epsilon)
        ce = tf.keras.losses.binary_crossentropy(y_true, y_pred, from_logits=False)
        rce = tf.keras.losses.binary_crossentropy(y_pred, y_true, from_logits=False)
        if class_weight:
            class_0 = tf.squeeze(tf.multiply(tf.abs(1.0-y_true), class_weight[0]))
            class_1 = tf.squeeze(tf.multiply(y_true, class_weight[1]))
            weight = tf.add(class_0, class_1)
            ce = tf.multiply(weight, ce)
            rce = tf.multiply(weight, rce)
        ce = tf.reduce_mean(ce)
        rce = tf.reduce_mean(rce)
        loss = ce_weight*ce + rce_weight*rce
        return loss
    return sce
