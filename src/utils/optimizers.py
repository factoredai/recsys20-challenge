import tensorflow as tf


class WarmMaxConvSchedule(tf.keras.optimizers.schedules.LearningRateSchedule):
    def __init__(self, lr_max, lr_conv, warmup_steps=4000, name="warm_maxconv"):
        """ 
            lr_max
             /\ 
            /  \__________ lr_conv (-> inf)
           /   
          /
         /
        /
        |<-ws->|
        lr_max : float
          Maximum learning rate
        lr_conv : float
          Convergence learning rate
        warmup_steps : int
          Warm up steps
        """
        super(WarmMaxConvSchedule, self).__init__()

        self.lr_max_native = lr_max
        self.lr_max = tf.cast(self.lr_max_native, tf.float32)
        self.name = name

        self.c = 0.5  # Speed to converge

        self.lr_conv_native = lr_conv
        self.lr_conv = tf.cast(self.lr_conv_native, tf.float32)

        self.warmup_steps = warmup_steps

    def __call__(self, step):
        arg1 = self.lr_max/self.warmup_steps*step
        arg2 = (self.warmup_steps**self.c*(self.lr_max-self.lr_conv))/(step**self.c)+self.lr_conv
        return tf.math.minimum(arg1, arg2)

    def get_config(self):
        config = {
            'lr_max': self.lr_max_native,
            'lr_conv': self.lr_conv_native,
            'warmup_steps': self.warmup_steps,
            'name': self.name
        }
        return config
