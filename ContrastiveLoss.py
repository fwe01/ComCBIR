import tensorflow as tf

class ContrastiveLoss(tf.keras.losses.Loss):
    def __init__(self, margin=1.0, verbose=False, name="contrastive_loss"):
        super().__init__(name=name)
        self.verbose = verbose
        self.margin = margin
        self.class_centroids = None

    def call(self, y_true, y_pred):
        """
        Args:
        y_true: Tensor of shape (batch_size, 1) containing the class labels.
        y_pred: Tensor of shape (batch_size, feature_dim) containing the feature representations.
        """

        # Check for NaNs in y_pred
        if tf.reduce_any(tf.math.is_nan(y_pred)):
            raise ValueError("NaN values found in y_pred")
        
        y_true = tf.cast(y_true, tf.int32)

        # Calculate pairwise distances
        # Reshape y_pred to (batch_size/2, 2, feature_dim)
        y_pred_reshaped = tf.reshape(y_pred, (-1, 2, y_pred.shape[1]))
        positive_dist = tf.reduce_sum(tf.square(y_pred_reshaped[:, 0] - y_pred_reshaped[:, 1]), axis=1)

        # Reshape y_true to (batch_size/2, 2)
        y_true_reshaped = tf.reshape(y_true, (-1, 2))
        labels = tf.cast(y_true_reshaped[:, 0] == y_true_reshaped[:, 1], tf.float32)

        # Calculate contrastive loss
        positive_loss = labels * positive_dist
        # negative_loss = (1 - labels) * tf.square(tf.maximum(self.margin - tf.sqrt(positive_dist), 0))
        negative_loss = (1 - labels) * tf.maximum(self.margin * self.margin - positive_dist, 0)
        loss = tf.reduce_mean(positive_loss + negative_loss)

        if self.verbose:
            tf.print("Positive distances:", positive_dist)
            tf.print("Labels:", labels)
            tf.print("Positive loss:", positive_loss)
            tf.print("Negative loss:", negative_loss)
            tf.print("Loss:", loss)

        return loss