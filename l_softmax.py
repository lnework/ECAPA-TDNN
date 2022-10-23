#导入库
import tensorflow.python as tf
from tensorflow import contrib
def Angular_Softmax_Loss():

    def forward(self, embeddings, labels=None, margin=2):
        """
                Note:(about the value of margin)
                as for binary-class case, the minimal value of margin is 2+sqrt(3)
                as for multi-class  case, the minimal value of margin is 3

                the value of margin proposed by the author of paper is 4.
                here the margin value is 4.
                """
        l = 0.
        embeddings = tf.random_normal((2, 10))
        labels = tf.convert_to_tensor([[1], [2]], dtype=tf.int64)
        x_norm = tf.norm(embeddings, axis=1)

        with tf.variable_scope("softmax"):
            weights = tf.get_variable(name='embedding_weights',
                                      shape=[embeddings.get_shape().as_list()[-1], 10],
                                      initializer=contrib.layers.xavier_initializer())
            W = tf.nn.l2_normalize(weights, axis=0)
            # cacualting the cos value of angles between embeddings and W
            orgina_logits = tf.matmul(embeddings, W)
            N = embeddings.get_shape()[0]  # get batch_size
            single_sample_label_index = tf.concat([tf.constant(list(range(N)), tf.int64, shape=(N, 1)), labels], axis=1)
            # N = 128, labels = [1,0,...,9]
            # single_sample_label_index:
            # [ [0,1],
            #   [1,0],
            #   ....
            #   [128,9]]
            # 这里就是F_y_i,根据有目标的位置来选取需要计算的loss位置.
            f_y_i = tf.gather_nd(orgina_logits, single_sample_label_index)
            # NOTE 因为 \parallel W\parallel =1 所以 cos(theta)=f_y_i/x_norm
            cos_theta = tf.div(f_y_i, x_norm)
            cos_theta_2 = tf.pow(cos_theta, 2)
            cos_theta_4 = tf.pow(cos_theta, 4)

            sign0 = tf.sign(cos_theta)
            sign3 = tf.multiply(tf.sign(2 * cos_theta_2 - 1), sign0)
            sign4 = 2 * sign0 + sign3 - 3
            result = sign3 * (8 * cos_theta_4 - 8 * cos_theta_2 + 1) + sign4

            margin_logits = tf.multiply(result, x_norm)
            f = 1.0 / (1.0 + l)
            ff = 1.0 - f
            combined_logits = tf.add(orgina_logits,
                                     tf.scatter_nd(single_sample_label_index,
                                                   tf.subtract(margin_logits, f_y_i),
                                                   orgina_logits.get_shape()))
            updated_logits = ff * orgina_logits + f * combined_logits
            loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits_v2(logits=updated_logits,
                                                                                    labels=tf.reshape(labels, (-1,))))
            pred_prob = tf.nn.softmax(logits=updated_logits)
            return loss, pred_prob