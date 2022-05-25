
import tensorflow as tf
import logictensornetworks as ltn
from tensorflow.keras import layers
from config import config
from metrics import Top1Accuracy
from tensorflow.keras.regularizers import l2
import numpy as np

initializer= tf.keras.initializers.TruncatedNormal(stddev=0.01,  seed=None)


def function_is_class(matrix1,matrix2,config_file):
    return  tf.exp(-config_file.alpha * tf.square(tf.norm(matrix1 - matrix2, axis=1)))

class embeddingModel(tf.keras.Model):
    def __init__(self,config_file):
        super(embeddingModel, self).__init__()
        self.denses = [layers.Dense(s, activation=config_file.activation_function, kernel_initializer=initializer) for s in config_file.hidden_dense_sizes]
        self.densesuper = layers.Dense(2048,activation=config_file.activation_function, kernel_initializer=initializer)

        self.satisfabilityaggregation = ltn.fuzzy_ops.Aggreg_pMeanError(p=config_file.p_agg)
        self.config_file=config_file
        if config_file.similarity=="euclidean_distance":
            self.isOfClass=  ltn.Predicate.Lambda(lambda vars: function_is_class(vars[0],vars[1],config_file))
        else:
            self.isOfClass = ltn.Predicate.Lambda(
                lambda vars: tf.exp(-config_file.alpha * tf.norm(vars[0] - vars[1], axis=1)))

        self.Not = ltn.Wrapper_Connective(ltn.fuzzy_ops.Not_Std())
        self.And = ltn.Wrapper_Connective(ltn.fuzzy_ops.And_Prod())
        self.Or = ltn.Wrapper_Connective(ltn.fuzzy_ops.Or_ProbSum())
        self.Implies = ltn.Wrapper_Connective(ltn.fuzzy_ops.Implies_Reichenbach())
        if config_file.forAllAggregator=="Aggreg_pMeanError":
            self.Forall = ltn.Wrapper_Quantifier(ltn.fuzzy_ops.Aggreg_pMeanError(), semantics="forall")
        elif config_file.forAllAggregator=="Aggreg_pProd":
            self.Forall = ltn.Wrapper_Quantifier(ltn.fuzzy_ops.Aggreg_pProd(), semantics="forall")
        else:
            return -1
        self.StandardForall = ltn.Wrapper_Quantifier(ltn.fuzzy_ops.Aggreg_pMeanError(), semantics="forall")
        self.Exists = ltn.Wrapper_Quantifier(ltn.fuzzy_ops.Aggreg_pMean(), semantics="exists")
        self.satisfiabilityAggregation = ltn.fuzzy_ops.Aggreg_pMeanError(p=config_file.p_agg)

    def call(self, x, training=False):
        for dense in self.denses:
            x = dense(x)
        return x



    def axioms(self,train_feature, train_label, search_space, prototype1,config_file,p_schedule=tf.constant(2.)):

        prototype1_label = ltn.variable("prototype1_label", search_space)
        train_feature = ltn.variable("train_feature", train_feature)
        train_label = ltn.variable("train_label", train_label)
        print("train_label",train_label)
        print("prototype1_label",prototype1_label)

        if config_file.negation_axioms:

            axioms = [
                self.Forall(
                    ltn.diag(train_feature, train_label),
                    self.Forall(
                        ltn.diag(prototype1, prototype1_label),
                        self.isOfClass([train_feature, prototype1]),
                        mask_vars=[train_label, prototype1_label],
                        mask_fn=lambda vars: tf.math.equal(vars[0], vars[1]),
                        p=tf.constant(2.)
                    ),
                    p=tf.constant(2.)
                ),
                self.StandardForall(
                    ltn.diag(train_feature, train_label),
                    self.StandardForall(
                        ltn.diag(prototype1, prototype1_label),
                        self.Not(self.isOfClass([train_feature, prototype1])),
                        mask_vars=[train_label, prototype1_label],
                        mask_fn=lambda vars: tf.math.not_equal(vars[0], vars[1]),
                        p=tf.constant(2.)
                    ),
                    p=tf.constant(2.)
                )
            ]
        else:
            axioms = [
                self.Forall(
                    ltn.diag(train_feature, train_label),
                    self.Forall(
                        ltn.diag(prototype1, prototype1_label),
                        self.isOfClass([train_feature, prototype1]),
                        mask_vars=[train_label, prototype1_label],
                        mask_fn=lambda vars: tf.math.equal(vars[0], vars[1]),
                        p=p_schedule
                    ),
                    p=p_schedule
                )
            ]

        axioms = tf.stack(axioms)

        return axioms, prototype1



