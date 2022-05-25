import os


class config:
    def __init__(self):
        self.buffer_size = 32
        self.batch_size = 32
        self.base_path = 'data/AwA2_data/'

        self.project = ""
        self.epochs =10
        self.p_agg = 1
        self.p_agg_for_all = 2
        self.satisfiabilityAggregation = "Aggreg_pMeanError" #sempre questo

        self.loss="1-aggregator"#"1-aggregator" #  log
        self.forAllAggregator = "Aggreg_pProd" #Aggreg_pMeanError Aggreg_pProd
        self.experiment_name = 'Awa2_ENDTOEND_ZSL_prototypes_with_LTN_v4_negation_TRUNCATED_0.01'
        self.negative_experiment = False
        self.pretrained = False
        self.weights=""
        self.similarity = "euclidean_distance" # normal_distance
        self.negation_axioms=False
        self.activation_function= "relu"
        self.hidden_dense_sizes = [1600,2048]
        self.eps = 1e-4
        self.alpha = 1e-7
        self.regularize = True
        self.neptune_flag = False
        self.regularization_parameter = 1e-5
        self.learning_rate = 1e-4


        #neptune
        self.project_name=""

        self.token =""


