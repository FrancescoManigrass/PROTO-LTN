import os

import neptune


class neptune_class:
    def __init__(self,config_file):
        self.config_file =config_file
        self.sources = [f for f in os.listdir() if f.endswith(".py")]
        self.sources.append(self.config_file.base_path+"/DatasetOps.py")

    def get_tags(self):

        new_dict = {"ExperimentName": self.config_file.experiment_name,
                    "learning_rate": self.config_file.learning_rate,
                    "alpha": self.config_file.alpha,
                    "regularization_parameter": self.config_file.regularization_parameter if self.config_file.regularize is True else False,
                    "epochs": self.config_file.epochs,
                    "loss": self.config_file.loss,
                    "forAll": self.config_file.loss,
                    "negative_examples": self.config_file.negative_experiment,
                    "distance": self.config_file.similarity,
                    }
        return [x + "=" + y.__str__() for x, y in new_dict.items()]

    def get_params(self):
        new_dict = {'learning_rate': self.config_file.learning_rate,
                    'epochs': self.config_file.epochs,
                    'optimizer': 'Adam',
                    'loss': self.config_file.loss,
                    'p_agg': self.config_file.p_agg,
                    'p_forall': self.config_file.p_agg_for_all,
                    'hidden_dense_sizes': self.config_file.hidden_dense_sizes,
                    'activation_functions': self.config_file.activation_function,
                    'batch_size': self.config_file.batch_size,
                    'similarity': self.config_file.similarity,
                    # 'dropout': dropout,
                    'alpha_similarity': self.config_file.alpha,
                    'regularize': self.config_file.regularize,
                    'regularization_parameter': self.config_file.regularization_parameter,
                    'dataset': self.config_file.base_path
                    }

        return new_dict

    def log_history_metric(self,name, values):
        for i in range(len(values)):

            if values[i]:
                neptune.log_metric(name, values[i])

    def create_experiment_neptune(self):
        if self.config_file.neptune_flag:
            self.experiment = neptune.init(self.config_file.project_name,
                                      api_token=self.config_file.token)

        # create experiment with defined parameters, uploaded source code and tags
        if self.config_file.neptune_flag:
            neptune.create_experiment(name=self.config_file.experiment_name, params=self.get_params(),
                                      upload_source_files=self.sources)

        # add convenient tags for future filtering
        if self.config_file.neptune_flag:
            neptune.append_tags(self.get_tags())

    def stop(self):
        neptune.stop()