# vim: set fileencoding=utf-8 :
import itertools
import abc

from room007.data import feature_data

class Option(object):
    def __init__(self, options, default):
        self.choices = options
        self.default = default


class OptionsSetter(object):
    def __init__(self):
        self.options = {}

    def set(self, instance, arg_options):
        """ set the attributes of the instances dependent on this options.
        if an option_name is set in kwargs, then those are used.
        else the default values are used
        """
        for option_name, option in self.options.items():
            choice = arg_options.get(option_name, option.default)
            option_value = option.choices[choice]
            setattr(instance, option_name, option_value)

    def combinations(self, arg_options):
        """ get all combinations of all options possible, filtered with arg_options
        for the moment filtering is not implemented
        """
        def option_combination_name(options):
            template = "|||" + "{}:{}&"*len(options) + " |||"
            return template.format(*itertools.chain(*options.items()))
        def subselection(option_name, option_choices):
            if option_name not in arg_options:
                return option_choices
            else:
                return [arg_options[option_name]]
        option_choices = [subselection(option_name, option.choices.keys())
                          for option_name, option in self.options.items()]
        option_combinations = [dict(zip(self.options.keys(), combination))
                               for combination in itertools.product(*option_choices)
                               ]
        if option_combinations[0]:
            return [(option_combination_name(combination), combination) for combination in option_combinations]
        else:
            return [('||| no parameter combinations |||', {option_name: option.default for option_name, option in self.options.items()})]


class Features(object):
    __metaclass__ = abc.ABCMeta

    def save_features_to_dataframe(self, data):
        feature_data.FeatureManager.save = True
        self._get_data_independent_features(data)
        feature_data.FeatureManager.save = False

    @abc.abstractmethod
    def _get_data_independent_features(self, data):
        pass

class Predictor(object):
    OptionsSetter = OptionsSetter

    def __init__(self):
        pass

    def set_options(self, arg_options):
        options_setter = self.OptionsSetter()
        options_setter.set(self, arg_options)

    def fit(self, train_data):
        print('start fitting {}'.format(len(train_data)))

    def predict(self, test_dataframe):
        print('start predicting')
        predictions = [[''] for _ in range(test_dataframe)]
        return predictions

