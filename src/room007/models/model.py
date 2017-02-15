# vim: set fileencoding=utf-8 :
import itertools


class Option(object):
    def __init__(self, options, default):
        self.choices = {}
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
        option_choices = [option.choices.keys() for option in self.options.values()]
        option_combinations = [dict(zip(self.options.keys(), combination))
                               for combination in itertools.product(*option_choices)
                               ]
        if option_combinations[0]:
            return [(option_combination_name(combination), combination) for combination in option_combinations]
        else:
            return [('||| no parameter combinations |||', {option_name: option.default for option_name, option in self.options.items()})]

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
