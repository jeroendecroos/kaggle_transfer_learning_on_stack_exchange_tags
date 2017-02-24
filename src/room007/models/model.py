# vim: set fileencoding=utf-8 :
import itertools


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


class FeatureManager(object):
    save = False
    def __init__(self, fun=None):
        self.fun = fun

    def __call__(self, *args, **kwargs):
        data = args[0]
        feature_name = self._get_feature_name(args, kwargs)
        if feature_name in data:
            features = data[feature_name].apply(lambda x: eval(x) if '[' in x else x)
        else:
            features = self.fun(*args, **kwargs)
        if self.save:
            data[feature_name] = features
        return features

    def _get_feature_name(self, args, kwargs):
        """ get the feature name for a function, dependent on the argument and kwargs
        args & kwargs are not expanded because they are treated as single arguments
        """
        feature_name = self.fun.__name__
        if len(args)>1:
            feature_name += '_'
            feature_name += '_'.join(args[1:])
        if kwargs:
            feature_name += '_'
            feature_name += '_'.join([key+':'+value for key, value in sorted(kwargs.items())])
        return feature_name

class Features(object):
    def save_features_to_dataframe(self, data):
        FeatureManager.save = True
        self._get_data_independent_features(data)
        FeatureManager.save = False


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

