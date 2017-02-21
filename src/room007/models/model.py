# vim: set fileencoding=utf-8 :
import importlib
from itertools import chain, product
import logging

from room007.logging import loggingmgr

loggingmgr.set_up()
logger = logging.getLogger(__name__)


def create_predictor(model, args=None, kwargs=None):
    # XXX args.args are not used
    logger.info('creating predictor')
    predictor_class = importlib.import_module(model).Predictor
    predictor = predictor_class(kwargs or dict())
    logger.info('predictor created')
    return predictor_class, predictor


class Option(object):
    def __init__(self, options, default):
        self.choices = options
        self.default = default


class OptionsSetter(object):
    _cls2inst = dict()

    # Make sure to create just one instance of each class inheriting from this
    # class.
    def __new__(cls):
        if cls not in OptionsSetter._cls2inst:
            OptionsSetter._cls2inst[cls] = (
                super(OptionsSetter, cls).__new__(cls))
        inst = OptionsSetter._cls2inst[cls]
        return inst

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
        """\
        Gets all combinations of all options possible, filtered with
        arg_options.

        :param Mapping arg_options: mapping from option names to values, to
            which the combinations should be filtered

        """

        def option_combination_name(options):
            template = "|||" + "{}:{}&"*len(options) + " |||"
            return template.format(*chain.from_iterable(options.items()))

        opt_names, opt_values = zip(*((name, option.choices.keys())
                                    for name, option in self.options.items()))
        all_combinations = (dict(zip(opt_names, combination))
                            for combination in product(*opt_values))
        filtered = [combination for combination in all_combinations
                    if all(combination[name] == value
                        for name, value in arg_options.items())]
        if filtered[0]:
            return [(option_combination_name(combination), combination)
                    for combination in filtered]
        else:
            return [('||| no parameter combinations |||',
                    {option_name: option.default
                    for option_name, option in self.options.items()})]


class Predictor(object):

    def __init__(self, options):
        self.set_options(options)

    def get_options(self):
        return OptionsSetter()

    def set_options(self, options):
        self.get_options().set(self, options)
        return self

    def fit(self, train_data):
        return self

    def predict(self, test_dataframe):
        predictions = [[''] for _ in range(test_dataframe)]
        return predictions
