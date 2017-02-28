# vim: set fileencoding=utf-8 :

class FeatureManager(object):
    save = False
    def __init__(self, fun=None):
        self.fun = fun

    def __call__(self, *args, **kwargs):
        data = args[0]
        feature_name = self._get_feature_name(args, kwargs)
        if feature_name in data:
            if '[' in data[feature_name].iloc[0]:
                # This is to store lists in the CSV file. We should find a better way
                features = data[feature_name].apply(lambda x: eval(x))
            else:
                features = data[feature_name]
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

