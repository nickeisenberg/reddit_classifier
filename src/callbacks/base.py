from abc import ABC


class Callback(ABC):
    def before_all_epochs(self, *args, **kwargs):
        pass

    def before_train_epoch_pass(self, *args, **kwargs):
        pass

    def before_train_batch_pass(self, *args, **kwargs):
        pass

    def after_train_batch_pass(self, *args, **kwargs):
        pass

    def after_train_epoch_pass(self, *args, **kwargs):
        pass

    def before_validation_epoch_pass(self, *args, **kwargs):
        pass

    def before_validation_batch_pass(self, *args, **kwargs):
        pass

    def after_validation_batch_pass(self, *args, **kwargs):
        pass

    def after_validation_epoch_pass(self, *args, **kwargs):
        pass

    def after_all_epochs(self, *args, **kwargs):
        pass
