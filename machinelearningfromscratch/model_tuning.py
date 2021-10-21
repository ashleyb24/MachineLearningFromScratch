import numpy as np
import copy
import matplotlib.pyplot as plt
from matplotlib.text import Text
from matplotlib.widgets import Slider, Button


title_text = "Original score: {:.5f}, Current score: {:.5f}"


class AdjustableCoefficient:

    def __init__(self, name, init_value):
        self.name = name
        self.init_value = init_value

    def __repr__(self):
        return "{} : {}".format(self.name, self.init_value)

    def __str__(self):
        return "{} : {}".format(self.name, self.init_value)

    def _min_max_values(self):
        switch = 1
        if self.init_value < 0:
            switch = -1
        min_val = self.init_value * -10 * switch
        max_val = self.init_value * 10 * switch
        return min_val, max_val

    def build_slider(self, axes):
        slider_min, slider_max = self._min_max_values()
        return Slider(ax=axes, label=self.name, valmin=slider_min, valmax=slider_max, valinit=self.init_value)


class LinearRegressionTuning:

    def __init__(self, model, x, y, predictions):
        self._x = x
        self._y = y
        self._predictions = predictions
        self._feature_count = np.size(x[0, :])

        # copies
        self._original_coefficients = model.coefficients
        self._original_score = model.score(y, predictions)
        self._model = copy.copy(model)
        self._coefficients = self._model.coefficients

        # init new variables
        self._features = []
        self._feature_coefficients = {}
        self._feature_indices = {}
        self._feature_prediction_plot = {}
        self._fig_title = Text()
        self._sliders = {}

    def _adjustable_features(self, names):
        if len(self._features) == 0:
            raise
        adjustable_features = []
        for name in names:
            if name not in self._features:
                raise ValueError('Incorrect feature name given')
            adjustable_feature = AdjustableCoefficient(name, self._feature_coefficients[name])
            adjustable_features.append(adjustable_feature)
        return adjustable_features

    def _slider_update(self, val):
        # update coefficients for model based on slider values
        for key in self._sliders:
            slider = self._sliders[key]
            index = self._feature_indices[key]
            self._coefficients[index] = slider.val
        # use updated model to make new predictions
        new_predictions = []
        for new_row in self._x:
            new_pred = self._model.predict(new_row)
            new_predictions.append(new_pred)
        new_score = self._model.score(self._y, new_predictions)
        # plot new predictions
        for key in self._feature_prediction_plot:
            plot = self._feature_prediction_plot[key]
            index = self._feature_indices[key]
            plot.set_offsets(np.c_[self._x[:, index], new_predictions])
            self._fig_title.set_text(title_text.format(self._original_score, new_score))

    def _slider_reset(self, event):
        for key in self._sliders:
            slider = self._sliders[key]
            slider.reset()

    def _close_plot(self, event):
        plt.close()

    def set_features(self, names):
        if self._feature_count != len(names):
            raise ValueError('Incorrect number of feature names given')
        else:
            self._features = names
            for i, name in enumerate(names):
                self._feature_coefficients[name] = self._coefficients[i]
                self._feature_indices[name] = i

    def plot_features(self, feature_names):
        if len(feature_names) > 4:
            raise ValueError('Currently no more than 4 features can be plotted at once.')
        fig, axes = plt.subplots(nrows=1, ncols=len(feature_names), figsize=(14, 7))
        self._fig_title = fig.text(0.25, 0.90, title_text.format(self._original_score, self._original_score))
        plt.subplots_adjust(bottom=0.35)
        adjustable_features = self._adjustable_features(feature_names)
        for i, feat in enumerate(adjustable_features):
            # plot actual and predictions against features
            x_index = self._feature_indices[feat.name]
            axes[i].scatter(self._x[:, x_index], self._y, marker='.', label='Actual')
            self._feature_prediction_plot[feat.name] = axes[i].scatter(self._x[:, x_index], self._predictions,
                                                                       marker='.', label='Prediction')
            axes[i].set_xlabel(feat.name)
            axes[i].legend()

            # add slider for feature
            slider_ax = plt.axes([0.15, 0.1 + (0.05 * i), 0.65, 0.03])
            slider = feat.build_slider(slider_ax)
            slider.on_changed(self._slider_update)
            self._sliders[feat.name] = slider

        # add reset button for sliders
        reset_ax = plt.axes([0.05, 0.025, 0.1, 0.04])
        button = Button(ax=reset_ax, label='Reset', hovercolor='0.975')
        button.on_clicked(self._slider_reset)

        # add button to save model
        save_ax = plt.axes([0.8, 0.025, 0.1, 0.04])
        save_button = Button(ax=save_ax, label='Save', hovercolor='0.975')
        save_button.on_clicked(self._close_plot)

        plt.show()
        return self._model
