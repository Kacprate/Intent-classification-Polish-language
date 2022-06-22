import argparse
import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sklearn
from sklearn import metrics


class DataHandler(ABC):
    """
    Abstract class for data handling.
    """
    
    @abstractmethod
    def load_data(self) -> Tuple[np.ndarray, int]:
        pass

class CSVHandler(DataHandler):
    """
    CSV Data handler
    """
    
    def __init__(self, csv_file: str, logger: logging.Logger) -> None:
        super().__init__()
        assert csv_file is not None, "No file provided."
        assert csv_file.endswith('.csv'), "Provided file must be a csv file."
        self._file_path: str = csv_file
        self._logger: logging.Logger = logger
        
    def load_data(self) -> Tuple[np.ndarray, int]:
        try:
            self._logger.info("Loading data")
            csv_data = pd.read_csv(self._file_path, header=None)
        except Exception as e:
            self._logger.error(e)
            raise e
        prediction_vector = csv_data.loc[:, :59].to_numpy()
        ground_truth_label = csv_data.loc[:, 60].to_numpy()
        self._logger.info(f"Data loaded successfully (samples count: {len(csv_data)})")
        return (prediction_vector, ground_truth_label)

class AnalyticOutput(ABC):
    """
    Abstract class for analytic output.
    """
    
    @abstractmethod
    def print(self) -> None:
        pass
    
    @abstractmethod
    def plot(self) -> None:
        pass
    
    @abstractmethod
    def save_to_file(self, path) -> None:
        pass
    
@dataclass
class ConfusionMatrixAnalyticOutput(AnalyticOutput):
    """
    Analytic output class for confusion matrix data.
    """
    
    y_true: np.ndarray
    y_pred: np.ndarray
    confusion_matrix: np.ndarray
    
    def print(self) -> None:
        raise NotImplementedError
    
    def plot(self) -> None:
        disp = metrics.ConfusionMatrixDisplay.from_predictions(self.y_true, self.y_pred, include_values=False)
        disp.plot()
        plt.show()
    
    def save_to_file(self, path) -> None:
        raise NotImplementedError
    
@dataclass
class ClassificationReportOutput(AnalyticOutput):
    """
    Analytic output class for confusion matrix data.
    """
    
    classification_report: dict
    
    def print(self) -> None:
        print(self.classification_report)
    
    def plot(self) -> None:
        raise NotImplementedError
    
    def save_to_file(self, path) -> None:
        raise NotImplementedError
    
@dataclass
class Analytic(ABC):
    """
    Abstract class for analytics.
    """
    data: Tuple[np.ndarray, int]
        
    @abstractmethod
    def evaluate(self) -> AnalyticOutput:
        pass

class ConfusionMatrixAnalytic(Analytic):
    """
    Analytic for evaluating the confusion matrix data.
    """
    
    def evaluate(self) -> AnalyticOutput:
        y_pred = self.data[0].argmax(axis=1)
        y_true = self.data[1]
        confusion_matrix = metrics.confusion_matrix(y_true, y_pred)
        output = ConfusionMatrixAnalyticOutput(y_true, y_pred, confusion_matrix)
        return output
    
class ClassificationReportAnalytic(Analytic):
    """
    Analytic for evaluating the confusion matrix data.
    """
    
    def evaluate(self) -> AnalyticOutput:
        y_pred = self.data[0].argmax(axis=1)
        y_true = self.data[1]
        classification_report = metrics.classification_report(y_true, y_pred)
        output = ClassificationReportOutput(classification_report)
        return output

        
def main():
    logging.basicConfig()
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)

    parser = argparse.ArgumentParser()
    parser.add_argument('--data', type=str, default=None)
    
    args = parser.parse_args()
    
    csv_handler = CSVHandler(args.data, logger)
    data = csv_handler.load_data()
    
    confusion_matrix_analytic = ConfusionMatrixAnalytic(data)
    confusion_matrix_analytic_output = confusion_matrix_analytic.evaluate()
    # confusion_matrix_analytic_output.plot()
    
    classification_report_analytic = ClassificationReportAnalytic(data)
    classification_report_output = classification_report_analytic.evaluate()
    classification_report_output.print()
    

if __name__ == "__main__":
    main()
