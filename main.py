import sys
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from PyQt6.QtWidgets import (
    QApplication, QWidget, QTabWidget, QVBoxLayout, QComboBox, QLabel,
    QPlainTextEdit, QTableView
)
from PyQt6.QtCore import Qt, QAbstractTableModel

from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas


class PandasModel(QAbstractTableModel):
    def __init__(self, df=pd.DataFrame()):
        super().__init__()
        self._df = df

    def rowCount(self, parent=None):
        return self._df.shape[0]

    def columnCount(self, parent=None):
        return self._df.shape[1]

    def data(self, index, role=Qt.ItemDataRole.DisplayRole):
        if not index.isValid():
            return None
        if role == Qt.ItemDataRole.DisplayRole:
            value = self._df.iloc[index.row(), index.column()]
            return str(value)
        return None

    def headerData(self, section, orientation, role=Qt.ItemDataRole.DisplayRole):
        if role == Qt.ItemDataRole.DisplayRole:
            if orientation == Qt.Orientation.Horizontal:
                return str(self._df.columns[section])
            elif orientation == Qt.Orientation.Vertical:
                return str(self._df.index[section])
        return None


class LogMixin:
    def __init__(self):
        self.log_widget = None

    def set_log_widget(self, widget):
        self.log_widget = widget

    def log(self, message):
        if self.log_widget:
            self.log_widget.appendPlainText(message)


def log_action(func):
    def wrapper(self, *args, **kwargs):
        self.log(f"Действие: {func.__name__}")
        return func(self, *args, **kwargs)
    return wrapper


class DataApp(QWidget, LogMixin):
    def __init__(self):
        super().__init__()
        LogMixin.__init__(self)

        self.setWindowTitle("Приложение для визуализации данных")
        self.resize(1100, 800)

        self.data_tables = {
            "Электромобили": pd.read_csv("electric_vehicles_spec_2025.csv.csv")
        }

        self.tabs = QTabWidget()
        self.layout = QVBoxLayout()
        self.layout.addWidget(self.tabs)
        self.setLayout(self.layout)

        self.init_stat_tab()
        self.init_corr_tab()
        self.init_heatmap_tab()
        self.init_lineplot_tab()
        self.init_log_tab()

    @log_action
    def init_stat_tab(self):
        tab = QWidget()
        layout = QVBoxLayout()

        self.stat_table_select = QComboBox()
        self.stat_table_select.addItems(self.data_tables.keys())
        self.stat_table_select.currentTextChanged.connect(self.update_stat)

        self.df_stat = self.get_statistics(self.stat_table_select.currentText())
        self.stat_model = PandasModel(self.df_stat)
        self.stat_table = QTableView()
        self.stat_table.setModel(self.stat_model)
        self.stat_table.resizeColumnsToContents()
        self.stat_table.setAlternatingRowColors(True)

        layout.addWidget(QLabel("Выберите таблицу:"))
        layout.addWidget(self.stat_table_select)
        layout.addWidget(QLabel("Статистика по числовым столбцам:"))
        layout.addWidget(self.stat_table)

        tab.setLayout(layout)
        self.tabs.addTab(tab, "Статистика")

    @log_action
    def update_stat(self, table_name):
        df_stat = self.get_statistics(table_name)
        self.stat_model = PandasModel(df_stat)
        self.stat_table.setModel(self.stat_model)
        self.stat_table.resizeColumnsToContents()

    def get_statistics(self, table_name):
        df = self.data_tables.get(table_name)
        if df is not None:
            df_num = df.select_dtypes(include='number')
            desc = df_num.describe().transpose()
            desc = desc.round(2)  # округляем для наглядности
            return desc
        else:
            return pd.DataFrame([["Таблица не найдена"]], columns=["Ошибка"])

    @log_action
    def init_corr_tab(self):
        tab = QWidget()
        layout = QVBoxLayout()

        self.corr_canvas = FigureCanvas(plt.Figure(figsize=(9, 7)))
        layout.addWidget(self.corr_canvas)

        tab.setLayout(layout)
        self.tabs.addTab(tab, "Корреляция")

        self.plot_corr()

    @log_action
    def plot_corr(self):
        df = self.data_tables[list(self.data_tables.keys())[0]]
        df_num = df.select_dtypes(include=['number'])
        corr = df_num.corr()

        self.corr_canvas.figure.clear()
        ax = self.corr_canvas.figure.add_subplot()

        sns.heatmap(corr, annot=True, ax=ax, cmap='coolwarm')
        ax.set_title("Матрица корреляции")
        ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right', fontsize=9)
        ax.set_yticklabels(ax.get_yticklabels(), rotation=0, fontsize=9)
        self.corr_canvas.figure.tight_layout()
        self.corr_canvas.draw()

    @log_action
    def init_heatmap_tab(self):
        tab = QWidget()
        layout = QVBoxLayout()

        self.heatmap_canvas = FigureCanvas(plt.Figure(figsize=(9, 7)))
        layout.addWidget(self.heatmap_canvas)

        tab.setLayout(layout)
        self.tabs.addTab(tab, "Тепловая карта")

        self.plot_heatmap()

    @log_action
    def plot_heatmap(self):
        df = self.data_tables[list(self.data_tables.keys())[0]]
        df_num = df.select_dtypes(include=['number'])

        self.heatmap_canvas.figure.clear()
        ax = self.heatmap_canvas.figure.add_subplot()

        sns.heatmap(df_num.corr(), annot=True, cmap='viridis', ax=ax)
        ax.set_title("Тепловая карта числовых данных")
        ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right', fontsize=9)
        ax.set_yticklabels(ax.get_yticklabels(), rotation=0, fontsize=9)

        self.heatmap_canvas.figure.tight_layout()
        self.heatmap_canvas.draw()

    @log_action
    def init_lineplot_tab(self):
        tab = QWidget()
        layout = QVBoxLayout()

        self.lineplot_column_select = QComboBox()
        df = self.data_tables[list(self.data_tables.keys())[0]]
        num_cols = df.select_dtypes(include=['number']).columns.tolist()
        self.lineplot_column_select.addItems(num_cols)
        self.lineplot_column_select.currentTextChanged.connect(self.plot_lineplot)

        self.lineplot_canvas = FigureCanvas(plt.Figure(figsize=(9, 7)))

        layout.addWidget(QLabel("Выберите числовой столбец:"))
        layout.addWidget(self.lineplot_column_select)
        layout.addWidget(self.lineplot_canvas)

        tab.setLayout(layout)
        self.tabs.addTab(tab, "Линейный график")

        self.plot_lineplot(self.lineplot_column_select.currentText())

    @log_action
    def plot_lineplot(self, column):
        df = self.data_tables[list(self.data_tables.keys())[0]]

        self.lineplot_canvas.figure.clear()
        ax = self.lineplot_canvas.figure.add_subplot()
        ax.plot(df.index, df[column], marker='o')
        ax.set_title(f"Линейный график по столбцу {column}")
        ax.set_xlabel("Индекс")
        ax.set_ylabel(column)
        self.lineplot_canvas.figure.tight_layout()
        self.lineplot_canvas.draw()

    @log_action
    def init_log_tab(self):
        tab = QWidget()
        layout = QVBoxLayout()

        self.log_view = QPlainTextEdit()
        self.log_view.setReadOnly(True)

        layout.addWidget(QLabel("Лог действий пользователя"))
        layout.addWidget(self.log_view)
        tab.setLayout(layout)
        self.tabs.addTab(tab, "Лог")

        self.set_log_widget(self.log_view)


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = DataApp()
    window.show()
    sys.exit(app.exec())
