import os
import sys

from PySide6.QtCore import Qt
from PySide6.QtGui import QPixmap
from PySide6.QtWidgets import (
    QApplication,
    QFileDialog,
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QProgressBar,
    QSizePolicy,
    QVBoxLayout,
    QWidget,
)
from qfluentwidgets import (
    BodyLabel,
    CaptionLabel,
    CardWidget,
    MessageBox,
    MSFluentWindow,
    PrimaryPushButton,
    PushButton,
    ScrollArea,
    Theme,
    TitleLabel,
    setTheme,
    setThemeColor,
)
from qfluentwidgets import FluentIcon as FIF

from model_handler import CLASS_NAMES, ModelHandler
from treatments_en import treatments_en
from treatments_sq import treatments_sq

MODEL_CELL_WIDTH = 320
IMAGE_CELL_WIDTH = 270

UI_LABELS = {
    "en": {
        "title": "Eye Disease Recognition",
        "load_model": "Load Models",
        "select_image": "Select Images",
        "predict": "Predict All",
        "original_image": "Original Image",
        "no_image": "No Images Selected",
        "status_ready": "Status: Ready",
        "status_loading": "Status: {count} model(s) loaded",
        "status_selected": "Status: {count} image(s) selected",
        "status_predicting": "Status: Predicting...",
        "status_complete": "Status: Prediction Complete",
        "status_error": "Status: Error during prediction.",
        "lang_btn": "Shqip",
        "treatment_title": "Treatment Information",
        "treatment_tab": "Treatment",
        "error_title": "Error",
        "file": "File",
        "waiting": "Waiting for prediction",
        "models": "Models",
        "images": "Images",
        "predictions": "Predictions",
        "model_column": "Model {number}",
    },
    "sq": {
        "title": "Identifikimi i Semundjeve te Syrit",
        "load_model": "Ngarko Modelet",
        "select_image": "Zgjidh Imazhet",
        "predict": "Parashiko te Gjitha",
        "original_image": "Imazhi Origjinal",
        "no_image": "Asnje imazh",
        "status_ready": "Statusi: Gati",
        "status_loading": "Statusi: {count} model(e) u ngarkuan",
        "status_selected": "Statusi: {count} imazh(e) u zgjodhen",
        "status_predicting": "Statusi: Duke parashikuar...",
        "status_complete": "Statusi: Perfundoi",
        "status_error": "Statusi: Gabim gjate parashikimit.",
        "lang_btn": "English",
        "treatment_title": "Informacioni i Trajtimit",
        "treatment_tab": "Trajtimi",
        "error_title": "Gabim",
        "file": "Skedari",
        "waiting": "Ne pritje te parashikimit",
        "models": "Modele",
        "images": "Imazhe",
        "predictions": "Parashikime",
        "model_column": "Modeli {number}",
    },
}


class AspectImageLabel(QLabel):
    def __init__(self, parent=None):
        super().__init__(parent)
        self._pixmap = None
        self.setMinimumSize(220, 220)
        self.setAlignment(Qt.AlignCenter)
        self.setStyleSheet("background: #f5f7fb; border: 1px solid #d8dee9; border-radius: 6px;")
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)

    def set_image(self, image_path):
        pixmap = QPixmap(image_path)
        self._pixmap = pixmap if not pixmap.isNull() else None
        self._update_pixmap()

    def resizeEvent(self, event):
        super().resizeEvent(event)
        self._update_pixmap()

    def _update_pixmap(self):
        if self._pixmap is None:
            self.clear()
            return

        scaled = self._pixmap.scaled(self.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation)
        self.setPixmap(scaled)


class PredictionBars(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.layout = QVBoxLayout(self)
        self.layout.setContentsMargins(0, 0, 0, 0)
        self.layout.setSpacing(7)
        self.bars = {}

        for class_name in CLASS_NAMES:
            bar = QProgressBar()
            bar.setRange(0, 10000)
            bar.setValue(0)
            bar.setAlignment(Qt.AlignCenter)
            bar.setFormat(f"{class_name} 0.00%")
            bar.setFixedHeight(23)
            bar.setStyleSheet(
                """
                QProgressBar {
                    border: 1px solid #d0d7de;
                    border-radius: 5px;
                    background: #f6f8fa;
                    color: #1f2328;
                    font-weight: 600;
                    text-align: center;
                }
                QProgressBar::chunk {
                    border-radius: 4px;
                    background-color: #1a73e8;
                }
                """
            )
            self.bars[class_name] = bar
            self.layout.addWidget(bar)

    def set_probabilities(self, probabilities):
        for class_name in CLASS_NAMES:
            probability = float(probabilities.get(class_name, 0.0))
            self.bars[class_name].setValue(int(round(probability * 10000)))
            self.bars[class_name].setFormat(f"{class_name} {probability * 100:.2f}%")


class ModelPredictionCell(QWidget):
    def __init__(self, loaded_model, model_number, labels, parent=None):
        super().__init__(parent)
        self.loaded_model = loaded_model
        self.result = None
        self.model_number = model_number

        self.setMinimumWidth(MODEL_CELL_WIDTH)
        self.layout = QVBoxLayout(self)
        self.layout.setContentsMargins(8, 0, 8, 0)
        self.layout.setSpacing(8)

        self.model_label = BodyLabel()
        self.model_label.setAlignment(Qt.AlignCenter)
        self.model_label.setStyleSheet("font-weight: bold; color: #1a73e8;")

        self.path_label = CaptionLabel(os.path.basename(str(loaded_model.path)))
        self.path_label.setAlignment(Qt.AlignCenter)
        self.path_label.setWordWrap(True)

        self.result_label = BodyLabel(labels["waiting"])
        self.result_label.setAlignment(Qt.AlignCenter)
        self.result_label.setStyleSheet("font-size: 16px; font-weight: bold; color: #1a73e8;")

        self.bars = PredictionBars()

        self.layout.addWidget(self.model_label)
        self.layout.addWidget(self.path_label)
        self.layout.addWidget(self.result_label)
        self.layout.addWidget(self.bars)
        self.layout.addStretch(1)
        self.set_labels(labels)

    def set_labels(self, labels):
        self.model_label.setText(labels["model_column"].format(number=self.model_number))
        if self.result is None:
            self.result_label.setText(labels["waiting"])

    def set_result(self, result):
        self.result = result
        label = result["label"].replace("___", " - ")
        confidence = result["confidence"] * 100
        self.result_label.setText(f"{label}\n({confidence:.2f}%)")
        self.bars.set_probabilities(result.get("probabilities", {}))


class ImagePredictionRow(CardWidget):
    def __init__(self, image_path, loaded_models, labels, parent=None):
        super().__init__(parent)
        self.image_path = image_path
        self.cells = []

        self.setMinimumHeight(335)
        self.layout = QGridLayout(self)
        self.layout.setContentsMargins(16, 16, 16, 16)
        self.layout.setHorizontalSpacing(18)
        self.layout.setVerticalSpacing(10)

        self.original_title = BodyLabel(labels["original_image"])
        self.original_title.setAlignment(Qt.AlignCenter)

        self.original_image = AspectImageLabel()
        self.original_image.set_image(image_path)
        self.original_image.setMinimumWidth(IMAGE_CELL_WIDTH)

        self.file_label = CaptionLabel(f'{labels["file"]}: {os.path.basename(image_path)}')
        self.file_label.setAlignment(Qt.AlignCenter)
        self.file_label.setWordWrap(True)

        self.layout.addWidget(self.original_title, 0, 0)
        self.layout.addWidget(self.original_image, 1, 0)
        self.layout.addWidget(self.file_label, 2, 0)
        self.layout.setColumnMinimumWidth(0, IMAGE_CELL_WIDTH)

        for index, loaded_model in enumerate(loaded_models, start=1):
            cell = ModelPredictionCell(loaded_model, index, labels)
            self.cells.append(cell)
            self.layout.addWidget(cell, 0, index, 3, 1)
            self.layout.setColumnMinimumWidth(index, MODEL_CELL_WIDTH)

        self.setMinimumWidth(IMAGE_CELL_WIDTH + (MODEL_CELL_WIDTH * max(1, len(loaded_models))) + 80)

    def set_labels(self, labels):
        self.original_title.setText(labels["original_image"])
        self.file_label.setText(f'{labels["file"]}: {os.path.basename(self.image_path)}')
        for cell in self.cells:
            cell.set_labels(labels)


class TreatmentInterface(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent=parent)
        self.setObjectName("treatmentInterface")
        self.layout = QVBoxLayout(self)
        self.layout.setContentsMargins(30, 30, 30, 30)
        self.layout.setSpacing(20)

        self.title_label = TitleLabel("Treatment Information")
        self.layout.addWidget(self.title_label)

        self.scroll_area = ScrollArea(self)
        self.scroll_area.setWidgetResizable(True)
        self.scroll_area.setStyleSheet("background: transparent; border: none;")

        self.content_widget = QWidget()
        self.content_widget.setStyleSheet("background: transparent;")
        self.content_layout = QVBoxLayout(self.content_widget)

        self.content_label = BodyLabel("No treatment information available yet. Please run a prediction.")
        self.content_label.setWordWrap(True)
        self.content_label.setStyleSheet("font-size: 16px; line-height: 1.5;")
        self.content_label.setAlignment(Qt.AlignTop | Qt.AlignLeft)

        self.content_layout.addWidget(self.content_label)
        self.content_layout.addStretch(1)

        self.scroll_area.setWidget(self.content_widget)
        self.layout.addWidget(self.scroll_area)

    def set_content(self, title, text):
        self.title_label.setText(title)
        self.content_label.setText(text)


class MainWindow(MSFluentWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Eye Disease Recognition")
        self.setMinimumSize(1180, 800)

        setTheme(Theme.LIGHT)
        setThemeColor("#1a73e8")

        self.model_handler = ModelHandler()
        self.selected_image_paths = []
        self.image_rows = []
        self.prediction_results = []
        self.current_lang = "en"

        self.init_ui()
        self.load_default_model()

    def init_ui(self):
        self.container = QWidget()
        self.container.setObjectName("homeInterface")
        self.main_layout = QVBoxLayout(self.container)
        self.main_layout.setContentsMargins(30, 30, 30, 30)
        self.main_layout.setSpacing(16)

        self.header_label = TitleLabel(UI_LABELS[self.current_lang]["title"])
        self.header_label.setAlignment(Qt.AlignCenter)
        self.main_layout.addWidget(self.header_label)

        controls_layout = QHBoxLayout()
        controls_layout.setSpacing(15)

        self.load_model_btn = PushButton(FIF.FOLDER, UI_LABELS[self.current_lang]["load_model"])
        self.load_model_btn.setFixedWidth(170)
        self.load_model_btn.clicked.connect(self.on_load_models)

        self.select_img_btn = PushButton(FIF.PHOTO, UI_LABELS[self.current_lang]["select_image"])
        self.select_img_btn.setFixedWidth(170)
        self.select_img_btn.clicked.connect(self.on_select_images)

        self.predict_btn = PrimaryPushButton(FIF.PLAY, UI_LABELS[self.current_lang]["predict"])
        self.predict_btn.setFixedWidth(160)
        self.predict_btn.clicked.connect(self.on_predict)
        self.predict_btn.setEnabled(False)

        self.lang_btn = PushButton(FIF.LANGUAGE, UI_LABELS[self.current_lang]["lang_btn"])
        self.lang_btn.setFixedWidth(130)
        self.lang_btn.clicked.connect(self.toggle_language)

        controls_layout.addStretch(1)
        controls_layout.addWidget(self.load_model_btn)
        controls_layout.addWidget(self.select_img_btn)
        controls_layout.addWidget(self.predict_btn)
        controls_layout.addWidget(self.lang_btn)
        controls_layout.addStretch(1)
        self.main_layout.addLayout(controls_layout)

        self.summary_card = CardWidget()
        summary_layout = QHBoxLayout(self.summary_card)
        summary_layout.setContentsMargins(16, 10, 16, 10)
        summary_layout.setSpacing(24)

        self.models_count_label = BodyLabel()
        self.images_count_label = BodyLabel()
        self.predictions_count_label = BodyLabel()
        self.model_names_label = CaptionLabel()
        self.model_names_label.setWordWrap(True)

        summary_layout.addWidget(self.models_count_label)
        summary_layout.addWidget(self.images_count_label)
        summary_layout.addWidget(self.predictions_count_label)
        summary_layout.addWidget(self.model_names_label, 1)
        self.main_layout.addWidget(self.summary_card)

        self.results_scroll = ScrollArea(self)
        self.results_scroll.setWidgetResizable(True)
        self.results_scroll.setStyleSheet("background: transparent; border: none;")

        self.results_widget = QWidget()
        self.results_layout = QVBoxLayout(self.results_widget)
        self.results_layout.setContentsMargins(0, 0, 0, 0)
        self.results_layout.setSpacing(14)

        self.empty_label = BodyLabel(UI_LABELS[self.current_lang]["no_image"])
        self.empty_label.setAlignment(Qt.AlignCenter)
        self.empty_label.setMinimumHeight(360)
        self.results_layout.addWidget(self.empty_label)
        self.results_layout.addStretch(1)

        self.results_scroll.setWidget(self.results_widget)
        self.main_layout.addWidget(self.results_scroll, 1)

        self.status_label = CaptionLabel(UI_LABELS[self.current_lang]["status_ready"])
        self.main_layout.addWidget(self.status_label)

        self.addSubInterface(self.container, FIF.HOME, "Home")

        self.treatment_interface = TreatmentInterface(self)
        self.addSubInterface(self.treatment_interface, FIF.INFO, UI_LABELS[self.current_lang]["treatment_tab"])
        self.update_summary()

    def toggle_language(self):
        self.current_lang = "sq" if self.current_lang == "en" else "en"
        self.update_ui_text()

    def update_ui_text(self):
        labels = UI_LABELS[self.current_lang]
        self.header_label.setText(labels["title"])
        self.load_model_btn.setText(labels["load_model"])
        self.select_img_btn.setText(labels["select_image"])
        self.predict_btn.setText(labels["predict"])
        self.lang_btn.setText(labels["lang_btn"])
        self.empty_label.setText(labels["no_image"])

        self.setMenuTabProperty(self.treatment_interface, "text", labels["treatment_tab"])
        self.treatment_interface.title_label.setText(labels["treatment_title"])

        for row in self.image_rows:
            row.set_labels(labels)

        self.update_summary()
        if self.prediction_results:
            self.update_treatment_content()

    def load_default_model(self):
        success, message = self.model_handler.load_default_model()
        if success:
            self.status_label.setText(
                UI_LABELS[self.current_lang]["status_loading"].format(count=len(self.model_handler.models))
            )
            self.rebuild_image_rows()
            self.update_predict_button()
            self.update_summary()
        else:
            self.status_label.setText(f"Status: Model not loaded. {message}")

    def on_load_models(self):
        file_paths, _ = QFileDialog.getOpenFileNames(
            self,
            "Select Model Files",
            "",
            "Keras Model (*.keras *.h5)",
        )
        if not file_paths:
            return

        success, message = self.model_handler.load_models(file_paths)
        if success:
            self.prediction_results = []
            self.rebuild_image_rows()
            self.update_predict_button()
            self.update_summary()
            self.status_label.setText(
                UI_LABELS[self.current_lang]["status_loading"].format(count=len(self.model_handler.models))
            )
            if "Some models loaded" in message:
                MessageBox("Warning", message, self).exec()
        else:
            MessageBox("Error", f"Failed to load models: {message}", self).exec()
            self.update_predict_button()
            self.update_summary()

    def on_select_images(self):
        file_paths, _ = QFileDialog.getOpenFileNames(
            self,
            "Select Image Files",
            "",
            "Images (*.png *.jpg *.jpeg *.JPG *.JPEG)",
        )
        if not file_paths:
            return

        self.selected_image_paths = file_paths
        self.prediction_results = []
        self.rebuild_image_rows()
        self.update_predict_button()
        self.update_summary()
        self.status_label.setText(
            UI_LABELS[self.current_lang]["status_selected"].format(count=len(self.selected_image_paths))
        )

    def on_predict(self):
        if not self.selected_image_paths or not self.model_handler.models:
            return

        self.status_label.setText(UI_LABELS[self.current_lang]["status_predicting"])
        self.prediction_results = []

        for row in self.image_rows:
            for cell in row.cells:
                result, error = self.model_handler.predict_with_model(cell.loaded_model, row.image_path)
                if error:
                    MessageBox(UI_LABELS[self.current_lang]["error_title"], error, self).exec()
                    self.status_label.setText(UI_LABELS[self.current_lang]["status_error"])
                    return

                cell.set_result(result)
                self.prediction_results.append((row.image_path, cell.loaded_model, result))

        self.update_treatment_content()
        self.update_summary()
        self.status_label.setText(UI_LABELS[self.current_lang]["status_complete"])

    def rebuild_image_rows(self):
        while self.results_layout.count():
            item = self.results_layout.takeAt(0)
            widget = item.widget()
            if widget is not None:
                widget.setParent(None)

        self.image_rows = []
        if not self.selected_image_paths:
            self.results_layout.addWidget(self.empty_label)
            self.results_layout.addStretch(1)
            self.results_widget.setMinimumWidth(0)
            return

        labels = UI_LABELS[self.current_lang]
        for image_path in self.selected_image_paths:
            row = ImagePredictionRow(image_path, self.model_handler.models, labels)
            self.image_rows.append(row)
            self.results_layout.addWidget(row)

        min_width = IMAGE_CELL_WIDTH + (MODEL_CELL_WIDTH * max(1, len(self.model_handler.models))) + 120
        self.results_widget.setMinimumWidth(min_width)
        self.results_layout.addStretch(1)

    def update_predict_button(self):
        self.predict_btn.setEnabled(bool(self.selected_image_paths) and bool(self.model_handler.models))

    def update_summary(self):
        labels = UI_LABELS[self.current_lang]
        model_count = len(self.model_handler.models)
        image_count = len(self.selected_image_paths)
        prediction_count = len(self.prediction_results)
        expected_predictions = model_count * image_count

        self.models_count_label.setText(f'{labels["models"]}: {model_count}')
        self.images_count_label.setText(f'{labels["images"]}: {image_count}')
        self.predictions_count_label.setText(f'{labels["predictions"]}: {prediction_count}/{expected_predictions}')

        model_names = ", ".join(model.name for model in self.model_handler.models)
        self.model_names_label.setText(model_names)

    def update_treatment_content(self):
        labels = UI_LABELS[self.current_lang]
        treatments = treatments_en if self.current_lang == "en" else treatments_sq
        sections = []

        for image_path, loaded_model, result in self.prediction_results:
            label = result["label"]
            confidence = result["confidence"] * 100
            description = treatments.get(
                label,
                "No treatment info available." if self.current_lang == "en" else "Nuk ka informacion per trajtimin.",
            )
            sections.append(
                f"{os.path.basename(image_path)}\n"
                f"Model: {loaded_model.name}\n"
                f"Prediction: {label} ({confidence:.2f}%)\n\n"
                f"{description.strip()}"
            )

        self.treatment_interface.set_content(labels["treatment_title"], "\n\n---\n\n".join(sections))


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec())
