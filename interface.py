from PySide6.QtWidgets import (
    QApplication, QMainWindow, QTabWidget, QWidget,
    QVBoxLayout, QPushButton, QLabel, QFileDialog, QHBoxLayout, QLineEdit
)
from PySide6.QtCore import Qt
from PySide6.QtGui import QPixmap
import sys


class ImageSelectorApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Image Selector Interface")
        self.setGeometry(100, 100, 800, 600)  # Increased size to accommodate image display

        # Main layout with tabs
        self.tabs = QTabWidget()
        self.setCentralWidget(self.tabs)

        # Add tabs
        self.single_pair_tab = self.create_single_pair_tab()
        self.multiple_pairs_tab = self.create_multiple_pairs_tab()
        self.tabs.addTab(self.single_pair_tab, "Single Pair")
        self.tabs.addTab(self.multiple_pairs_tab, "Multiple Pairs")

    def create_single_pair_tab(self):
        # Create the Single Pair tab layout
        tab = QWidget()
        layout = QVBoxLayout()

        # Image selectors
        self.image1_label = QLabel("Image 1: Not selected")
        self.image1_label.setAlignment(Qt.AlignLeft)
        self.image2_label = QLabel("Image 2: Not selected")
        self.image2_label.setAlignment(Qt.AlignLeft)

        select_image1_btn = QPushButton("Select Image 1")
        select_image1_btn.clicked.connect(self.select_image1)

        select_image2_btn = QPushButton("Select Image 2")
        select_image2_btn.clicked.connect(self.select_image2)

        # Image display areas
        self.image1_display = QLabel("Image 1 Preview")
        self.image1_display.setAlignment(Qt.AlignCenter)
        self.image1_display.setFixedSize(300, 300)  # Fixed size for image preview
        self.image1_display.setStyleSheet("border: 1px solid black;")

        self.image2_display = QLabel("Image 2 Preview")
        self.image2_display.setAlignment(Qt.AlignCenter)
        self.image2_display.setFixedSize(300, 300)  # Fixed size for image preview
        self.image2_display.setStyleSheet("border: 1px solid black;")

        # Number of lines input
        self.num_lines_label = QLabel("Enter number of lines:")
        self.num_lines_input = QLineEdit()
        self.num_lines_input.setPlaceholderText("e.g., 5")

        # Solve button
        solve_btn = QPushButton("Solve")
        solve_btn.clicked.connect(self.solve_single_pair)

        # Arrange widgets in layout
        image_select_layout = QHBoxLayout()
        image_select_layout.addWidget(self.image1_display)
        image_select_layout.addWidget(self.image2_display)

        layout.addWidget(self.image1_label)
        layout.addWidget(select_image1_btn)
        layout.addWidget(self.image2_label)
        layout.addWidget(select_image2_btn)
        layout.addLayout(image_select_layout)
        layout.addWidget(self.num_lines_label)
        layout.addWidget(self.num_lines_input)
        layout.addWidget(solve_btn)

        tab.setLayout(layout)
        return tab

    def create_multiple_pairs_tab(self):
        # Create the Multiple Pairs tab layout
        tab = QWidget()
        layout = QVBoxLayout()

        instructions = QLabel("This tab is for handling multiple image pairs.")
        instructions.setAlignment(Qt.AlignLeft)

        # Solve button for multiple pairs
        solve_btn = QPushButton("Solve Multiple Pairs")
        solve_btn.clicked.connect(self.solve_multiple_pairs)

        layout.addWidget(instructions)
        layout.addWidget(solve_btn)
        tab.setLayout(layout)

        return tab

    def select_image1(self):
        # Open file dialog to select the first image
        file_name, _ = QFileDialog.getOpenFileName(self, "Select Image 1", "", "Images (*.png *.jpg *.jpeg *.bmp *.tiff)")
        if file_name:
            self.image1_label.setText(f"Image 1: {file_name}")
            pixmap = QPixmap(file_name).scaled(300, 300, Qt.KeepAspectRatio, Qt.SmoothTransformation)
            self.image1_display.setPixmap(pixmap)

    def select_image2(self):
        # Open file dialog to select the second image
        file_name, _ = QFileDialog.getOpenFileName(self, "Select Image 2", "", "Images (*.png *.jpg *.jpeg *.bmp *.tiff)")
        if file_name:
            self.image2_label.setText(f"Image 2: {file_name}")
            pixmap = QPixmap(file_name).scaled(300, 300, Qt.KeepAspectRatio, Qt.SmoothTransformation)
            self.image2_display.setPixmap(pixmap)

    def solve_single_pair(self):
        # Handle the solve logic for a single pair
        image1 = self.image1_label.text().replace("Image 1: ", "")
        image2 = self.image2_label.text().replace("Image 2: ", "")
        num_lines = self.num_lines_input.text()

        if "Not selected" in (image1, image2):
            self.statusBar().showMessage("Please select both images before solving.", 3000)
        elif not num_lines.isdigit():
            self.statusBar().showMessage("Please enter a valid number of lines.", 3000)
        else:
            self.statusBar().showMessage(f"Solving for images: {image1} and {image2} with {num_lines} lines.", 3000)

    def solve_multiple_pairs(self):
        # Handle the solve logic for multiple pairs (not implemented)
        self.statusBar().showMessage("Solve functionality for multiple pairs is not yet implemented.", 3000)


# Main execution
if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = ImageSelectorApp()
    window.show()
    sys.exit(app.exec())
