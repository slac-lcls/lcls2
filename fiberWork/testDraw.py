import pickle
import sys
import os
from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, 
                            QHBoxLayout, QPushButton, QLabel, QScrollArea, 
                            QFileDialog, QMessageBox, QMenuBar, QMenu, QAction,
                            QStatusBar, QToolBar)
from PyQt5.QtCore import Qt, QPoint, pyqtSignal
from PyQt5.QtGui import QPixmap, QPainter, QCursor

class DraggableImageLabel(QLabel):
    """A draggable image label widget"""
    
    delete_requested = pyqtSignal(object)  # Signal to request deletion
    
    def __init__(self, image_path, max_size=(200, 200), parent=None):
        super().__init__(parent)
        self.image_path = image_path
        self.max_size = max_size
        self.drag_start_position = QPoint()
        self.is_dragging = False
        
        # Load and set the image
        self.load_image()
        self.images_info={}
        # Enable mouse tracking and set cursor
        self.setMouseTracking(True)
        self.setCursor(QCursor(Qt.OpenHandCursor))
        
        # Set up context menu
        self.setContextMenuPolicy(Qt.CustomContextMenu)
        self.customContextMenuRequested.connect(self.show_context_menu)
        
    def load_image(self):
        """Load and resize the image"""
        try:
            pixmap = QPixmap(self.image_path)
            if not pixmap.isNull():
                # Scale image while maintaining aspect ratio
                scaled_pixmap = pixmap.scaled(
                    self.max_size[0], self.max_size[1], 
                    Qt.KeepAspectRatio, Qt.SmoothTransformation
                )
                self.setPixmap(scaled_pixmap)
                self.resize(scaled_pixmap.size())
                
                # Store original for potential resizing
                self.original_pixmap = pixmap
            else:
                self.setText("Failed to load image")
        except Exception as e:
            self.setText(f"Error: {str(e)}")
    
    def mousePressEvent(self, event):
        """Handle mouse press events"""
        if event.button() == Qt.LeftButton:
            self.drag_start_position = event.globalPos()
            self.setCursor(QCursor(Qt.ClosedHandCursor))
            self.is_dragging = True
            # Bring to front
            self.raise_()
        super().mousePressEvent(event)
    
    def mouseMoveEvent(self, event):
        """Handle mouse move events for dragging"""
        if (event.buttons() == Qt.LeftButton and self.is_dragging and
            not self.drag_start_position.isNull()):
            
            # Calculate the distance moved
            distance = (event.globalPos() - self.drag_start_position).manhattanLength()
            
            # Start drag if moved enough
            if distance >= QApplication.startDragDistance():
                # Calculate new position
                new_pos = self.mapToParent(event.pos() - self.mapFromGlobal(self.drag_start_position))
                
                # Keep within parent bounds
                parent_rect = self.parent().rect()
                new_x = max(0, min(new_pos.x(), parent_rect.width() - self.width()))
                new_y = max(0, min(new_pos.y(), parent_rect.height() - self.height()))
                
                self.move(new_x, new_y)
                self.drag_start_position = event.globalPos()
        
        super().mouseMoveEvent(event)
    
    def mouseReleaseEvent(self, event):
        """Handle mouse release events"""
        if event.button() == Qt.LeftButton:
            self.setCursor(QCursor(Qt.OpenHandCursor))
            
            print(self.image_path, "moved to", self.pos())  
            self.is_dragging = False
            self.drag_start_position = QPoint()
        super().mouseReleaseEvent(event)
    
    def show_context_menu(self, position):
        """Show context menu for image operations"""
        menu = QMenu(self)
        
        delete_action = menu.addAction("Delete Image")
        delete_action.triggered.connect(lambda: self.delete_requested.emit(self))
        
        menu.addSeparator()
        
        info_action = menu.addAction("Image Info")
        info_action.triggered.connect(self.show_info)
        
        menu.exec_(self.mapToGlobal(position))
    
    def show_info(self):
        """Show image information"""
        info = f"Path: {self.image_path}\n"
        info += f"Size: {self.size().width()} x {self.size().height()}"
        QMessageBox.information(self, "Image Info", info)

class ImageCanvas(QWidget):
    """Canvas widget to hold draggable images"""
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setMinimumSize(800, 600)
        self.setStyleSheet("background-color: white;")
        #self.images = []
        self.image_info = {}
        # Accept drops
        self.setAcceptDrops(True)
        # Load the dictionary from the file
        if os.path.exists('data.pkl'):
            with open('data.pkl', 'rb') as f:
                loaded_dict = pickle.load(f)
                self.image_info = loaded_dict
                
    def add_image(self, image_path, x=50, y=50):
        """Add a draggable image to the canvas"""
        id = len(self.image_info) + 1
        try:
            image_label = DraggableImageLabel(image_path, parent=self)
            image_label.move(x, y)
            image_label.show()
            
            image_label.delete_requested.connect(self.remove_image)
            self.image_info[id+1] = {"name": image_path, "position": (x, y), "label": image_label }

            # Save the dictionary to a file
            with open('data.pkl', 'wb') as f:
                pickle.dump(self.image_info, f)

            return image_label
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Could not add image: {str(e)}")
            return None
    
    def remove_image(self, image_label):
        key2remove=[]
        """Remove an image from the canvas"""
        reply = QMessageBox.question(self, "Delete Image", 
                                   "Are you sure you want to delete this image?",
                                   QMessageBox.Yes | QMessageBox.No)
        if reply == QMessageBox.Yes:
            
            for key in self.image_info.keys():    
                if self.image_info[key]["name"] == image_label.image_path:
                    key2remove.append(key)        
            for key in key2remove:del self.image_info[key]
            image_label.deleteLater()
    
    def clear_all_images(self):
        """Remove all images from the canvas"""
        reply = QMessageBox.question(self, "Clear All", 
                                    "Remove all images?",
                                    QMessageBox.Yes | QMessageBox.No)
        if reply == QMessageBox.Yes:
            for key in self.image_info:  # Copy list to avoid modification during iteration
                self.image_info[key]["label"].deleteLater()
            self.image_info={}
    
    def dragEnterEvent(self, event):
        """Handle drag enter events"""
        if event.mimeData().hasUrls():
            event.acceptProposedAction()
    
    def dropEvent(self, event):
        """Handle drop events for drag and drop functionality"""
        for url in event.mimeData().urls():
            file_path = url.toLocalFile()
            if file_path.lower().endswith(('.png', '.jpg', '.jpeg', '.gif', '.bmp', '.tiff')):
                # Add image at drop position
                self.add_image(file_path, event.pos().x(), event.pos().y())

class ImageManagerWindow(QMainWindow):
    """Main application window"""
    
    def __init__(self):
        super().__init__()
        self.setWindowTitle("PyQt5 Dynamic Image Manager")
        self.setGeometry(100, 100, 1000, 700)
        
        # Create central widget and layout
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        
        layout = QVBoxLayout(central_widget)
        
        # Create scroll area
        scroll_area = QScrollArea()
        scroll_area.setWidgetResizable(True)
        scroll_area.setHorizontalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        scroll_area.setVerticalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        
        # Create canvas
        self.canvas = ImageCanvas()
        scroll_area.setWidget(self.canvas)
        
        layout.addWidget(scroll_area)
        
        # Set up UI components
        self.setup_menu_bar()
        self.setup_toolbar()
        self.setup_status_bar()
        
        # Enable drag and drop on main window
        self.setAcceptDrops(True)
        
        self.update_status("Ready - Drag images to move them, right-click for options")
    
    def setup_menu_bar(self):
        """Set up the menu bar"""
        menubar = self.menuBar()
        
        # File menu
        file_menu = menubar.addMenu('File')
        
        add_action = QAction('Add Image', self)
        add_action.setShortcut('Ctrl+O')
        add_action.triggered.connect(self.add_single_image)
        file_menu.addAction(add_action)
        
        add_multiple_action = QAction('Add Multiple Images', self)
        add_multiple_action.setShortcut('Ctrl+Shift+O')
        add_multiple_action.triggered.connect(self.add_multiple_images)
        file_menu.addAction(add_multiple_action)
        
        file_menu.addSeparator()
        
        clear_action = QAction('Clear All', self)
        clear_action.setShortcut('Ctrl+X')
        clear_action.triggered.connect(self.canvas.clear_all_images)
        file_menu.addAction(clear_action)
        
        file_menu.addSeparator()
        
        exit_action = QAction('Exit', self)
        exit_action.setShortcut('Ctrl+Q')
        exit_action.triggered.connect(self.close)
        file_menu.addAction(exit_action)
        
        # Help menu
        help_menu = menubar.addMenu('Help')
        
        about_action = QAction('About', self)
        about_action.triggered.connect(self.show_about)
        help_menu.addAction(about_action)
    
    def setup_toolbar(self):
        """Set up the toolbar"""
        toolbar = QToolBar()
        self.addToolBar(toolbar)
        
        # Add image button
        add_btn = QPushButton("Add Image")
        add_btn.clicked.connect(self.add_single_image)
        toolbar.addWidget(add_btn)
        
        # Add multiple images button
        add_multiple_btn = QPushButton("Add Multiple")
        add_multiple_btn.clicked.connect(self.add_multiple_images)
        toolbar.addWidget(add_multiple_btn)
        
        toolbar.addSeparator()
        
        # Clear all button
        clear_btn = QPushButton("Clear All")
        clear_btn.clicked.connect(self.canvas.clear_all_images)
        toolbar.addWidget(clear_btn)
    
    def setup_status_bar(self):
        """Set up the status bar"""
        self.status_bar = QStatusBar()
        self.setStatusBar(self.status_bar)
    
    def add_single_image(self):
        """Add a single image file"""
        file_path, _ = QFileDialog.getOpenFileName(
            self, "Select an image", "",
            "Image files (*.png *.jpg *.jpeg *.gif *.bmp *.tiff);;All files (*.*)"
        )
        
        if file_path:
            image_label = self.canvas.add_image(file_path)
            if image_label:
                self.update_status(f"Added image: {os.path.basename(file_path)}")
    
    def add_multiple_images(self):
        """Add multiple image files"""
        file_paths, _ = QFileDialog.getOpenFileNames(
            self, "Select images", "",
            "Image files (*.png *.jpg *.jpeg *.gif *.bmp *.tiff);;All files (*.*)"
        )
        
        for i, file_path in enumerate(file_paths):
            # Offset each image slightly
            x_offset = (i % 5) * 50
            y_offset = (i // 5) * 50
            self.canvas.add_image(file_path, x=50 + x_offset, y=50 + y_offset)
        
        if file_paths:
            self.update_status(f"Added {len(file_paths)} images")
    
    def show_about(self):
        """Show about dialog"""
        QMessageBox.about(self, "About", 
                         "PyQt5 Dynamic Image Manager\n\n"
                         "Features:\n"
                         "• Drag and drop images to move them\n"
                         "• Right-click for context menu\n"
                         "• Add single or multiple images\n"
                         "• Drag files from file explorer")
    
    def update_status(self, message):
        """Update status bar message"""
        self.status_bar.showMessage(message)
    
    def dragEnterEvent(self, event):
        """Handle drag enter events on main window"""
        if event.mimeData().hasUrls():
            event.acceptProposedAction()
    
    def dropEvent(self, event):
        """Handle drop events on main window"""
        for url in event.mimeData().urls():
            file_path = url.toLocalFile()
            if file_path.lower().endswith(('.png', '.jpg', '.jpeg', '.gif', '.bmp', '.tiff')):
                self.canvas.add_image(file_path)
        
        self.update_status("Images added via drag and drop")

def main():
    app = QApplication(sys.argv)
    
    # Set application properties
    app.setApplicationName("PyQt5 Image Manager")
    app.setOrganizationName("Your Organization")
    
    # Create and show main window
    window = ImageManagerWindow()
    window.show()
    
    sys.exit(app.exec_())

if __name__ == "__main__":
    main()