"""
PhantomTouch Modern Dark Theme Stylesheet and Color Tokens.
"""

DARK_THEME_QSS = """
/* Base Window */
QMainWindow, QWidget#centralWidget {
    background-color: #0f1117;
    color: #e2e8f0;
    font-family: "Segoe UI", -apple-system, BlinkMacSystemFont, "Roboto", sans-serif;
    font-size: 13px;
}

/* Scroll areas */
QScrollArea {
    border: none;
    background: transparent;
}

/* Card Containers */
QFrame.card {
    background-color: #1a1d26;
    border: 1px solid #2d313e;
    border-radius: 12px;
    padding: 14px;
}

QFrame.card-highlight {
    background-color: #1f2330;
    border: 1px solid #4f46e5;
    border-radius: 12px;
    padding: 14px;
}

/* Headers */
QLabel.header-title {
    font-size: 20px;
    font-weight: 700;
    color: #ffffff;
    letter-spacing: 0.5px;
}

QLabel.section-header {
    font-size: 14px;
    font-weight: 600;
    color: #94a3b8;
    text-transform: uppercase;
    letter-spacing: 1px;
    margin-bottom: 4px;
}

QLabel.status-value {
    font-size: 14px;
    font-weight: 600;
    color: #f8fafc;
}

QLabel.subtext {
    font-size: 11px;
    color: #64748b;
}

/* Buttons */
QPushButton {
    background-color: #262a36;
    color: #f1f5f9;
    border: 1px solid #3b4252;
    border-radius: 8px;
    padding: 8px 16px;
    font-weight: 600;
    font-size: 13px;
    min-height: 22px;
}

QPushButton:hover {
    background-color: #323848;
    border-color: #4f46e5;
    color: #ffffff;
}

QPushButton:pressed {
    background-color: #1e222e;
}

QPushButton:disabled {
    background-color: #181a20;
    color: #475569;
    border-color: #232733;
}

/* Primary Action Buttons */
QPushButton.btn-primary {
    background: qlineargradient(x1:0, y1:0, x2:1, y2:0, stop:0 #6366f1, stop:1 #4f46e5);
    color: #ffffff;
    border: none;
    font-weight: 700;
}

QPushButton.btn-primary:hover {
    background: qlineargradient(x1:0, y1:0, x2:1, y2:0, stop:0 #7c3aed, stop:1 #6366f1);
}

QPushButton.btn-primary:pressed {
    background-color: #4338ca;
}

/* Success Button */
QPushButton.btn-success {
    background-color: #059669;
    color: #ffffff;
    border: none;
}

QPushButton.btn-success:hover {
    background-color: #10b981;
}

/* Danger Button */
QPushButton.btn-danger {
    background-color: #dc2626;
    color: #ffffff;
    border: none;
}

QPushButton.btn-danger:hover {
    background-color: #ef4444;
}

/* Controls */
QComboBox {
    background-color: #1e222e;
    color: #f1f5f9;
    border: 1px solid #333a4c;
    border-radius: 6px;
    padding: 6px 10px;
    min-height: 20px;
}

QComboBox:hover {
    border-color: #6366f1;
}

QComboBox::drop-down {
    border: none;
    width: 20px;
}

QComboBox QAbstractItemView {
    background-color: #1a1d26;
    color: #f1f5f9;
    border: 1px solid #333a4c;
    selection-background-color: #4f46e5;
    selection-color: #ffffff;
}

/* Sliders */
QSlider::groove:horizontal {
    height: 6px;
    background: #282d3c;
    border-radius: 3px;
}

QSlider::sub-page:horizontal {
    background: #6366f1;
    border-radius: 3px;
}

QSlider::handle:horizontal {
    background: #ffffff;
    border: 2px solid #6366f1;
    width: 14px;
    margin-top: -4px;
    margin-bottom: -4px;
    border-radius: 7px;
}

QSlider::handle:horizontal:hover {
    background: #e0e7ff;
}

/* Checkboxes */
QCheckBox {
    color: #e2e8f0;
    spacing: 8px;
    font-size: 13px;
}

QCheckBox::indicator {
    width: 18px;
    height: 18px;
    border-radius: 4px;
    border: 1px solid #3b4252;
    background-color: #1e222e;
}

QCheckBox::indicator:checked {
    background-color: #4f46e5;
    border-color: #6366f1;
}

/* Badges */
QLabel.badge {
    border-radius: 6px;
    padding: 3px 8px;
    font-weight: 700;
    font-size: 11px;
}

QLabel.badge-success {
    background-color: #064e3b;
    color: #34d399;
    border: 1px solid #059669;
}

QLabel.badge-warning {
    background-color: #78350f;
    color: #fbbf24;
    border: 1px solid #d97706;
}

QLabel.badge-danger {
    background-color: #7f1d1d;
    color: #f87171;
    border: 1px solid #b91c1c;
}

QLabel.badge-info {
    background-color: #1e1b4b;
    color: #a5b4fc;
    border: 1px solid #4f46e5;
}
"""
