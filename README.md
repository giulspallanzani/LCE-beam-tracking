Cantilever Actuation Tracking and Analysis
This project provides a Python-based toolkit for tracking and analyzing the actuation of cantilever beams from video data. It leverages OpenCV and scientific Python libraries to extract quantitative metrics such as curvature, tip angle, and projections, with a focus on flexible configuration and interactive region-of-interest (ROI) selection.

🚀 Features

Interactive ROI Selection: Select regions of interest for both analysis and calibration directly from video frames.
HSV Color Filtering: Tune HSV thresholds for robust object detection, even with varying colors.
Focus Filtering: Identify and process only in-focus regions for improved accuracy.
Edge Detection & Clustering: Advanced image processing pipeline for extracting beam contours.
Curvature & Angle Calculation: Quantifies beam deformation and tip angle over time.
Exportable Results: Save processed data and plots for further analysis.
Highly Configurable: All key parameters are user-editable in config.py.


📂 Project Structure
cantilever-actuation-analysis/
│
├── main.py                # Main script to run the analysis
├── set_the_parameters.py  # HSV filter tuning utility
├── config.py              # User-editable configuration file
├── init.py                # Global variable initialization (do not modify)
├── DPI_calculation.py     # Functions for mm/pixel scaling (do not modify)
├── functions.py           # Core analysis functions (do not modify)
├── README.md              # Project documentation
└── requirements.txt       # Python dependencies


⚙️ Installation

1.Clone the repository:

git clone https://github.com/<your-username>/<repo-name>.git
cd <repo-name>

2.Install dependencies: All required Python packages are listed in requirements.txt.

pip install -r requirements.txt


▶️ Usage


1.Configure your analysis:

Edit config.py to set the video path, start time, frame skip, and HSV color thresholds.
Use set_the_parameters.py to interactively determine optimal HSV filter values for your video.

2.Run the main analysis:

python main.py

The script will guide you through ROI selection and other interactive steps.
Results and plots can be saved at the end of processing.

3.Optional utilities:

set_the_parameters.py: Helps you find HSV filter settings by selecting areas of interest in your video.
DPI_calculation.py: Assists with calibrating the pixel-to-mm scale using a known reference.


📝 Configuration

config.py:
Edit this file to set:

video_path: Path to your video file.
start_time_str: Start time for analysis (format: HH:MM:SS).
frame_skip: Number of frames to skip between analyses.
HSV color thresholds (lower, upper): For color-based filtering.
Other parameters for clustering, segmentation, and filtering.


🤝 Contributing
Contributions are welcome!
Please open issues or submit pull requests for improvements, bug fixes, or new features.

📜 License
This project is licensed under the BSD License. See LICENSE file for details.

🙏 Acknowledgements

Developed by Giulia Spallanzani.
Uses OpenCV, NumPy, SciPy, Matplotlib, Pandas, and scikit-learn.
